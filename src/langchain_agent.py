"""LangChain agent integration wrapping the existing voice modules."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain.memory import ConversationBufferMemory
    from langchain.tools import Tool
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "LangChain is required for the agent integration. Install with `pip install langchain>=0.2`."
    ) from exc

from .conversation import ConversationManager
from .nlp_module import RetrievalResult
from .tts_module import TTSResponse

logger = logging.getLogger(__name__)


class SiliconFlowChatModel(BaseChatModel):
    """Adapter that allows using SiliconFlowClient through LangChain."""

    def __init__(self, client, *, temperature: float = 0.6, top_p: float = 0.9) -> None:  # noqa: ANN001
        super().__init__(temperature=temperature, top_p=top_p)
        object.__setattr__(self, "client", client)

    class Config:  # type: ignore[override]
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "siliconflow-chat"

    def _generate(  # type: ignore[override]
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        payload = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                role = "user"
            payload.append({"role": role, "content": message.content})

        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        logger.debug("Calling SiliconFlow via LangChain temperature=%s top_p=%s", temperature, top_p)
        text = self.client.chat(payload, temperature=temperature, top_p=top_p)
        if stop:
            for token in stop:
                if text.endswith(token):
                    text = text[: -len(token)]
        generation = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[generation])


@dataclass
class AgentRunResult:
    text: str
    audio_path: Optional[Path]
    citations: List[RetrievalResult]
    tool_messages: List[str]


class LangChainVoiceAgent:
    """LangChain agent that can orchestrate ASR/NLP/TTS tools dynamically."""

    def __init__(
        self,
        manager: ConversationManager,
        *,
        system_prompt: Optional[str] = None,
        retrieval_top_k: int = 4,
        auto_tts: bool = True,
        llm_temperature: float = 0.6,
        llm_top_p: float = 0.9,
    ) -> None:
        self.manager = manager
        self.retrieval_top_k = retrieval_top_k
        self.auto_tts = auto_tts
        self.system_prompt = system_prompt or (
            "You are a multilingual reasoning assistant. Choose tools when needed to transcribe audio,"
            " retrieve knowledge, answer user questions, and synthesize speech."
        )
        self._last_citations: List[RetrievalResult] = []
        self._last_audio_path: Optional[Path] = None
        self._tool_events: List[str] = []

        self.llm = SiliconFlowChatModel(
            manager.nlp.llm,
            temperature=llm_temperature,
            top_p=llm_top_p,
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.tools = self._build_tools()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(agent=agent, tools=self.tools, verbose=False, memory=self.memory)

    def _build_tools(self) -> List[Tool]:
        tools: List[Tool] = []

        if self.manager.asr:
            tools.append(
                Tool(
                    name="transcribe_audio",
                    description="Use this to transcribe a local audio file path into text before reasoning.",
                    func=self._speech_to_text,
                )
            )

        tools.append(
            Tool(
                name="rag_answer",
                description=(
                    "Use this tool to run retrieval-augmented generation against the knowledge base when the user"
                    " asks a factual question. Input should be the original user query."
                ),
                func=self._rag_answer,
            )
        )

        tools.append(
            Tool(
                name="synthesize_speech",
                description=(
                    "Convert the final assistant response into speech audio. Input must be the exact text you want"
                    " to speak."
                ),
                func=self._text_to_speech,
            )
        )
        return tools

    def _speech_to_text(self, audio_path: str) -> str:
        result = self.manager.asr.transcribe(audio_path)
        self._tool_events.append(f"transcribe_audio:{audio_path}")
        return result.text

    def _rag_answer(self, query: str) -> str:
        memory_context: Optional[str] = None
        if getattr(self.manager, "memory_manager", None):
            memory_context = self.manager.memory_manager.prepare_context(user_text=query)

        nlp_result = self.manager.nlp.answer(
            query,
            chat_history=self.manager.state.history_text(),
            memory_context=memory_context,
            top_k=self.retrieval_top_k,
        )
        self._last_citations = nlp_result.citations
        self._tool_events.append("rag_answer")
        citations_text = "\n".join(
            [
                f"- Score {citation.score:.3f}: {citation.text}"
                for citation in nlp_result.citations
            ]
        ) or "- No citations"
        return f"Answer:\n{nlp_result.answer}\n\nCitations:\n{citations_text}"

    def _text_to_speech(self, text: str) -> str:
        if not text.strip():
            raise ValueError("Cannot synthesize empty text")
        tts_response: TTSResponse = self.manager.tts.synthesize(text, self.manager.tts_output_dir)
        self._last_audio_path = tts_response.audio_path
        self._tool_events.append("synthesize_speech")
        return str(tts_response.audio_path)

    def run(
        self,
        *,
        text: Optional[str] = None,
        audio_path: Optional[str | Path] = None,
        auto_tts: Optional[bool] = None,
    ) -> AgentRunResult:
        if text and audio_path:
            raise ValueError("Provide either text or audio input, not both")

        self._tool_events.clear()
        self._last_citations = []
        self._last_audio_path = None

        if audio_path:
            asr_result = self.manager.asr.transcribe(audio_path)
            text = asr_result.text
            self._tool_events.append(f"auto_transcribe:{audio_path}")

        if not text or not text.strip():
            raise ValueError("Agent requires non-empty text input")

        result = self.executor.invoke({"input": text})
        assistant_text: str = result["output"]
        audio_path_out = self._last_audio_path

        should_tts = auto_tts if auto_tts is not None else self.auto_tts
        if should_tts and assistant_text and audio_path_out is None:
            tts_response = self.manager.tts.synthesize(assistant_text, self.manager.tts_output_dir)
            audio_path_out = tts_response.audio_path
            self._tool_events.append("auto_synthesize_speech")

        turn = self.manager.add_turn(
            user_text=text,
            assistant_text=assistant_text,
            citations=self._last_citations,
            audio=audio_path_out,
        )
        return AgentRunResult(
            text=turn.assistant_text,
            audio_path=turn.audio,
            citations=self._last_citations,
            tool_messages=list(self._tool_events),
        )

    def reset(self) -> None:
        """Clear conversation memory for a fresh agent session."""

        self.memory.clear()
        self.manager.state.turns.clear()
        self._last_audio_path = None
        self._last_citations = []
        self._tool_events.clear()






