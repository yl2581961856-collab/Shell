"""Language understanding, retrieval, and generation module."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("`requests` is required for DeepSeek HTTP calls. Install with `pip install requests`.") from exc

try:  # pragma: no cover - optional at runtime
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:  # pragma: no cover - optional at runtime
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    text: str
    score: float
    metadata: dict


@dataclass
class NLPResult:
    answer: str
    citations: List[RetrievalResult]


class KnowledgeBase:
    """Simple FAISS-backed retriever."""

    def __init__(
        self,
        embedding_model: str,
        index_path: str | Path,
    ) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "`sentence-transformers` is required for KnowledgeBase. Install with `pip install sentence-transformers`."
            )
        if faiss is None:
            raise RuntimeError("`faiss-cpu` is required for KnowledgeBase. Install with `pip install faiss-cpu`.")

        self.embedding_model = SentenceTransformer(embedding_model)
        self.index_path = Path(index_path)
        self.documents: List[str] = []
        self.metadatas: List[dict] = []
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)
        if self.index_path.exists():
            self._load()
        logger.debug("KnowledgeBase initialized with %d docs", len(self.documents))

    def _load(self) -> None:
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.documents = payload.get("documents", [])
        self.metadatas = payload.get("metadatas", [{} for _ in self.documents])
        if self.documents:
            import numpy as np

            matrix = self.embedding_model.encode(self.documents, normalize_embeddings=True)
            self.index.add(matrix.astype("float32"))
        logger.info("Loaded %d documents into FAISS index", len(self.documents))

    def _persist(self) -> None:
        payload = {
            "documents": self.documents,
            "metadatas": self.metadatas,
        }
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        logger.debug("Persisted knowledge base with %d docs", len(self.documents))

    def add_documents(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[dict]] = None,
        persist: bool = True,
    ) -> None:
        if not texts:
            return
        import numpy as np

        embeddings = self.embedding_model.encode(list(texts), normalize_embeddings=True)
        self.index.add(embeddings.astype("float32"))
        self.documents.extend(texts)
        if metadatas:
            self.metadatas.extend(list(metadatas))
        else:
            self.metadatas.extend({} for _ in texts)
        if persist:
            self._persist()
        logger.info("Added %d documents to knowledge base", len(texts))

    def search(self, query: str, top_k: int = 4) -> List[RetrievalResult]:
        if self.index.ntotal == 0:
            return []
        query_emb = self.embedding_model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(query_emb.astype("float32"), top_k)
        results: List[RetrievalResult] = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            results.append(
                RetrievalResult(
                    text=self.documents[idx],
                    score=float(scores[0][i]),
                    metadata=self.metadatas[idx],
                )
            )
        return results


class SiliconFlowClient:
    """HTTP client for OpenAI-compatible chat APIs (SiliconFlow, Qwen, etc.)."""

    def __init__(
        self,
        api_key: Optional[str],
        model: str,
        base_url: str = "https://api.siliconflow.cn",
        *,
        timeout: float = 120,
        headers: Optional[dict] = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.model = model
        cleaned_base = (base_url or "https://api.siliconflow.cn").strip()
        self.base_url = cleaned_base.rstrip("/") or "https://api.siliconflow.cn"
        self.timeout = float(timeout or 120)
        self.default_headers = dict(headers or {})

    def chat(
        self,
        messages: Sequence[dict],
        *,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
    ) -> str:
        if self.base_url.endswith("/v1"):
            url = f"{self.base_url}/chat/completions"
        else:
            url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json", **self.default_headers}
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        if response.status_code >= 400:
            raise RuntimeError(f"Chat backend error {response.status_code}: {response.text}")
        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError(f"Chat backend returned invalid JSON: {response.text[:200]}") from exc
        return data["choices"][0]["message"]["content"].strip()


class NLPModule:
    """Combines retrieval-augmented generation with an OpenAI-compatible chat backend."""

    def __init__(
        self,
        llm_client: SiliconFlowClient,
        knowledge_base: Optional[KnowledgeBase] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.llm = llm_client
        self.knowledge_base = knowledge_base
        self.system_prompt = system_prompt or (
            "You are a helpful bilingual assistant that reasons step-by-step before answering."
        )

    def _build_messages(
        self,
        query: str,
        context: Optional[str],
        memory_context: Optional[str],
        retrieved: Iterable[RetrievalResult],
        *,
        system_prompt: Optional[str] = None,
    ) -> List[dict]:
        prompt = system_prompt or self.system_prompt
        messages: List[dict] = [
            {"role": "system", "content": prompt},
        ]
        if context:
            messages.append({"role": "user", "content": f"Conversation history:\n{context}"})
        if memory_context:
            messages.append({"role": "user", "content": f"Memory notes:\n{memory_context}"})
        retrieved = list(retrieved)
        if retrieved:
            knowledge = "\n\n".join(f"Doc {i+1}: {doc.text}" for i, doc in enumerate(retrieved))
            messages.append({"role": "user", "content": f"Relevant knowledge:\n{knowledge}"})
        messages.append({"role": "user", "content": query})
        return messages

    def answer(
        self,
        query: str,
        *,
        chat_history: Optional[str] = None,
        memory_context: Optional[str] = None,
        top_k: int = 4,
        system_prompt: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> NLPResult:
        retrieved: List[RetrievalResult] = []
        if self.knowledge_base:
            retrieved = self.knowledge_base.search(query, top_k=top_k)
        messages = self._build_messages(
            query,
            chat_history,
            memory_context,
            retrieved,
            system_prompt=system_prompt,
        )
        llm_params: Dict[str, Any] = {}
        if generation_kwargs:
            if generation_kwargs.get("temperature") is not None:
                llm_params["temperature"] = float(generation_kwargs["temperature"])
            if generation_kwargs.get("top_p") is not None:
                llm_params["top_p"] = float(generation_kwargs["top_p"])
            if generation_kwargs.get("max_tokens") is not None:
                llm_params["max_tokens"] = int(generation_kwargs["max_tokens"])

        answer = self.llm.chat(messages, **llm_params)
        return NLPResult(answer=answer, citations=retrieved)
