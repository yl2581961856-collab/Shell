"""Mem0-backed memory implementations."""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional

from requests import RequestException, Session

from .base import BaseLongTermMemory, BaseShortTermMemory

logger = logging.getLogger(__name__)


class SlidingWindowMemory(BaseShortTermMemory):
    """In-memory sliding window storing the latest conversation turns."""

    def __init__(self, window_size: int = 6) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be greater than zero")
        self.window_size = window_size
        self._buffer: Deque[str] = deque(maxlen=window_size)

    def append(self, turn: Any) -> None:
        user_text = getattr(turn, "user_text", None)
        assistant_text = getattr(turn, "assistant_text", None)
        parts: List[str] = []
        if user_text:
            parts.append(f"User: {user_text}")
        if assistant_text:
            parts.append(f"Assistant: {assistant_text}")
        if parts:
            self._buffer.append("\n".join(parts))

    def get_context(self) -> Optional[str]:
        if not self._buffer:
            return None
        return "\n\n".join(self._buffer)


class Mem0LongTermMemory(BaseLongTermMemory):
    """Persist long-term memory via a locally hosted Mem0 service."""

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        llm_client: Optional[Any] = None,
    ) -> None:
        base_url = (config.get("base_url") or "").strip()
        if not base_url:
            raise ValueError("Mem0 configuration requires 'base_url'")
        self.base_url = base_url.rstrip("/")
        self.session: Session = Session()

        api_key = config.get("api_key")
        headers = config.get("headers", {})
        if api_key:
            headers = {**headers, "Authorization": f"Bearer {api_key}"}
        headers.setdefault("Content-Type", "application/json")
        self.session.headers.update(headers)

        self.add_path = self._normalise_path(
            config.get("add_path")
            or config.get("add_endpoint")
            or "/v1/memories"
        )
        self.search_path = self._normalise_path(
            config.get("search_path")
            or config.get("search_endpoint")
            or "/v1/memories/search"
        )
        self.user_id = config.get("user_id", "default")
        self.tags: List[str] = list(config.get("tags", []) or [])
        self.metadata_template: Dict[str, Any] = dict(config.get("metadata", {}) or {})
        self.extra_add_payload: Dict[str, Any] = dict(config.get("extra_add_payload", {}) or {})
        self.extra_search_payload: Dict[str, Any] = dict(config.get("extra_search_payload", {}) or {})
        self.top_k = int(config.get("top_k", 5))
        self.timeout = float(config.get("timeout", 30))

        self.llm_client = llm_client
        self.enable_summary = bool(config.get("summarize", True)) and llm_client is not None
        self.summary_system_prompt = config.get(
            "summary_system_prompt",
            "You extract durable, factual memories about the user from conversations.",
        )
        self.summary_prompt = config.get(
            "summary_prompt",
            (
                "Summarize the following exchange between the user and assistant into bullet points that capture"
                " lasting facts, preferences, or follow-up actions. Avoid redundant phrasing.\n\n{dialogue}"
            ),
        )
        self.summary_temperature = float(config.get("summary_temperature", 0.2))
        self.summary_top_p = float(config.get("summary_top_p", 0.8))

    def _normalise_path(self, path: str) -> str:
        path = (path or "").strip()
        if not path:
            return ""
        if path.startswith("http://") or path.startswith("https://"):
            return path
        if not path.startswith("/"):
            path = f"/{path}"
        return path

    def add_snapshot(self, turn: Any, summary: Optional[str] = None) -> None:
        memory_text = summary or self._build_memory_text(turn)
        if not memory_text:
            return

        if self.enable_summary:
            summarized = self._summarize(memory_text)
            if summarized:
                memory_text = summarized

        payload: Dict[str, Any] = {
            "memory": memory_text,
            "user_id": self.user_id,
        }
        if self.tags:
            payload["tags"] = self.tags
        if self.metadata_template:
            payload["metadata"] = self.metadata_template
        if self.extra_add_payload:
            payload.update(self.extra_add_payload)

        try:
            self._post(self.add_path, payload)
        except RequestException as exc:
            logger.error("Failed to persist memory turn to Mem0: %s", exc)

    def query(self, query_text: str) -> Optional[str]:
        payload: Dict[str, Any] = {
            "query": query_text,
            "user_id": self.user_id,
            "limit": self.top_k,
        }
        if self.tags:
            payload["tags"] = self.tags
        if self.extra_search_payload:
            payload.update(self.extra_search_payload)

        try:
            response = self._post(self.search_path, payload)
        except RequestException as exc:
            logger.error("Mem0 query failed: %s", exc)
            return None
        memories = self._extract_memories(response)
        if not memories:
            return None

        lines: List[str] = []
        for entry in memories[: self.top_k]:
            memory_text = self._extract_memory_text(entry)
            if not memory_text:
                continue
            score = entry.get("score")
            if isinstance(score, (float, int)):
                lines.append(f"[score={score:.3f}] {memory_text}")
            else:
                lines.append(memory_text)
        return "\n".join(lines) if lines else None

    def _post(self, path: str, payload: Dict[str, Any]) -> Optional[Any]:
        if not path:
            raise ValueError("Mem0 endpoint path cannot be empty")
        url = path if path.startswith("http://") or path.startswith("https://") else f"{self.base_url}{path}"
        response = self.session.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            try:
                return response.json()
            except ValueError:
                logger.debug("Mem0 response returned non-JSON payload despite JSON content type")
        return None

    def _build_memory_text(self, turn: Any) -> str:
        user_text = (getattr(turn, "user_text", None) or "").strip()
        assistant_text = (getattr(turn, "assistant_text", None) or "").strip()
        parts: List[str] = []
        if user_text:
            parts.append(f"User: {user_text}")
        if assistant_text:
            parts.append(f"Assistant: {assistant_text}")

        citations = getattr(turn, "citations", None)
        if citations:
            formatted = self._format_citations(citations)
            if formatted:
                parts.append(f"Citations: {formatted}")
        return "\n".join(parts)

    def _format_citations(self, citations: Iterable[Any]) -> str:
        formatted: List[str] = []
        for idx, citation in enumerate(citations, start=1):
            text = getattr(citation, "text", None) or citation.get("text") if isinstance(citation, dict) else None
            score = getattr(citation, "score", None) or citation.get("score") if isinstance(citation, dict) else None
            if not text:
                continue
            if isinstance(score, (int, float)):
                formatted.append(f"{idx}. {text} (score={score:.3f})")
            else:
                formatted.append(f"{idx}. {text}")
        return "; ".join(formatted)

    def _summarize(self, dialogue_text: str) -> Optional[str]:
        if not self.llm_client or not dialogue_text.strip():
            return None
        prompt = self.summary_prompt.format(dialogue=dialogue_text.strip())
        messages = [
            {"role": "system", "content": self.summary_system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            summary = self.llm_client.chat(
                messages,
                temperature=self.summary_temperature,
                top_p=self.summary_top_p,
            )
            return summary.strip()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Mem0 summarization failed: %s", exc)
            return None

    def _extract_memories(self, payload: Any) -> List[Dict[str, Any]]:
        if payload is None:
            return []
        if isinstance(payload, list):
            return [entry for entry in payload if isinstance(entry, dict)]
        if isinstance(payload, dict):
            for key in ("results", "memories", "data", "items", "rows"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [entry for entry in value if isinstance(entry, dict)]
            if all(key in payload for key in ("memory", "score")):
                return [payload]  # Single entry encoded as dict
        return []

    def _extract_memory_text(self, entry: Dict[str, Any]) -> Optional[str]:
        for key in ("memory", "content", "text", "value"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

