"""Memory manager scaffold coordinating short-term and long-term memory."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import BaseLongTermMemory, BaseShortTermMemory

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Structured representation returned to upstream modules."""

    short_term_context: Optional[str] = None
    long_term_context: Optional[str] = None

    def merged_context(self) -> Optional[str]:
        parts = [ctx for ctx in (self.short_term_context, self.long_term_context) if ctx]
        return "\n\n".join(parts) if parts else None


class MemoryManager:
    """Lightweight coordinator for short-term and long-term memory components."""

    def __init__(
        self,
        short_term: Optional[BaseShortTermMemory] = None,
        long_term: Optional[BaseLongTermMemory] = None,
    ) -> None:
        self.short_term = short_term
        self.long_term = long_term

    def prepare_context(self, user_text: str) -> Optional[str]:
        """Return merged memory context for the incoming user query."""
        snapshot = MemorySnapshot()
        if self.short_term:
            snapshot.short_term_context = self.short_term.get_context()
        if self.long_term:
            snapshot.long_term_context = self.long_term.query(user_text)
        return snapshot.merged_context()

    def record_turn(self, turn: Any) -> None:
        """Persist turn information across memory components."""
        if self.short_term:
            self.short_term.append(turn)
        if self.long_term:
            try:
                self.long_term.add_snapshot(turn)
            except NotImplementedError:
                logger.debug("Long-term memory snapshot not implemented yet")


def build_memory_manager(config: Dict[str, Any]) -> Optional[MemoryManager]:
    """Factory returning a configured MemoryManager or ``None``.

    The default implementation wires placeholders only. Integrations can replace
    this function with Redis/Milvus-backed classes while preserving the public API.
    """

    memory_cfg = config.get("memory", {})
    if not memory_cfg.get("enabled"):
        return None

    # Placeholders: actual implementations should be provided during integration.
    short_term: Optional[BaseShortTermMemory] = None
    long_term: Optional[BaseLongTermMemory] = None

    if memory_cfg.get("short_term"):
        logger.warning("Short-term memory backend not implemented yet. Please plug in custom class.")
    if memory_cfg.get("long_term"):
        logger.warning("Long-term memory backend not implemented yet. Please plug in custom class.")

    return MemoryManager(short_term=short_term, long_term=long_term)
