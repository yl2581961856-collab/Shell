"""Base classes and protocols for the memory subsystem."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseShortTermMemory(ABC):
    """Abstract short-term memory interface handling recent turns."""

    @abstractmethod
    def append(self, turn: Any) -> None:
        """Persist a new conversation turn in short-term memory."""

    def get_context(self) -> Optional[str]:
        """Return a textual context representation (if available)."""
        return None


class BaseLongTermMemory(ABC):
    """Abstract long-term memory interface for durable facts/preferences."""

    @abstractmethod
    def add_snapshot(self, turn: Any, summary: Optional[str] = None) -> None:
        """Persist an aggregated snapshot of the provided turn."""

    def query(self, query_text: str) -> Optional[str]:
        """Retrieve relevant long-term info for the given query."""
        return None
