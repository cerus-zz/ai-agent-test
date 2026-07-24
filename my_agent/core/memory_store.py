"""Simple in-memory store for conversation context (can be swapped for Redis)."""

from __future__ import annotations

from typing import Any

from my_agent.core.base import BaseMemory


class DictMemory(BaseMemory):
    """Simple dict-based memory. Swap for RedisMemory in production."""

    def __init__(self):
        self._store: dict[str, Any] = {}

    def add(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def clear(self) -> None:
        self._store.clear()

    def __contains__(self, key: str) -> bool:
        return key in self._store
