from __future__ import annotations

import threading
import time
from typing import Generic, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    """Cache mémoire thread-safe avec TTL et taille maximale.

    Éviction simple : quand la taille max est atteinte, on purge d'abord les
    entrées expirées, puis, si nécessaire, les plus anciennes insérées.
    """

    __slots__ = ("_data", "_lock", "_ttl", "_max_entries")

    def __init__(self, ttl: float = 7 * 24 * 3600, max_entries: int = 200_000) -> None:
        self._data: dict[str, tuple[float, T]] = {}
        self._lock = threading.Lock()
        self._ttl = float(ttl)
        self._max_entries = int(max_entries)

    def get(self, key: str) -> T | None:
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at <= now:
                self._data.pop(key, None)
                return None
            return value

    def set(self, key: str, value: T) -> None:
        now = time.monotonic()
        with self._lock:
            if key not in self._data and len(self._data) >= self._max_entries:
                self._evict_locked(now)
            self._data[key] = (now + self._ttl, value)

    def _evict_locked(self, now: float) -> None:
        """Purge les entrées expirées, puis 10 % des plus anciennes si besoin."""
        expired = [k for k, (exp, _) in self._data.items() if exp <= now]
        for k in expired:
            self._data.pop(k, None)

        if len(self._data) < self._max_entries:
            return

        # dict conserve l'ordre d'insertion : les premières clés sont les plus anciennes.
        drop = max(1, self._max_entries // 10)
        for k in list(self._data.keys())[:drop]:
            self._data.pop(k, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
