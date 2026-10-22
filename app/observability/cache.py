"""Process-local LRU cache."""

import threading
from collections import OrderedDict
from typing import Generic, TypeVar

Key = TypeVar("Key")
Value = TypeVar("Value")


class LruCache(Generic[Key, Value]):
    def __init__(self, max_entries: int) -> None:
        self._max_entries = max_entries
        self._items: OrderedDict[Key, Value] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Key) -> Value | None:
        if self._max_entries < 1:
            return None
        with self._lock:
            value = self._items.get(key)
            if value is None:
                return None
            self._items.move_to_end(key)
            return value

    def put(self, key: Key, value: Value) -> None:
        if self._max_entries < 1:
            return
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self._max_entries:
                self._items.popitem(last=False)
