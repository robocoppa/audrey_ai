"""
Audrey — LRU cache with TTL for response deduplication.
"""

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any

from config import CACHE_MAX, CACHE_TTL


class LRUCache:
    def __init__(self, max_entries: int, ttl: int):
        self._store: OrderedDict[str, tuple] = OrderedDict()
        self._max = max_entries
        self._ttl = ttl
        self._hits = 0
        self._misses = 0

    def _key(
        self,
        msgs: list[dict[str, Any]],
        model: str,
        temp: float,
        *,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: Any | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
    ) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "m": msgs,
                    "model": model,
                    "t": temp,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "stop": stop,
                    "fp": frequency_penalty,
                    "pp": presence_penalty,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def get(
        self,
        msgs: list[dict[str, Any]],
        model: str,
        temp: float,
        *,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: Any | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
    ) -> str | None:
        k = self._key(
            msgs, model, temp,
            max_tokens=max_tokens, top_p=top_p, stop=stop,
            frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
        )
        entry = self._store.get(k)
        if entry is None:
            self._misses += 1
            return None
        ts, txt = entry
        if time.time() - ts > self._ttl:
            del self._store[k]
            self._misses += 1
            return None
        self._store.move_to_end(k)
        self._hits += 1
        return txt

    def put(
        self,
        msgs: list[dict[str, Any]],
        model: str,
        temp: float,
        txt: str,
        *,
        max_tokens: int | None = None,
        top_p: float | None = None,
        stop: Any | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
    ) -> None:
        k = self._key(
            msgs, model, temp,
            max_tokens=max_tokens, top_p=top_p, stop=stop,
            frequency_penalty=frequency_penalty, presence_penalty=presence_penalty,
        )
        self._store[k] = (time.time(), txt)
        self._store.move_to_end(k)
        while len(self._store) > self._max:
            self._store.popitem(last=False)

    @property
    def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._store)}


# Module-level singleton
cache = LRUCache(CACHE_MAX, CACHE_TTL)
