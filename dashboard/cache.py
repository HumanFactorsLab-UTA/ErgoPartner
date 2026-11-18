"""A time-to-live memo cache that does not require a UI framework.

``dashboard/data_loader.py`` scans the studies folder and was cached with
``@st.cache_data``. That worked, but it made filesystem discovery -- the layer
every interface needs -- depend on Streamlit, so an HTTP API could not read a
participant catalog without importing a web-app framework it never renders with.

The semantics kept are the ones the call sites actually rely on: memoize on the
argument tuple, expire after ``ttl`` seconds, and hand back a value the caller
cannot use to corrupt the next reader. The semantics deliberately dropped are
Streamlit's cross-session sharing and its widget-aware invalidation, neither of
which a filesystem scan needs.

Returned values are deep-copied on the way out. The cached objects here are small
catalogs of a few dozen dictionaries, so the copy is cheap next to the directory
walk it avoids, and it removes a whole class of bug where one caller's in-place
edit silently rewrites what the next caller sees.
"""

from __future__ import annotations

import copy
import functools
import threading
import time
from collections.abc import Callable, Hashable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

#: Every cache built here, so a process can clear them all at once the way
#: ``st.cache_data.clear()`` did.
_REGISTRY: list[Callable[[], None]] = []


def ttl_cache(ttl: float = 30.0, *, copy_result: bool = True) -> Callable[[F], F]:
    """Memoize a function on its arguments for ``ttl`` seconds.

    The wrapped function gains ``.cache_clear()`` and ``.clear()``. Both names
    exist deliberately: ``cache_clear`` matches ``functools.lru_cache`` and
    ``clear`` matches ``st.cache_data``, which is what the existing call sites
    were written against, so replacing the decorator is not also a rename.

    Arguments must be hashable; an unhashable argument bypasses the cache rather
    than raising, because a scanner called with an odd argument should still
    return the right answer.
    """

    def decorate(function: F) -> F:
        entries: dict[Hashable, tuple[float, Any]] = {}
        lock = threading.Lock()

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                key: Hashable = (args, tuple(sorted(kwargs.items())))
                hash(key)
            except TypeError:
                return function(*args, **kwargs)
            now = time.monotonic()
            with lock:
                hit = entries.get(key)
                if hit is not None and now - hit[0] < ttl:
                    return copy.deepcopy(hit[1]) if copy_result else hit[1]
            value = function(*args, **kwargs)
            with lock:
                entries[key] = (now, value)
            return copy.deepcopy(value) if copy_result else value

        def cache_clear() -> None:
            with lock:
                entries.clear()

        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
        wrapper.clear = cache_clear  # type: ignore[attr-defined]
        _REGISTRY.append(cache_clear)
        return wrapper  # type: ignore[return-value]

    return decorate


def clear_all_caches() -> None:
    """Drop every ``ttl_cache`` in this process."""

    for clear in _REGISTRY:
        clear()


__all__ = ["clear_all_caches", "ttl_cache"]
