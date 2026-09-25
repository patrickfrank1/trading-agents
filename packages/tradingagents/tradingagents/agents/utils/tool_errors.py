"""Shared tool error handling: logging + post-run error collection.

Every ``@tool`` function in the pipeline should be wrapped with
``@safe_tool`` (applied *below* ``@tool``) so that:

1. Exceptions are caught and never crash the graph run.
2. The error is logged via Python ``logging`` at WARNING level so it
   appears in the console / log file immediately.
3. A structured record is appended to the thread-safe
   :class:`ToolErrorCollector` singleton, which the graph drains after
   each run and persists into the state log for post-run debugging.

Usage::

    from langchain_core.tools import tool
    from tradingagents.agents.utils.tool_errors import safe_tool

    @tool
    @safe_tool
    def get_foo(ticker: Annotated[str, "ticker"]) -> str:
        \"""Docstring.\"""
        ...  # may raise; safe_tool catches it
"""

from __future__ import annotations

import functools
import logging
import threading
import traceback
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger("tradingagents.tools")


def _safe_repr(obj: Any, limit: int = 500) -> str:
    """Best-effort repr that never raises."""
    try:
        r = repr(obj)
        return r[:limit] + ("…" if len(r) > limit else "")
    except Exception:
        return "<unreprable>"


class ToolErrorCollector:
    """Thread-safe accumulator for tool errors during a graph run.

    A single module-level instance (:data:`tool_error_collector`) is shared
    across the entire process. The graph clears it at the start of each run
    and drains it at the end to persist errors into the state log.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._errors: list[dict[str, Any]] = []

    def record(
        self,
        tool_name: str,
        args: tuple,
        kwargs: dict,
        exc: BaseException,
    ) -> None:
        """Record a tool failure and log it immediately."""
        entry = {
            "tool": tool_name,
            "args": _safe_repr(args),
            "kwargs": _safe_repr(kwargs),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat(),
        }
        with self._lock:
            self._errors.append(entry)
        logger.warning(
            "Tool '%s' failed: %s",
            tool_name,
            entry["error"],
            exc_info=True,
        )

    def drain(self) -> list[dict[str, Any]]:
        """Return all recorded errors and clear the collector."""
        with self._lock:
            errors = list(self._errors)
            self._errors.clear()
        return errors

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a copy of recorded errors without clearing."""
        with self._lock:
            return list(self._errors)

    def clear(self) -> None:
        """Discard all recorded errors."""
        with self._lock:
            self._errors.clear()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._errors)


#: Module-level singleton — the graph clears/drains this per run.
tool_error_collector = ToolErrorCollector()


def safe_tool(func: Callable) -> Callable:
    """Decorator that catches exceptions, logs them, records to the
    error collector, and returns a clean error string to the LLM.

    Apply **below** ``@tool`` so that langchain's ``@tool`` sees the
    wrapped function's (preserved) signature and docstring::

        @tool
        @safe_tool
        def my_tool(ticker: Annotated[str, "ticker"]) -> str:
            \"""Docstring.\"""
            ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            tool_error_collector.record(
                getattr(func, "__name__", "unknown_tool"),
                args,
                kwargs,
                exc,
            )
            return f"[Tool Error] {getattr(func, '__name__', 'unknown_tool')}: {exc}"

    return wrapper
