"""Cooperative cancellation for in-flight graph runs.

Why cooperative, and not ``asyncio.Task.cancel()``
--------------------------------------------------
Every node in this project is a **synchronous** function (``nodes.py`` has 93
``def`` and 0 ``async def``), so LangGraph executes them on a worker thread via
``asyncio.to_thread``. Cancelling the awaiting asyncio task therefore does
*not* stop the work: the thread keeps running, the provider call still leaves,
still gets paid for, and its writes can land in the checkpointer after the
socket has already moved on. The only way to actually stop a run is for the
running code to check a flag and raise -- which is what this module provides.

Shape of the mechanism
----------------------
* A run is identified by its **LangGraph thread id** -- the same value
  ``main.py`` already puts in ``config["configurable"]["thread_id"]``, and the
  only identifier the websocket has for "the thing that is currently running".
* The websocket handler marks the run key on a :class:`~contextvars.ContextVar`
  (:func:`setRunKey`) before invoking the graph, exactly the way ``setSpendSession``
  in :mod:`graphic_designer.utils` already does. ``asyncio.to_thread`` copies the
  context into node threads, and the ``ThreadPoolExecutor`` call sites inside the
  critique pipeline copy it explicitly with ``contextvars.copy_context()``, so
  the key reaches every place work actually happens.
* :func:`requestCancel` (called from the websocket reader task, on the event
  loop, while the run is mid-flight) flips a flag in a module-level registry.
* :func:`checkCancelled` reads the ContextVar, looks the key up in the registry
  and raises :class:`RunCancelled` if it is set. It is called from the two
  chokepoints every unit of work passes through -- ``@profile`` (wraps every
  node) and ``apiLog(status='start')`` (immediately before every provider call
  leaves) -- so a cancel takes effect at the next node boundary or, at worst,
  before the *next* provider call is paid for.

:class:`RunCancelled` inherits :class:`BaseException` for the same reason
:class:`~graphic_designer.utils.SpendLimitExceeded` does: every image executor
and every model fallback loop in this codebase wraps its call in a bare
``except Exception`` and treats whatever it catches as "this model failed, try
the next one". A cancellation built on ``Exception`` would be swallowed there
and *feed* the retry chain it exists to stop.
"""

from __future__ import annotations

import contextvars
import threading
import time

__all__ = [
    "RunCancelled",
    "setRunKey",
    "resetRunKey",
    "currentRunKey",
    "activeRun",
    "requestCancel",
    "clearCancel",
    "isCancelled",
    "checkCancelled",
    "cancelState",
]


class RunCancelled(BaseException):
    """Raised inside a running graph node when the user cancelled the run.

    Inherits :class:`BaseException`, **not** ``Exception``, on purpose -- see
    the module docstring. Carries ``scope`` ("generation" | "suggestion" |
    ...) and ``run_key`` so the websocket layer can name what it stopped in the
    frame it sends back.
    """

    def __init__(self, run_key: str = "", scope: str = "generation",
                 reason: str = "", where: str = ""):
        self.run_key = run_key
        self.scope = scope or "generation"
        self.reason = reason or "cancelled by user"
        self.where = where
        super().__init__(
            f"run {run_key or '-'} cancelled by user "
            f"(scope={self.scope}{', at ' + where if where else ''})"
        )


# run key (== LangGraph thread id) of the run executing on this context.
# A ContextVar rather than a thread-local so it survives the hop into
# LangGraph's node worker threads and the critique pipeline's own executors,
# which a thread-local cannot -- same reasoning as `_current_node` /
# `_current_session` in utils.py.
_run_key = contextvars.ContextVar("kognie_run_key", default="")

_lock = threading.Lock()
# run key -> {"cancelled": bool, "scope": str, "reason": str,
#             "requested_at": float, "active": bool, "started_at": float}
_runs: dict[str, dict] = {}

# A cancel can arrive in the tiny window between the reader task seeing the
# frame and the graph actually starting (or just after a run ends). Entries are
# kept this long so such a request is not silently dropped, then GC'd.
_ENTRY_TTL_S = 600.0


def _gc(now: float) -> None:
    """Drop finished, unreferenced entries. Caller holds ``_lock``."""
    stale = [
        k for k, v in _runs.items()
        if not v.get("active")
        and now - max(v.get("requested_at", 0.0), v.get("started_at", 0.0)) > _ENTRY_TTL_S
    ]
    for k in stale:
        _runs.pop(k, None)


def setRunKey(run_key: str):
    """Attribute work on this context to ``run_key``; returns a reset token."""
    return _run_key.set(run_key or "")


def resetRunKey(token) -> None:
    try:
        _run_key.reset(token)
    except Exception:
        pass


def currentRunKey() -> str:
    """The run key on this context, or ``""`` outside a run."""
    try:
        return _run_key.get()
    except LookupError:
        return ""


class activeRun:
    """Context manager marking ``run_key`` as executing right now.

    Entering clears any *stale* cancel flag (one left over from a previous,
    already-finished run on the same thread id) so a fresh generation is never
    killed the moment it starts, while a cancel that arrived in the last
    moments before the invoke -- i.e. genuinely aimed at this run -- is kept.

    ``fresh=True`` (the launch of a brand new run) always starts clean;
    ``fresh=False`` is used for the auto-continue resumes inside one logical
    run, which must inherit a cancel requested while the previous leg ran.
    """

    def __init__(self, run_key: str, fresh: bool = False):
        self.run_key = run_key or ""
        self.fresh = fresh
        self._token = None

    def __enter__(self):
        now = time.time()
        if self.run_key:
            with _lock:
                _gc(now)
                entry = _runs.setdefault(self.run_key, {})
                if self.fresh:
                    entry["cancelled"] = False
                    entry["scope"] = ""
                    entry["reason"] = ""
                    entry["requested_at"] = 0.0
                entry["active"] = True
                entry["started_at"] = now
        self._token = setRunKey(self.run_key)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.run_key:
            with _lock:
                entry = _runs.get(self.run_key)
                if entry is not None:
                    entry["active"] = False
        resetRunKey(self._token)
        return False


def requestCancel(run_key: str, scope: str = "generation", reason: str = "") -> dict:
    """Flag ``run_key`` for cancellation. Safe to call from the event loop.

    Returns a snapshot dict: ``{"was_running": bool, "scope": str, ...}`` so the
    caller can tell the frontend whether it actually stopped something or the
    run had already finished.
    """
    if not run_key:
        return {"was_running": False, "scope": scope, "reason": reason}
    now = time.time()
    with _lock:
        _gc(now)
        entry = _runs.setdefault(run_key, {"active": False, "started_at": 0.0})
        was_running = bool(entry.get("active"))
        entry["cancelled"] = True
        entry["scope"] = scope or "generation"
        entry["reason"] = reason or "cancelled by user"
        entry["requested_at"] = now
    print(f"[cancel] requested for run {run_key} (scope={scope}, running={was_running})")
    return {"was_running": was_running, "scope": scope or "generation",
            "reason": reason or "cancelled by user"}


def clearCancel(run_key: str) -> None:
    """Forget any cancellation recorded for ``run_key``."""
    if not run_key:
        return
    with _lock:
        entry = _runs.get(run_key)
        if entry is not None:
            entry["cancelled"] = False
            entry["scope"] = ""
            entry["reason"] = ""
            entry["requested_at"] = 0.0


def cancelState(run_key: str = None) -> dict:
    """The recorded cancel state for a run (``{}`` when nothing is recorded)."""
    key = run_key if run_key is not None else currentRunKey()
    if not key:
        return {}
    with _lock:
        entry = _runs.get(key)
        return dict(entry) if entry else {}


def isCancelled(run_key: str = None) -> bool:
    """True when a cancel has been requested for this run. Never raises."""
    key = run_key if run_key is not None else currentRunKey()
    if not key:
        return False
    with _lock:
        entry = _runs.get(key)
        return bool(entry and entry.get("cancelled"))


def checkCancelled(where: str = "") -> None:
    """Raise :class:`RunCancelled` if this run has been cancelled.

    The cancellation chokepoint. Cheap enough (one ContextVar read, one dict
    lookup under an uncontended lock) to sit in front of every node and every
    provider call.
    """
    key = currentRunKey()
    if not key:
        return
    with _lock:
        entry = _runs.get(key)
        if not (entry and entry.get("cancelled")):
            return
        scope = entry.get("scope") or "generation"
        reason = entry.get("reason") or "cancelled by user"
    print(f"[cancel] stopping run {key} at {where or '?'} (scope={scope})")
    raise RunCancelled(run_key=key, scope=scope, reason=reason, where=where)
