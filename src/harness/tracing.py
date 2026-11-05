"""Trace events: how the CLI shows what the memory machinery is doing.

Nodes call ``trace("event.name", key=value, ...)``. Inside a running graph this
goes to LangGraph's custom stream (``get_stream_writer``), which the CLI reads
with ``stream_mode="custom"``. Outside a graph (tests, CLI commands) events are
collected in a module-level list instead of being dropped.

Trace is NOT part of graph state on purpose: it must never be checkpointed.
"""

from __future__ import annotations

from typing import Any

from langgraph.config import get_stream_writer

_buffer: list[dict[str, Any]] = []


def trace(event: str, **data: Any) -> None:
    payload = {"event": event, **data}
    try:
        writer = get_stream_writer()
    except RuntimeError:
        writer = None
    if writer is not None:
        writer(payload)
    else:
        _buffer.append(payload)


def drain_buffer() -> list[dict[str, Any]]:
    events, _buffer[:] = list(_buffer), []
    return events
