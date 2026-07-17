"""Debrix — open-source instrumentation SDK for AI agents."""

from debrix.config import configure, force_flush
from debrix.semconv import SPAN_KINDS, Attr, Event, SpanKind
from debrix.span import DebrixSpan
from debrix.tracing import get_tracer, trace_agent, trace_span, trace_tool

__version__ = "0.1.0a3"

__all__ = [
    "__version__",
    "SpanKind",
    "Attr",
    "Event",
    "SPAN_KINDS",
    "configure",
    "force_flush",
    "DebrixSpan",
    "get_tracer",
    "trace_agent",
    "trace_tool",
    "trace_span",
]
