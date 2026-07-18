"""Debrix — open-source instrumentation SDK for AI agents."""

from debrix import llm
from debrix.config import configure, force_flush
from debrix.mcp import MockableClient
from debrix.mocks import MockToolError
from debrix.semconv import SPAN_KINDS, Attr, Event, SpanKind, Stub
from debrix.span import DebrixSpan
from debrix.tracing import get_tracer, trace_agent, trace_span, trace_tool

__version__ = "0.1.0a4"

__all__ = [
    "__version__",
    "SpanKind",
    "Stub",
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
    "MockableClient",
    "MockToolError",
    "llm",
]
