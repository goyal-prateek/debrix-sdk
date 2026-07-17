"""Debrix semantic conventions — shared span-kind and attribute-key contract.

This module is the Python source of truth for the Debrix semantic model. It
mirrors ``docs/Semantic_Model.md`` and the Rust ``semconv.rs`` in the desktop
app. Change the spec doc first, then keep all three in lockstep.
"""

from __future__ import annotations

__all__ = ["SpanKind", "Attr", "Event", "SPAN_KINDS"]


class SpanKind:
    """Allowed values for the ``debrix.span.kind`` attribute."""

    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    MCP = "mcp"
    MEMORY = "memory"
    EVALUATION = "evaluation"
    HUMAN = "human"
    CUSTOM = "custom"


class Attr:
    """Debrix-owned attribute keys. All use the ``debrix.`` prefix."""

    # Identity
    SPAN_KIND = "debrix.span.kind"
    AGENT_NAME = "debrix.agent.name"
    TOOL_NAME = "debrix.tool.name"
    MCP_SERVER = "debrix.mcp.server"
    MCP_TOOL = "debrix.mcp.tool"

    # Messages / response (legacy inline + blob model)
    MESSAGES = "debrix.messages"
    MESSAGES_PREVIEW = "debrix.messages.preview"
    MESSAGES_BLOB_REF = "debrix.messages.blob_ref"
    MESSAGES_BYTES = "debrix.messages.bytes"
    MESSAGES_CHARS = "debrix.messages.chars"
    MESSAGES_COUNT = "debrix.messages.count"
    MESSAGES_TRUNCATED = "debrix.messages.truncated"
    MESSAGES_CAPTURE_ERROR = "debrix.messages.capture_error"

    RESPONSE = "debrix.response"
    RESPONSE_PREVIEW = "debrix.response.preview"
    RESPONSE_BLOB_REF = "debrix.response.blob_ref"
    RESPONSE_BYTES = "debrix.response.bytes"
    RESPONSE_CHARS = "debrix.response.chars"
    RESPONSE_TRUNCATED = "debrix.response.truncated"
    RESPONSE_CAPTURE_ERROR = "debrix.response.capture_error"

    # Status / errors
    ERROR_SUMMARY = "debrix.error.summary"

    # Mock / replay / eval (reserved for later phases)
    MOCKED = "debrix.mocked"
    REPLAY_INPUT = "debrix.replay.input"
    REPLAY_OUTPUT = "debrix.replay.output"
    REPLAY_SEQUENCE_INDEX = "debrix.replay.sequence_index"
    EVAL_SOURCE_TRACE_ID = "debrix.eval.source_trace_id"

    PAYLOAD_KIND = "debrix.payload.kind"
    PAYLOAD_BLOB_REF = "debrix.payload.blob_ref"


class Event:
    """Debrix span event names."""

    PAYLOAD_READY = "debrix.payload.ready"


SPAN_KINDS: tuple[str, ...] = (
    SpanKind.AGENT,
    SpanKind.LLM,
    SpanKind.TOOL,
    SpanKind.MCP,
    SpanKind.MEMORY,
    SpanKind.EVALUATION,
    SpanKind.HUMAN,
    SpanKind.CUSTOM,
)
