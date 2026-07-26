"""Debrix semantic conventions — shared span-kind and attribute-key contract.

This module is the Python source of truth for the Debrix semantic model. It
mirrors ``docs/Semantic_Model.md`` and the Rust ``semconv.rs`` in the desktop
app. Change the spec doc first, then keep all three in lockstep.
"""

from __future__ import annotations

__all__ = ["SpanKind", "Stub", "Attr", "Event", "SPAN_KINDS"]


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


class Stub:
    """Allowed values for the ``debrix.stub`` attribute."""

    MOCK = "mock"
    REPLAY = "replay"


class Attr:
    """Debrix-owned attribute keys. All use the ``debrix.`` prefix."""

    # Identity
    SPAN_KIND = "debrix.span.kind"
    AGENT_NAME = "debrix.agent.name"
    AGENT_ARGUMENTS = "debrix.agent.arguments"
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

    # Stub source / replay / eval
    # Values: Stub.MOCK | Stub.REPLAY (mutually exclusive; omit when live).
    STUB = "debrix.stub"
    REPLAY_INPUT = "debrix.replay.input"
    REPLAY_INPUT_DESCRIPTOR = "debrix.replay.input_descriptor"
    REPLAY_OUTPUT = "debrix.replay.output"
    REPLAY_SEQUENCE_INDEX = "debrix.replay.sequence_index"
    EVAL_SOURCE_TRACE_ID = "debrix.eval.source_trace_id"

    # FW v2 controlled-branch runtime provenance.
    CONTROL_BRANCH_ID = "debrix.control.branch_id"
    CONTROL_ATTEMPT_ID = "debrix.control.attempt_id"
    CONTROL_OCCURRENCE_ID = "debrix.control.occurrence_id"
    CONTROL_PROVENANCE = "debrix.control.provenance"
    CONTROL_INPUT_PROVENANCE = "debrix.control.input_provenance"
    CONTROL_RESULT_PROVENANCE = "debrix.control.result_provenance"

    # FW v3 managed no-override verification provenance.
    VERIFICATION_ATTEMPT_ID = "debrix.verification.attempt_id"
    VERIFICATION_PURPOSE = "debrix.verification.purpose"
    VERIFICATION_PROTOCOL_VERSION = "debrix.verification.protocol_version"
    VERIFICATION_ROOT_SPAN_ID = "debrix.verification.root_span_id"
    VERIFICATION_NO_OVERRIDES = "debrix.verification.no_overrides"

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
