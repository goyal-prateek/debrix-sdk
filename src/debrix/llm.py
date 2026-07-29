"""Opt-in LLM completion helper with Tool Mocker / Deterministic Replay.

Same resolve path as ``@trace_tool`` / ``MockableClient`` (``POST /mocks/resolve``
with ``kind=llm``). Use this instead of calling providers directly when you want
Mode B replay to stub historical responses.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from debrix.control import ControlInvoke, ControlReturn
from debrix.mocks import (
    apply_mock_decision,
    apply_mock_decision_async,
    is_stub_decision,
    resolve_mock,
)
from debrix.runtime_control import (
    apply_llm_messages_invoke,
    apply_llm_model_output_return,
    capture_llm_messages,
    is_live_control_trace,
    mark_live_span,
    resolve_runtime_control,
    resolve_runtime_control_async,
)
from debrix.semconv import Attr, SpanKind
from debrix.tracing import (
    _trace_span_async,
    current_agent_name,
    _dumps_replay,
    _mark_stub_decision,
    next_replay_sequence_index,
    trace_span,
)
from debrix.verification import check_boundary_async, check_boundary_sync

__all__ = ["complete", "acomplete"]

# call(messages) -> (content, usage_dict, model_name)
LiveCall = Callable[
    [Sequence[Mapping[str, Any]]],
    tuple[str, Mapping[str, Any], str],
]
AsyncLiveCall = Callable[
    [Sequence[Mapping[str, Any]]],
    Awaitable[tuple[str, Mapping[str, Any], str]],
]

_INLINE_REPLAY_LIMIT = 64_000


def _content_from_result(result: Any) -> str:
    """Extract assistant text from a replay/mock result payload."""
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        content = result.get("content")
        if isinstance(content, str):
            return content
        if content is not None:
            return str(content)
    return "" if result is None else str(result)


def _response_dict_from_result(
    result: Any,
    *,
    fallback_model: str | None = None,
) -> dict[str, Any]:
    if isinstance(result, Mapping):
        out = dict(result)
        if "content" not in out:
            out["content"] = _content_from_result(result)
        return out
    return {
        "content": _content_from_result(result),
        "model": fallback_model or "replay",
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def _record_replay_value(span: Any, attribute: str, value: Any) -> None:
    serialized = _dumps_replay(value)
    if len(serialized.encode("utf-8")) <= _INLINE_REPLAY_LIMIT:
        span.set_attribute(attribute, serialized)


def _start_llm_boundary(
    span: Any,
    messages: Sequence[Mapping[str, Any]],
) -> tuple[Any, int, list[dict[str, Any]]]:
    captured = capture_llm_messages(messages)
    if captured is None:
        raise ValueError(
            "messages must be a lossless JSON-compatible sequence of mappings"
        )
    recorded_messages = [
        dict(message) for message in captured.recorded_input["messages"]
    ]
    sequence_index = next_replay_sequence_index()
    span.set_attribute(Attr.REPLAY_SEQUENCE_INDEX, sequence_index)
    span.set_attribute(
        Attr.REPLAY_INPUT_DESCRIPTOR,
        _dumps_replay(captured.descriptor),
    )
    _record_replay_value(span, Attr.REPLAY_INPUT, captured.recorded_input)
    span.record_messages(recorded_messages)
    return captured, sequence_index, recorded_messages


def _live_response(
    result: tuple[str, Mapping[str, Any], str],
) -> dict[str, Any]:
    content, usage, used_model = result
    return {
        "content": content,
        "model": used_model,
        "usage": dict(usage),
    }


def _record_response(span: Any, response: Mapping[str, Any]) -> str:
    recorded = dict(response)
    span.record_response(recorded)
    _record_replay_value(span, Attr.REPLAY_OUTPUT, recorded)
    return _content_from_result(recorded)


def complete(
    messages: Sequence[Mapping[str, Any]],
    *,
    name: str = "complete",
    call: LiveCall | None = None,
    endpoint: str | None = None,
) -> str:
    """Run one LLM call inside a Debrix ``llm`` span with mock/replay resolve.

    Args:
        messages: Chat messages (system/user/assistant/tool).
        name: Span name and resolve ``name`` (default ``complete``).
        call: Live provider callable ``(messages) -> (content, usage, model)``.
            Required when Debrix returns passthrough (no armed LLM stub).
        endpoint: Optional OTLP base URL override for resolve.

    Returns:
        Assistant content string.
    """
    with trace_span(name, kind=SpanKind.LLM) as span:
        captured, sequence_index, recorded_messages = _start_llm_boundary(
            span, messages
        )
        effective_messages = recorded_messages
        managed = check_boundary_sync(span)
        live_execution = is_live_control_trace(span)
        if not managed:
            mark_live_span(span)
            controlled = resolve_runtime_control(
                    span,
                    operation_kind="llm",
                    operation_name=name,
                    operation_server=None,
                    agent_scope=current_agent_name(),
                    sequence_index=sequence_index,
                    input_value=captured.recorded_input,
                    input_descriptor=captured.descriptor,
                    capabilities=(
                        ("messages",)
                        if live_execution
                        else ("messages", "model_output")
                    ),
                    endpoint=endpoint,
                )
            if isinstance(controlled, ControlReturn):
                response = apply_llm_model_output_return(span, controlled)
                return _record_response(span, response)

            if isinstance(controlled, ControlInvoke):
                effective_messages = apply_llm_messages_invoke(
                    span, controlled, captured
                )
                _record_replay_value(
                    span,
                    Attr.REPLAY_INPUT,
                    {"messages": effective_messages},
                )
                if effective_messages != recorded_messages:
                    span.record_messages(effective_messages)
                if call is None:
                    raise RuntimeError(
                        "debrix.llm.complete requires call= for a controlled "
                        "message invocation"
                    )
                response = _live_response(call(effective_messages))
                post_control = resolve_runtime_control(
                    span,
                    operation_kind="llm",
                    operation_name=name,
                    operation_server=None,
                    agent_scope=current_agent_name(),
                    sequence_index=sequence_index,
                    input_value={"kind": "result", "value": response},
                    capabilities=("model_output",),
                    endpoint=endpoint,
                )
                if isinstance(post_control, ControlReturn):
                    response = apply_llm_model_output_return(span, post_control)
                return _record_response(span, response)

            mock_decision = resolve_mock(
                kind="llm",
                name=name,
                arguments={"messages": effective_messages},
                endpoint=endpoint,
                trace_id=span.trace_id_hex,
            )
            if is_stub_decision(mock_decision):
                _mark_stub_decision(span, mock_decision)
                result = apply_mock_decision(mock_decision)
                response = _response_dict_from_result(result)
                return _record_response(span, response)

        if call is None:
            raise RuntimeError(
                "debrix.llm.complete requires call= when Debrix returns "
                "passthrough (no armed LLM replay/mock). Pass a live provider "
                "callable, e.g. call=my_provider."
            )

        response = _live_response(call(effective_messages))
        if live_execution:
            post_control = resolve_runtime_control(
                span,
                operation_kind="llm",
                operation_name=name,
                operation_server=None,
                agent_scope=current_agent_name(),
                sequence_index=sequence_index,
                input_value={"kind": "result", "value": response},
                capabilities=("model_output",),
                endpoint=endpoint,
            )
            if isinstance(post_control, ControlReturn):
                response = apply_llm_model_output_return(span, post_control)
        return _record_response(span, response)


async def acomplete(
    messages: Sequence[Mapping[str, Any]],
    *,
    name: str = "complete",
    call: AsyncLiveCall | None = None,
    endpoint: str | None = None,
) -> str:
    """Async equivalent of :func:`complete` with the same control semantics."""

    async with _trace_span_async(name, kind=SpanKind.LLM) as span:
        captured, sequence_index, recorded_messages = _start_llm_boundary(
            span, messages
        )
        effective_messages = recorded_messages
        managed = await check_boundary_async(span)
        live_execution = is_live_control_trace(span)
        if not managed:
            mark_live_span(span)
            controlled = await resolve_runtime_control_async(
                    span,
                    operation_kind="llm",
                    operation_name=name,
                    operation_server=None,
                    agent_scope=current_agent_name(),
                    sequence_index=sequence_index,
                    input_value=captured.recorded_input,
                    input_descriptor=captured.descriptor,
                    capabilities=(
                        ("messages",)
                        if live_execution
                        else ("messages", "model_output")
                    ),
                    endpoint=endpoint,
                )
            if isinstance(controlled, ControlReturn):
                response = apply_llm_model_output_return(span, controlled)
                return _record_response(span, response)

            if isinstance(controlled, ControlInvoke):
                effective_messages = apply_llm_messages_invoke(
                    span, controlled, captured
                )
                _record_replay_value(
                    span,
                    Attr.REPLAY_INPUT,
                    {"messages": effective_messages},
                )
                if effective_messages != recorded_messages:
                    span.record_messages(effective_messages)
                if call is None:
                    raise RuntimeError(
                        "debrix.llm.acomplete requires call= for a controlled "
                        "message invocation"
                    )
                response = _live_response(await call(effective_messages))
                post_control = await resolve_runtime_control_async(
                    span,
                    operation_kind="llm",
                    operation_name=name,
                    operation_server=None,
                    agent_scope=current_agent_name(),
                    sequence_index=sequence_index,
                    input_value={"kind": "result", "value": response},
                    capabilities=("model_output",),
                    endpoint=endpoint,
                )
                if isinstance(post_control, ControlReturn):
                    response = apply_llm_model_output_return(span, post_control)
                return _record_response(span, response)

            mock_decision = resolve_mock(
                kind="llm",
                name=name,
                arguments={"messages": effective_messages},
                endpoint=endpoint,
                trace_id=span.trace_id_hex,
            )
            if is_stub_decision(mock_decision):
                _mark_stub_decision(span, mock_decision)
                result = await apply_mock_decision_async(mock_decision)
                response = _response_dict_from_result(result)
                return _record_response(span, response)

        if call is None:
            raise RuntimeError(
                "debrix.llm.acomplete requires call= when Debrix returns "
                "passthrough (no armed LLM replay/mock). Pass an async live "
                "provider callable."
            )
        response = _live_response(await call(effective_messages))
        if live_execution:
            post_control = await resolve_runtime_control_async(
                span,
                operation_kind="llm",
                operation_name=name,
                operation_server=None,
                agent_scope=current_agent_name(),
                sequence_index=sequence_index,
                input_value={"kind": "result", "value": response},
                capabilities=("model_output",),
                endpoint=endpoint,
            )
            if isinstance(post_control, ControlReturn):
                response = apply_llm_model_output_return(span, post_control)
        return _record_response(span, response)
