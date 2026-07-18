"""Opt-in LLM completion helper with Tool Mocker / Deterministic Replay.

Same resolve path as ``@trace_tool`` / ``MockableClient`` (``POST /mocks/resolve``
with ``kind=llm``). Use this instead of calling providers directly when you want
Mode B replay to stub historical responses.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from debrix.mocks import (
    apply_mock_decision,
    is_stub_decision,
    resolve_mock,
)
from debrix.semconv import Attr, SpanKind
from debrix.tracing import (
    _dumps_replay,
    _mark_stub_decision,
    _record_replay_io_start,
    trace_span,
)

__all__ = ["complete"]

# call(messages) -> (content, usage_dict, model_name)
LiveCall = Callable[
    [Sequence[Mapping[str, Any]]],
    tuple[str, Mapping[str, Any], str],
]


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
    msg_list = [dict(m) for m in messages]
    with trace_span(name, kind=SpanKind.LLM) as span:
        span.record_messages(msg_list)
        _record_replay_io_start(span, {"messages": msg_list})
        decision = resolve_mock(
            kind="llm",
            name=name,
            arguments={"messages": msg_list},
            endpoint=endpoint,
        )
        if is_stub_decision(decision):
            _mark_stub_decision(span, decision)
            result = apply_mock_decision(decision)
            response = _response_dict_from_result(result)
            span.record_response(response)
            span.set_attribute(Attr.REPLAY_OUTPUT, _dumps_replay(response))
            return _content_from_result(result)

        if call is None:
            raise RuntimeError(
                "debrix.llm.complete requires call= when Debrix returns "
                "passthrough (no armed LLM replay/mock). Pass a live provider "
                "callable, e.g. call=my_provider."
            )

        content, usage, used_model = call(msg_list)
        response = {
            "content": content,
            "model": used_model,
            "usage": dict(usage),
        }
        span.record_response(response)
        span.set_attribute(Attr.REPLAY_OUTPUT, _dumps_replay(response))
        return content
