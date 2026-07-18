"""Debrix instrumentation primitives: agents, tools, and generic spans."""

from __future__ import annotations

import functools
import inspect
import json
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ParamSpec, TypeVar, cast, overload

import opentelemetry.context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer

from debrix.config import configure
from debrix.mocks import (
    MockDecision,
    MockToolError,
    apply_mock_decision,
    apply_mock_decision_async,
    is_stub_decision,
    resolve_mock,
)
from debrix.semconv import Attr, SpanKind, Stub
from debrix.span import DebrixSpan

__all__ = [
    "trace_agent",
    "trace_tool",
    "trace_span",
    "get_tracer",
    "current_agent_name",
    "next_replay_sequence_index",
]

P = ParamSpec("P")
R = TypeVar("R")

_TRACER_NAME = "debrix"
_SKIP_BOUND_PARAMS = frozenset({"self", "cls"})
_AGENT_NAME_KEY = otel_context.create_key("debrix.agent.name")
# Agent-scoped tool/MCP sequence for deterministic replay tapes.
_REPLAY_SEQUENCE: ContextVar[int] = ContextVar("debrix.replay.sequence", default=0)


def current_agent_name() -> str | None:
    """Return the nearest enclosing ``trace_agent`` name, if any."""
    value = otel_context.get_value(_AGENT_NAME_KEY)
    return value if isinstance(value, str) and value else None


def next_replay_sequence_index() -> int:
    """Allocate the next ``debrix.replay.sequence_index`` in this context."""
    idx = _REPLAY_SEQUENCE.get()
    _REPLAY_SEQUENCE.set(idx + 1)
    return idx


def get_tracer() -> Tracer:
    """Return the Debrix tracer, ensuring a provider is configured."""
    current = trace.get_tracer_provider()
    if not isinstance(current, TracerProvider):
        configure()
    return trace.get_tracer(_TRACER_NAME)


def _attach_span(span: Any) -> object:
    ctx = trace.set_span_in_context(span)
    return otel_context.attach(ctx)


def _detach_token(token: object) -> None:
    otel_context.detach(token)


def _json_safe(value: Any) -> Any:
    """Convert a value into something ``json.dumps`` can encode."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return repr(value)


def _dumps_replay(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False)


def _bind_arguments(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Bind call args to parameter names for replay (skip ``self`` / ``cls``)."""
    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
    except TypeError:
        return {
            "args": list(args),
            "kwargs": dict(kwargs),
        }
    return {
        name: value
        for name, value in bound.arguments.items()
        if name not in _SKIP_BOUND_PARAMS
    }


@contextmanager
def trace_span(
    name: str,
    *,
    kind: str = SpanKind.CUSTOM,
    attributes: dict[str, str] | None = None,
) -> Iterator[DebrixSpan]:
    """Context manager for a Debrix-instrumented span.

    Args:
        name: Span name.
        kind: Value for ``debrix.span.kind`` (default ``custom``).
        attributes: Extra string attributes to set at start.
    """
    attrs: dict[str, str] = {Attr.SPAN_KIND: kind}
    if attributes:
        attrs.update(attributes)
    span = get_tracer().start_span(name, attributes=attrs)
    token = _attach_span(span)
    agent_token: object | None = None
    seq_token = None
    if kind == SpanKind.AGENT:
        agent_name = attrs.get(Attr.AGENT_NAME) or name
        agent_token = otel_context.attach(
            otel_context.set_value(_AGENT_NAME_KEY, agent_name)
        )
        # Reset tool/MCP sequence for each agent boundary.
        seq_token = _REPLAY_SEQUENCE.set(0)
    wrapper = DebrixSpan(span)
    exc: BaseException | None = None
    try:
        yield wrapper
    except BaseException as e:
        exc = e
        raise
    finally:
        if exc is not None:
            wrapper.record_exception(exc)
        span.end()
        if seq_token is not None:
            _REPLAY_SEQUENCE.reset(seq_token)
        if agent_token is not None:
            _detach_token(agent_token)
        _detach_token(token)


def _record_replay_io_start(span: DebrixSpan, bound: dict[str, Any]) -> None:
    """Write replay input + sequence index before the tool/MCP call."""
    span.set_attribute(Attr.REPLAY_INPUT, _dumps_replay(bound))
    span.set_attribute(Attr.REPLAY_SEQUENCE_INDEX, next_replay_sequence_index())


def _mark_stub_decision(span: DebrixSpan, decision: MockDecision) -> None:
    if decision.action == "replay":
        span.set_attribute(Attr.STUB, Stub.REPLAY)
    else:
        span.set_attribute(Attr.STUB, Stub.MOCK)


def _maybe_mock_tool(
    *,
    span_name: str,
    span_kind: str,
    bound_args: dict[str, Any],
) -> MockDecision | None:
    """Return a mock/replay decision when this is a tool span; else ``None``."""
    if span_kind != SpanKind.TOOL:
        return None
    return resolve_mock(kind="tool", name=span_name, arguments=bound_args)


def _wrap_function(
    fn: Callable[P, R],
    *,
    span_name: str,
    span_kind: str,
    attributes: dict[str, str],
    capture_io: bool = False,
) -> Callable[P, R]:
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            with trace_span(
                span_name, kind=span_kind, attributes=attributes
            ) as span:
                bound = (
                    _bind_arguments(fn, args, kwargs) if capture_io else {}
                )
                if capture_io:
                    # Record input before the call so failures still keep args.
                    _record_replay_io_start(span, bound)
                decision = _maybe_mock_tool(
                    span_name=span_name,
                    span_kind=span_kind,
                    bound_args=bound,
                )
                if is_stub_decision(decision):
                    assert decision is not None
                    _mark_stub_decision(span, decision)
                    try:
                        result = await apply_mock_decision_async(decision)
                    except MockToolError as exc:
                        if capture_io:
                            span.set_attribute(
                                Attr.REPLAY_OUTPUT,
                                _dumps_replay(
                                    {
                                        "error": exc.kind,
                                        "message": exc.message,
                                    }
                                ),
                            )
                        raise
                    if capture_io:
                        span.set_attribute(
                            Attr.REPLAY_OUTPUT, _dumps_replay(result)
                        )
                    return result
                result = await cast(Callable[..., Awaitable[Any]], fn)(
                    *args, **kwargs
                )
                if capture_io:
                    span.set_attribute(
                        Attr.REPLAY_OUTPUT, _dumps_replay(result)
                    )
                return result

        return cast(Callable[P, R], async_wrapper)

    @functools.wraps(fn)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with trace_span(span_name, kind=span_kind, attributes=attributes) as span:
            bound = _bind_arguments(fn, args, kwargs) if capture_io else {}
            if capture_io:
                _record_replay_io_start(span, bound)
            decision = _maybe_mock_tool(
                span_name=span_name,
                span_kind=span_kind,
                bound_args=bound,
            )
            if is_stub_decision(decision):
                assert decision is not None
                _mark_stub_decision(span, decision)
                try:
                    result = apply_mock_decision(decision)
                except MockToolError as exc:
                    if capture_io:
                        span.set_attribute(
                            Attr.REPLAY_OUTPUT,
                            _dumps_replay(
                                {
                                    "error": exc.kind,
                                    "message": exc.message,
                                }
                            ),
                        )
                    raise
                if capture_io:
                    span.set_attribute(
                        Attr.REPLAY_OUTPUT, _dumps_replay(result)
                    )
                return cast(R, result)
            result = fn(*args, **kwargs)
            if capture_io:
                span.set_attribute(Attr.REPLAY_OUTPUT, _dumps_replay(result))
            return result

    return sync_wrapper


def _instrument(
    *,
    span_kind: str,
    identity_key: str,
    func: Callable[..., Any] | None,
    name: str | None,
    capture_io: bool = False,
) -> Any:
    """Shared implementation for ``trace_agent`` / ``trace_tool``.

    Supports:
    - ``@trace_agent`` / ``@trace_tool``
    - ``@trace_agent(name=...)`` / ``@trace_tool(name=...)``
    - ``with trace_agent("name"):`` / ``with trace_tool("name"):``
    """
    # Context-manager form: first positional arg is a string span name.
    if isinstance(func, str):
        span_name = func

        @contextmanager
        def cm() -> Iterator[DebrixSpan]:
            with trace_span(
                span_name,
                kind=span_kind,
                attributes={identity_key: span_name},
            ) as span:
                yield span

        return cm()

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        span_name = name or fn.__name__
        return _wrap_function(
            fn,
            span_name=span_name,
            span_kind=span_kind,
            attributes={identity_key: span_name},
            capture_io=capture_io,
        )

    if func is not None:
        return decorate(func)
    return decorate


@overload
def trace_agent(func: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def trace_agent(name: str, /) -> Any: ...


@overload
def trace_agent(
    func: None = None,
    /,
    *,
    name: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def trace_agent(
    func: Callable[P, R] | str | None = None,
    /,
    *,
    name: str | None = None,
) -> Any:
    """Instrument an agent boundary.

    Usage::

        @trace_agent
        def run(): ...

        @trace_agent(name="planner")
        def run(): ...

        with trace_agent("planner") as span:
            ...
    """
    return _instrument(
        span_kind=SpanKind.AGENT,
        identity_key=Attr.AGENT_NAME,
        func=func,
        name=name,
        capture_io=False,
    )


@overload
def trace_tool(func: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def trace_tool(name: str, /) -> Any: ...


@overload
def trace_tool(
    func: None = None,
    /,
    *,
    name: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def trace_tool(
    func: Callable[P, R] | str | None = None,
    /,
    *,
    name: str | None = None,
) -> Any:
    """Instrument a tool call.

    When used as a decorator, records bound call arguments on
    ``debrix.replay.input`` and the return value on ``debrix.replay.output``
    (JSON strings) for later deterministic replay. Input is written before the
    call so failures still retain arguments. Context-manager form does not
    auto-capture I/O — set those attributes yourself if needed.

    Usage::

        @trace_tool
        def search(): ...

        @trace_tool(name="web_search")
        def search(): ...

        with trace_tool("search") as span:
            ...
    """
    return _instrument(
        span_kind=SpanKind.TOOL,
        identity_key=Attr.TOOL_NAME,
        func=func,
        name=name,
        capture_io=True,
    )
