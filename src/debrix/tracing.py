"""Debrix instrumentation primitives: agents, tools, and generic spans."""

from __future__ import annotations

import functools
import inspect
import json
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any, ParamSpec, TypeVar, cast, overload

import opentelemetry.context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer

from debrix.config import configure
from debrix.control import ControlInvoke, ControlUnmanaged
from debrix.mocks import (
    MockDecision,
    MockToolError,
    apply_mock_decision,
    apply_mock_decision_async,
    is_stub_decision,
    resolve_mock,
)
from debrix.runtime_control import (
    apply_runtime_control,
    apply_runtime_invoke,
    capture_bound_call,
    is_live_control_trace,
    mark_live_span,
    resolve_runtime_control,
    resolve_runtime_control_async,
)
from debrix.semconv import Attr, SpanKind, Stub
from debrix.span import DebrixSpan
from debrix.verification import (
    check_boundary_async,
    check_boundary_sync,
    prepare_span_async,
    prepare_span_sync,
    reset_span_context,
)

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


class _ReplaySequenceState:
    """Mutable trace-scoped counter shared by copied async contexts."""

    __slots__ = ("next_index",)

    def __init__(self) -> None:
        self.next_index = 0

    def allocate(self) -> int:
        index = self.next_index
        self.next_index += 1
        return index


# A mutable value is intentional: child asyncio contexts copy the ContextVar
# binding but continue allocating from the same trace-wide sequence.
_REPLAY_SEQUENCE: ContextVar[_ReplaySequenceState | None] = ContextVar(
    "debrix.replay.sequence", default=None
)


def current_agent_name() -> str | None:
    """Return the nearest enclosing ``trace_agent`` name, if any."""
    value = otel_context.get_value(_AGENT_NAME_KEY)
    return value if isinstance(value, str) and value else None


def next_replay_sequence_index() -> int:
    """Allocate the next trace-wide ``debrix.replay.sequence_index``."""
    state = _REPLAY_SEQUENCE.get()
    if state is None:
        state = _ReplaySequenceState()
        _REPLAY_SEQUENCE.set(state)
    return state.allocate()


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
        is_root_agent = current_agent_name() is None
        agent_name = attrs.get(Attr.AGENT_NAME) or name
        agent_token = otel_context.attach(
            otel_context.set_value(_AGENT_NAME_KEY, agent_name)
        )
        if is_root_agent:
            seq_token = _REPLAY_SEQUENCE.set(_ReplaySequenceState())
    wrapper = DebrixSpan(span)
    exc: BaseException | None = None
    verification_token = None
    try:
        verification_token = prepare_span_sync(wrapper)
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
        reset_span_context(verification_token)
        _detach_token(token)


@asynccontextmanager
async def _trace_span_async(
    name: str,
    *,
    kind: str = SpanKind.CUSTOM,
    attributes: dict[str, str] | None = None,
) -> AsyncIterator[DebrixSpan]:
    """Async internal equivalent that does not block during verification bind."""

    attrs: dict[str, str] = {Attr.SPAN_KIND: kind}
    if attributes:
        attrs.update(attributes)
    span = get_tracer().start_span(name, attributes=attrs)
    token = _attach_span(span)
    agent_token: object | None = None
    seq_token = None
    if kind == SpanKind.AGENT:
        is_root_agent = current_agent_name() is None
        agent_name = attrs.get(Attr.AGENT_NAME) or name
        agent_token = otel_context.attach(
            otel_context.set_value(_AGENT_NAME_KEY, agent_name)
        )
        if is_root_agent:
            seq_token = _REPLAY_SEQUENCE.set(_ReplaySequenceState())
    wrapper = DebrixSpan(span)
    exc: BaseException | None = None
    verification_token = None
    try:
        verification_token = await prepare_span_async(wrapper)
        yield wrapper
    except BaseException as error:
        exc = error
        raise
    finally:
        if exc is not None:
            wrapper.record_exception(exc)
        span.end()
        if seq_token is not None:
            _REPLAY_SEQUENCE.reset(seq_token)
        if agent_token is not None:
            _detach_token(agent_token)
        reset_span_context(verification_token)
        _detach_token(token)


def _record_replay_io_start(span: DebrixSpan, bound: dict[str, Any]) -> int:
    """Write replay input + sequence index before the tool/MCP call."""
    sequence_index = next_replay_sequence_index()
    span.set_attribute(Attr.REPLAY_INPUT, _dumps_replay(bound))
    span.set_attribute(Attr.REPLAY_SEQUENCE_INDEX, sequence_index)
    return sequence_index


def _record_agent_arguments(
    span: DebrixSpan,
    arguments: Mapping[str, Any],
) -> None:
    """Record JSON-safe arguments on an agent span."""
    span.set_attribute(Attr.AGENT_ARGUMENTS, _dumps_replay(dict(arguments)))


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
    trace_id: str,
) -> MockDecision | None:
    """Return a mock/replay decision when this is a tool span; else ``None``."""
    if span_kind != SpanKind.TOOL:
        return None
    return resolve_mock(
        kind="tool",
        name=span_name,
        arguments=bound_args,
        trace_id=trace_id,
    )


def _wrap_function(
    fn: Callable[P, R],
    *,
    span_name: str,
    span_kind: str,
    attributes: dict[str, str],
    capture_io: bool = False,
    capture_arguments: bool = False,
    input_schema: Mapping[str, Any] | None = None,
) -> Callable[P, R]:
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            async with _trace_span_async(
                span_name, kind=span_kind, attributes=attributes
            ) as span:
                captured_call = (
                    capture_bound_call(
                        fn, args, kwargs, json_schema=input_schema
                    )
                    if capture_io and span_kind == SpanKind.TOOL
                    else None
                )
                bound = (
                    captured_call.recorded_input
                    if captured_call is not None
                    else _bind_arguments(fn, args, kwargs)
                    if capture_io or capture_arguments
                    else {}
                )
                if capture_arguments:
                    _record_agent_arguments(span, bound)
                sequence_index: int | None = None
                if capture_io:
                    # Record input before the call so failures still keep args.
                    sequence_index = _record_replay_io_start(span, bound)
                    if captured_call is not None:
                        span.set_attribute(
                            Attr.REPLAY_INPUT_DESCRIPTOR,
                            _dumps_replay(captured_call.descriptor),
                        )
                managed = (
                    await check_boundary_async(span)
                    if span_kind == SpanKind.TOOL and sequence_index is not None
                    else False
                )
                if not managed:
                    mark_live_span(span)
                live_execution = is_live_control_trace(span)
                if span_kind == SpanKind.TOOL and sequence_index is not None:
                    if not managed:
                        controlled = (
                            ControlUnmanaged()
                            if live_execution and captured_call is None
                            else await resolve_runtime_control_async(
                                span,
                                operation_kind="tool",
                                operation_name=span_name,
                                operation_server=None,
                                agent_scope=current_agent_name(),
                                sequence_index=sequence_index,
                                input_value=bound,
                                input_descriptor=(
                                    captured_call.descriptor
                                    if captured_call is not None
                                    else None
                                ),
                                capabilities=(
                                    ("input",) if live_execution else None
                                ),
                            )
                        )
                        if isinstance(controlled, ControlInvoke):
                            invoke_args, invoke_kwargs = apply_runtime_invoke(
                                span, controlled, captured_call
                            )
                            result = await cast(
                                Callable[..., Awaitable[Any]], fn
                            )(*invoke_args, **invoke_kwargs)
                            span.set_attribute(
                                Attr.REPLAY_OUTPUT, _dumps_replay(result)
                            )
                            if is_live_control_trace(span):
                                post_control = await resolve_runtime_control_async(
                                    span,
                                    operation_kind="tool",
                                    operation_name=span_name,
                                    operation_server=None,
                                    agent_scope=current_agent_name(),
                                    sequence_index=sequence_index,
                                    input_value={"kind": "result", "value": result},
                                    capabilities=("result", "error"),
                                )
                                handled, controlled_result = apply_runtime_control(
                                    span, post_control
                                )
                                if handled:
                                    result = controlled_result
                            return result
                        handled, result = apply_runtime_control(
                            span, controlled
                        )
                        if handled:
                            return result
                if not managed:
                    decision = _maybe_mock_tool(
                        span_name=span_name,
                        span_kind=span_kind,
                        bound_args=bound,
                        trace_id=span.trace_id_hex,
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
                if (
                    span_kind == SpanKind.TOOL
                    and sequence_index is not None
                    and live_execution
                ):
                    post_control = await resolve_runtime_control_async(
                        span,
                        operation_kind="tool",
                        operation_name=span_name,
                        operation_server=None,
                        agent_scope=current_agent_name(),
                        sequence_index=sequence_index,
                        input_value={"kind": "result", "value": result},
                        capabilities=("result", "error"),
                    )
                    handled, controlled_result = apply_runtime_control(
                        span, post_control
                    )
                    if handled:
                        result = controlled_result
                return result

        return cast(Callable[P, R], async_wrapper)

    @functools.wraps(fn)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with trace_span(span_name, kind=span_kind, attributes=attributes) as span:
            captured_call = (
                capture_bound_call(fn, args, kwargs, json_schema=input_schema)
                if capture_io and span_kind == SpanKind.TOOL
                else None
            )
            bound = (
                captured_call.recorded_input
                if captured_call is not None
                else _bind_arguments(fn, args, kwargs)
                if capture_io or capture_arguments
                else {}
            )
            if capture_arguments:
                _record_agent_arguments(span, bound)
            sequence_index: int | None = None
            if capture_io:
                sequence_index = _record_replay_io_start(span, bound)
                if captured_call is not None:
                    span.set_attribute(
                        Attr.REPLAY_INPUT_DESCRIPTOR,
                        _dumps_replay(captured_call.descriptor),
                    )
            managed = (
                check_boundary_sync(span)
                if span_kind == SpanKind.TOOL and sequence_index is not None
                else False
            )
            if not managed:
                mark_live_span(span)
            live_execution = is_live_control_trace(span)
            if span_kind == SpanKind.TOOL and sequence_index is not None:
                if not managed:
                    controlled = (
                        ControlUnmanaged()
                        if live_execution and captured_call is None
                        else resolve_runtime_control(
                            span,
                            operation_kind="tool",
                            operation_name=span_name,
                            operation_server=None,
                            agent_scope=current_agent_name(),
                            sequence_index=sequence_index,
                            input_value=bound,
                            input_descriptor=(
                                captured_call.descriptor
                                if captured_call is not None
                                else None
                            ),
                            capabilities=(("input",) if live_execution else None),
                        )
                    )
                    if isinstance(controlled, ControlInvoke):
                        invoke_args, invoke_kwargs = apply_runtime_invoke(
                            span, controlled, captured_call
                        )
                        result = fn(*invoke_args, **invoke_kwargs)
                        if capture_io:
                            span.set_attribute(
                                Attr.REPLAY_OUTPUT, _dumps_replay(result)
                            )
                        if is_live_control_trace(span):
                            post_control = resolve_runtime_control(
                                span,
                                operation_kind="tool",
                                operation_name=span_name,
                                operation_server=None,
                                agent_scope=current_agent_name(),
                                sequence_index=sequence_index,
                                input_value={"kind": "result", "value": result},
                                capabilities=("result", "error"),
                            )
                            handled, controlled_result = apply_runtime_control(
                                span, post_control
                            )
                            if handled:
                                result = controlled_result
                        return cast(R, result)
                    handled, result = apply_runtime_control(span, controlled)
                    if handled:
                        return cast(R, result)
            if not managed:
                decision = _maybe_mock_tool(
                    span_name=span_name,
                    span_kind=span_kind,
                    bound_args=bound,
                    trace_id=span.trace_id_hex,
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
                span.set_attribute(
                    Attr.REPLAY_OUTPUT, _dumps_replay(result)
                )
            if (
                span_kind == SpanKind.TOOL
                and sequence_index is not None
                and live_execution
            ):
                post_control = resolve_runtime_control(
                    span,
                    operation_kind="tool",
                    operation_name=span_name,
                    operation_server=None,
                    agent_scope=current_agent_name(),
                    sequence_index=sequence_index,
                    input_value={"kind": "result", "value": result},
                    capabilities=("result", "error"),
                )
                handled, controlled_result = apply_runtime_control(
                    span, post_control
                )
                if handled:
                    result = controlled_result
            return cast(R, result)

    return sync_wrapper


def _instrument(
    *,
    span_kind: str,
    identity_key: str,
    func: Callable[..., Any] | None,
    name: str | None,
    capture_io: bool = False,
    capture_arguments: bool = False,
    context_arguments: Mapping[str, Any] | None = None,
    input_schema: Mapping[str, Any] | None = None,
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
                if context_arguments is not None:
                    _record_agent_arguments(span, context_arguments)
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
            capture_arguments=capture_arguments,
            input_schema=input_schema,
        )

    if func is not None:
        return decorate(func)
    return decorate


@overload
def trace_agent(func: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def trace_agent(
    name: str,
    /,
    *,
    arguments: Mapping[str, Any] | None = None,
) -> Any: ...


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
    arguments: Mapping[str, Any] | None = None,
) -> Any:
    """Instrument an agent boundary.

    Usage::

        @trace_agent
        def run(): ...

        @trace_agent(name="planner")
        def run(): ...

        with trace_agent(
            "planner",
            arguments={"task": "inspect"},
        ) as span:
            ...
    """
    if arguments is not None:
        if not isinstance(func, str):
            raise TypeError(
                "arguments is only supported by the trace_agent context manager"
            )
        if not isinstance(arguments, Mapping):
            raise TypeError("arguments must be a mapping")

    return _instrument(
        span_kind=SpanKind.AGENT,
        identity_key=Attr.AGENT_NAME,
        func=func,
        name=name,
        capture_arguments=True,
        context_arguments=arguments,
        input_schema=None,
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
    input_schema: Mapping[str, Any] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def trace_tool(
    func: Callable[P, R] | str | None = None,
    /,
    *,
    name: str | None = None,
    input_schema: Mapping[str, Any] | None = None,
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
        input_schema=input_schema,
    )
