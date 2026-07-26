"""Debrix-aware MCP client wrapper for Tool Mocker (Phase 3).

Wrap any object that exposes ``call_tool(name, arguments)`` (sync or async).
When Debrix has an enabled mock rule, the real MCP server is not called.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from debrix.control import ControlInvoke
from debrix.mocks import (
    MockToolError,
    _json_safe,
    apply_mock_decision,
    apply_mock_decision_async,
    is_stub_decision,
    resolve_mock,
)
from debrix.semconv import Attr, SpanKind
from debrix.runtime_control import (
    apply_mapping_invoke,
    apply_runtime_control,
    capture_mapping_input,
    is_live_control_trace,
    mark_live_span,
    resolve_runtime_control,
    resolve_runtime_control_async,
)
from debrix.tracing import (
    _trace_span_async,
    current_agent_name,
    _dumps_replay,
    _mark_stub_decision,
    _record_replay_io_start,
    trace_span,
)
from debrix.verification import check_boundary_async, check_boundary_sync

__all__ = ["MockableClient"]


class MockableClient:
    """Opt-in wrapper: resolve Debrix mocks, else forward to ``inner.call_tool``.

    Example::

        client = MockableClient(real_client, server="demo-db")
        result = await client.call_tool("query", {"sql": "select 1"})
    """

    def __init__(
        self,
        inner: Any,
        *,
        server: str | None = None,
        endpoint: str | None = None,
        input_schemas: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self._inner = inner
        self._server = server
        self._endpoint = endpoint
        self._input_schemas = dict(input_schemas or {})

    @property
    def inner(self) -> Any:
        return self._inner

    def _server_name(self) -> str | None:
        if self._server:
            return self._server
        for attr_name in ("server_name", "name", "server"):
            val = getattr(self._inner, attr_name, None)
            if isinstance(val, str) and val:
                return val
        return None

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> Any:
        """Sync entrypoint. If the inner client is async, returns a coroutine."""
        caller_used_keyword = "arguments" in kwargs
        args = dict(arguments or {})
        # Some MCP SDKs use keyword-only arguments=
        if not args and "arguments" in kwargs and isinstance(kwargs["arguments"], Mapping):
            args = dict(kwargs["arguments"])
        forward_kwargs = dict(kwargs)
        forward_kwargs.pop("arguments", None)

        inner_call = getattr(self._inner, "call_tool", None)
        if inner_call is None:
            raise AttributeError("inner client has no call_tool method")

        if inspect.iscoroutinefunction(inner_call):
            return self._call_tool_async(
                name,
                args,
                forward_kwargs,
                self._uses_keyword_arguments(inner_call, caller_used_keyword),
            )
        return self._call_tool_sync(
            name,
            args,
            forward_kwargs,
            self._uses_keyword_arguments(inner_call, caller_used_keyword),
        )

    @staticmethod
    def _uses_keyword_arguments(inner_call: Any, caller_used_keyword: bool) -> bool:
        try:
            parameter = inspect.signature(inner_call).parameters.get("arguments")
        except (TypeError, ValueError):
            return caller_used_keyword
        if parameter is None:
            return caller_used_keyword
        return parameter.kind == inspect.Parameter.KEYWORD_ONLY

    def _invoke_sync(
        self,
        name: str,
        args: dict[str, Any],
        forward_kwargs: dict[str, Any],
        keyword_arguments: bool,
    ) -> Any:
        if keyword_arguments:
            return self._inner.call_tool(
                name, arguments=args, **forward_kwargs
            )
        return self._inner.call_tool(name, args, **forward_kwargs)

    async def _invoke_async(
        self,
        name: str,
        args: dict[str, Any],
        forward_kwargs: dict[str, Any],
        keyword_arguments: bool,
    ) -> Any:
        if keyword_arguments:
            return await self._inner.call_tool(
                name, arguments=args, **forward_kwargs
            )
        return await self._inner.call_tool(name, args, **forward_kwargs)

    def _call_tool_sync(
        self,
        name: str,
        args: dict[str, Any],
        forward_kwargs: dict[str, Any],
        keyword_arguments: bool,
    ) -> Any:
        server = self._server_name()
        attrs: dict[str, str] = {
            Attr.TOOL_NAME: name,
        }
        if server:
            attrs[Attr.MCP_SERVER] = server
            attrs[Attr.MCP_TOOL] = name

        with trace_span(name, kind=SpanKind.MCP, attributes=attrs) as span:
            captured_input = capture_mapping_input(
                args,
                operation_kind="mcp",
                json_schema=self._input_schemas.get(name),
            )
            recorded_input = captured_input[0] if captured_input else args
            sequence_index = _record_replay_io_start(span, recorded_input)
            if captured_input is not None:
                span.set_attribute(
                    Attr.REPLAY_INPUT_DESCRIPTOR,
                    _dumps_replay(captured_input[1]),
                )
            managed = check_boundary_sync(span)
            if not managed:
                mark_live_span(span)
                live_execution = is_live_control_trace(span)
                controlled = resolve_runtime_control(
                    span,
                    operation_kind="mcp",
                    operation_name=name,
                    operation_server=server,
                    agent_scope=current_agent_name(),
                    sequence_index=sequence_index,
                    input_value=recorded_input,
                    input_descriptor=(
                        captured_input[1]
                        if captured_input is not None
                        else None
                    ),
                    capabilities=(("input",) if live_execution else None),
                )
                if isinstance(controlled, ControlInvoke):
                    invoked_args = apply_mapping_invoke(
                        span, controlled, captured_input
                    )
                    result = self._invoke_sync(
                        name,
                        invoked_args,
                        forward_kwargs,
                        keyword_arguments,
                    )
                    span.set_attribute(
                        Attr.REPLAY_OUTPUT, _dumps_replay(result)
                    )
                    if is_live_control_trace(span):
                        post_control = resolve_runtime_control(
                            span,
                            operation_kind="mcp",
                            operation_name=name,
                            operation_server=server,
                            agent_scope=current_agent_name(),
                            sequence_index=sequence_index,
                            input_value={"kind": "result", "value": result},
                            capabilities=("result", "error"),
                            endpoint=self._endpoint,
                        )
                        handled, controlled_result = apply_runtime_control(
                            span, post_control
                        )
                        if handled:
                            result = controlled_result
                    return result
                handled, result = apply_runtime_control(span, controlled)
                if handled:
                    return result
                decision = resolve_mock(
                    kind="mcp",
                    name=name,
                    arguments=_json_safe(args),
                    server=server,
                    endpoint=self._endpoint,
                    trace_id=span.trace_id_hex,
                )
                if is_stub_decision(decision):
                    _mark_stub_decision(span, decision)
                    try:
                        result = apply_mock_decision(decision)
                    except MockToolError as exc:
                        span.set_attribute(
                            Attr.REPLAY_OUTPUT,
                            _dumps_replay(
                                {"error": exc.kind, "message": exc.message}
                            ),
                        )
                        raise
                    span.set_attribute(
                        Attr.REPLAY_OUTPUT, _dumps_replay(result)
                    )
                    return result

            result = self._invoke_sync(
                name, args, forward_kwargs, keyword_arguments
            )
            span.set_attribute(Attr.REPLAY_OUTPUT, _dumps_replay(result))
            if not managed and is_live_control_trace(span):
                post_control = resolve_runtime_control(
                    span,
                    operation_kind="mcp",
                    operation_name=name,
                    operation_server=server,
                    agent_scope=current_agent_name(),
                    sequence_index=sequence_index,
                    input_value={"kind": "result", "value": result},
                    capabilities=("result", "error"),
                    endpoint=self._endpoint,
                )
                handled, controlled_result = apply_runtime_control(
                    span, post_control
                )
                if handled:
                    result = controlled_result
            return result

    async def _call_tool_async(
        self,
        name: str,
        args: dict[str, Any],
        forward_kwargs: dict[str, Any],
        keyword_arguments: bool,
    ) -> Any:
        server = self._server_name()
        attrs: dict[str, str] = {
            Attr.TOOL_NAME: name,
        }
        if server:
            attrs[Attr.MCP_SERVER] = server
            attrs[Attr.MCP_TOOL] = name

        async with _trace_span_async(
            name, kind=SpanKind.MCP, attributes=attrs
        ) as span:
            captured_input = capture_mapping_input(
                args,
                operation_kind="mcp",
                json_schema=self._input_schemas.get(name),
            )
            recorded_input = captured_input[0] if captured_input else args
            sequence_index = _record_replay_io_start(span, recorded_input)
            if captured_input is not None:
                span.set_attribute(
                    Attr.REPLAY_INPUT_DESCRIPTOR,
                    _dumps_replay(captured_input[1]),
                )
            managed = await check_boundary_async(span)
            if not managed:
                mark_live_span(span)
                live_execution = is_live_control_trace(span)
                controlled = await resolve_runtime_control_async(
                    span,
                    operation_kind="mcp",
                    operation_name=name,
                    operation_server=server,
                    agent_scope=current_agent_name(),
                    sequence_index=sequence_index,
                    input_value=recorded_input,
                    input_descriptor=(
                        captured_input[1]
                        if captured_input is not None
                        else None
                    ),
                    capabilities=(("input",) if live_execution else None),
                )
                if isinstance(controlled, ControlInvoke):
                    invoked_args = apply_mapping_invoke(
                        span, controlled, captured_input
                    )
                    result = await self._invoke_async(
                        name,
                        invoked_args,
                        forward_kwargs,
                        keyword_arguments,
                    )
                    span.set_attribute(
                        Attr.REPLAY_OUTPUT, _dumps_replay(result)
                    )
                    if is_live_control_trace(span):
                        post_control = await resolve_runtime_control_async(
                            span,
                            operation_kind="mcp",
                            operation_name=name,
                            operation_server=server,
                            agent_scope=current_agent_name(),
                            sequence_index=sequence_index,
                            input_value={"kind": "result", "value": result},
                            capabilities=("result", "error"),
                            endpoint=self._endpoint,
                        )
                        handled, controlled_result = apply_runtime_control(
                            span, post_control
                        )
                        if handled:
                            result = controlled_result
                    return result
                handled, result = apply_runtime_control(span, controlled)
                if handled:
                    return result
                decision = resolve_mock(
                    kind="mcp",
                    name=name,
                    arguments=_json_safe(args),
                    server=server,
                    endpoint=self._endpoint,
                    trace_id=span.trace_id_hex,
                )
                if is_stub_decision(decision):
                    _mark_stub_decision(span, decision)
                    try:
                        result = await apply_mock_decision_async(decision)
                    except MockToolError as exc:
                        span.set_attribute(
                            Attr.REPLAY_OUTPUT,
                            _dumps_replay(
                                {"error": exc.kind, "message": exc.message}
                            ),
                        )
                        raise
                    span.set_attribute(
                        Attr.REPLAY_OUTPUT, _dumps_replay(result)
                    )
                    return result

            result = await self._invoke_async(
                name, args, forward_kwargs, keyword_arguments
            )
            span.set_attribute(Attr.REPLAY_OUTPUT, _dumps_replay(result))
            if not managed and is_live_control_trace(span):
                post_control = await resolve_runtime_control_async(
                    span,
                    operation_kind="mcp",
                    operation_name=name,
                    operation_server=server,
                    agent_scope=current_agent_name(),
                    sequence_index=sequence_index,
                    input_value={"kind": "result", "value": result},
                    capabilities=("result", "error"),
                    endpoint=self._endpoint,
                )
                handled, controlled_result = apply_runtime_control(
                    span, post_control
                )
                if handled:
                    result = controlled_result
            return result
