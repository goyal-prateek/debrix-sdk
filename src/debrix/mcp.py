"""Debrix-aware MCP client wrapper for Tool Mocker (Phase 3).

Wrap any object that exposes ``call_tool(name, arguments)`` (sync or async).
When Debrix has an enabled mock rule, the real MCP server is not called.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from debrix.mocks import (
    MockToolError,
    _json_safe,
    apply_mock_decision,
    apply_mock_decision_async,
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
    ) -> None:
        self._inner = inner
        self._server = server
        self._endpoint = endpoint

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
        args = dict(arguments or {})
        # Some MCP SDKs use keyword-only arguments=
        if not args and "arguments" in kwargs and isinstance(kwargs["arguments"], Mapping):
            args = dict(kwargs["arguments"])

        inner_call = getattr(self._inner, "call_tool", None)
        if inner_call is None:
            raise AttributeError("inner client has no call_tool method")

        if inspect.iscoroutinefunction(inner_call):
            return self._call_tool_async(name, args, kwargs)
        return self._call_tool_sync(name, args, kwargs)

    def _call_tool_sync(
        self,
        name: str,
        args: dict[str, Any],
        forward_kwargs: dict[str, Any],
    ) -> Any:
        server = self._server_name()
        attrs: dict[str, str] = {
            Attr.TOOL_NAME: name,
        }
        if server:
            attrs[Attr.MCP_SERVER] = server
            attrs[Attr.MCP_TOOL] = name

        with trace_span(name, kind=SpanKind.MCP, attributes=attrs) as span:
            _record_replay_io_start(span, args)
            decision = resolve_mock(
                kind="mcp",
                name=name,
                arguments=_json_safe(args),
                server=server,
                endpoint=self._endpoint,
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
                span.set_attribute(Attr.REPLAY_OUTPUT, _dumps_replay(result))
                return result

            # Prefer positional (name, arguments); fall back to kwargs style.
            try:
                result = self._inner.call_tool(name, args, **forward_kwargs)
            except TypeError:
                result = self._inner.call_tool(
                    name, arguments=args, **forward_kwargs
                )
            span.set_attribute(Attr.REPLAY_OUTPUT, _dumps_replay(result))
            return result

    async def _call_tool_async(
        self,
        name: str,
        args: dict[str, Any],
        forward_kwargs: dict[str, Any],
    ) -> Any:
        server = self._server_name()
        attrs: dict[str, str] = {
            Attr.TOOL_NAME: name,
        }
        if server:
            attrs[Attr.MCP_SERVER] = server
            attrs[Attr.MCP_TOOL] = name

        with trace_span(name, kind=SpanKind.MCP, attributes=attrs) as span:
            _record_replay_io_start(span, args)
            decision = resolve_mock(
                kind="mcp",
                name=name,
                arguments=_json_safe(args),
                server=server,
                endpoint=self._endpoint,
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
                span.set_attribute(Attr.REPLAY_OUTPUT, _dumps_replay(result))
                return result

            try:
                result = await self._inner.call_tool(name, args, **forward_kwargs)
            except TypeError:
                result = await self._inner.call_tool(
                    name, arguments=args, **forward_kwargs
                )
            span.set_attribute(Attr.REPLAY_OUTPUT, _dumps_replay(result))
            return result
