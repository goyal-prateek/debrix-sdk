"""Tests for MockableClient."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from debrix import Attr, SpanKind, Stub
from debrix.mcp import MockableClient
from debrix.mocks import MockDecision, MockError, MockToolError, PASSTHROUGH


class _FakeInner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        args = dict(arguments or {})
        self.calls.append((name, args))
        return f"live:{name}:{args.get('q')}"


class _FakeAsyncInner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> str:
        args = dict(arguments or {})
        self.calls.append((name, args))
        return f"live:{name}"


def test_mockable_client_passthrough(
    memory_exporter: InMemorySpanExporter,
) -> None:
    inner = _FakeInner()
    client = MockableClient(inner, server="demo")
    with patch("debrix.mcp.resolve_mock", return_value=PASSTHROUGH):
        assert client.call_tool("search", {"q": "hi"}) == "live:search:hi"
    assert inner.calls == [("search", {"q": "hi"})]
    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes[Attr.SPAN_KIND] == SpanKind.MCP
    assert span.attributes[Attr.MCP_SERVER] == "demo"
    assert Attr.STUB not in span.attributes
    assert isinstance(span.attributes[Attr.REPLAY_SEQUENCE_INDEX], int)


def test_mockable_client_replay(
    memory_exporter: InMemorySpanExporter,
) -> None:
    inner = _FakeInner()
    client = MockableClient(inner, server="demo")
    fake = MockDecision(action="replay", result={"rows": [1]})
    with patch("debrix.mcp.resolve_mock", return_value=fake):
        out = client.call_tool("query", {"sql": "select 1"})
    assert out == {"rows": [1]}
    assert inner.calls == []
    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes[Attr.STUB] == Stub.REPLAY
    assert json.loads(span.attributes[Attr.REPLAY_OUTPUT]) == {"rows": [1]}


def test_mockable_client_mocks(
    memory_exporter: InMemorySpanExporter,
) -> None:
    inner = _FakeInner()
    client = MockableClient(inner, server="demo")
    fake = MockDecision(action="mock", result={"rows": []})
    with patch("debrix.mcp.resolve_mock", return_value=fake) as resolve:
        out = client.call_tool("query", {"sql": "select 1"})
    assert out == {"rows": []}
    assert inner.calls == []
    resolve.assert_called_once()
    kwargs = resolve.call_args.kwargs
    assert kwargs["kind"] == "mcp"
    assert kwargs["name"] == "query"
    assert kwargs["server"] == "demo"
    assert kwargs["arguments"] == {"sql": "select 1"}

    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes[Attr.STUB] == Stub.MOCK
    assert json.loads(span.attributes[Attr.REPLAY_OUTPUT]) == {"rows": []}


def test_mockable_client_async_mock(
    memory_exporter: InMemorySpanExporter,
) -> None:
    inner = _FakeAsyncInner()
    client = MockableClient(inner, server="db")
    fake = MockDecision(
        action="mock",
        error=MockError(kind="timeout", message="db timeout"),
    )

    async def _run() -> None:
        with patch("debrix.mcp.resolve_mock", return_value=fake):
            with pytest.raises(MockToolError, match="db timeout"):
                await client.call_tool("query", {"sql": "x"})

    asyncio.run(_run())
    assert inner.calls == []
    assert (
        memory_exporter.get_finished_spans()[0].attributes[Attr.STUB] == Stub.MOCK
    )
