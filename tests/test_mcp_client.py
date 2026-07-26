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
from debrix.control import (
    ControlInvoke,
    ControlResolvedInput,
    ControlUnmanaged,
    DebrixControlProtocolError,
)
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


class _KeywordOnlyInner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call_tool(self, name: str, *, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return "keyword-live"


class _TypeErrorInner:
    def __init__(self) -> None:
        self.calls = 0

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls += 1
        raise TypeError(f"real failure in {name}: {arguments}")


def _invoke(value: dict[str, Any]) -> ControlInvoke:
    return ControlInvoke(
        attempt_id="branch_attempt_123",
        branch_id="branch_123",
        occurrence_id="occurrence_123",
        decision_id="decision_123",
        input=ControlResolvedInput(provenance="edited", value=value),
        capture_live_result=True,
    )


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
    assert kwargs["trace_id"] == format(span.context.trace_id, "032x")
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


def test_managed_mcp_input_invokes_edited_arguments_once_and_bypasses_mock(
    memory_exporter: InMemorySpanExporter,
) -> None:
    inner = _FakeInner()
    client = MockableClient(inner, server="demo")
    with (
        patch(
            "debrix.mcp.resolve_runtime_control",
            return_value=_invoke({"q": "edited"}),
        ) as control,
        patch("debrix.mcp.resolve_mock") as mock_resolver,
    ):
        assert client.call_tool("search", {"q": "recorded"}) == (
            "live:search:edited"
        )

    assert inner.calls == [("search", {"q": "edited"})]
    mock_resolver.assert_not_called()
    assert control.call_args.kwargs["input_descriptor"] == {
        "schemaVersion": 1,
        "operationKind": "mcp",
        "jsonKind": "object",
        "parameters": [
            {
                "name": "q",
                "kind": "mapping_key",
                "required": True,
                "hasDefault": False,
                "editable": True,
                "jsonKind": "string",
                "pythonType": "builtins.str",
            }
        ],
    }
    attrs = memory_exporter.get_finished_spans()[0].attributes
    assert json.loads(attrs[Attr.REPLAY_INPUT]) == {"q": "edited"}
    assert attrs[Attr.CONTROL_INPUT_PROVENANCE] == "edited"
    assert attrs[Attr.CONTROL_RESULT_PROVENANCE] == "live"


def test_mcp_selects_keyword_call_style_before_invocation() -> None:
    inner = _KeywordOnlyInner()
    client = MockableClient(inner, server="demo")
    with (
        patch(
            "debrix.mcp.resolve_runtime_control",
            return_value=ControlUnmanaged(),
        ),
        patch("debrix.mcp.resolve_mock", return_value=PASSTHROUGH),
    ):
        assert client.call_tool("query", arguments={"sql": "select 1"}) == (
            "keyword-live"
        )
    assert inner.calls == [("query", {"sql": "select 1"})]


def test_real_mcp_type_error_is_not_retried() -> None:
    inner = _TypeErrorInner()
    client = MockableClient(inner, server="demo")
    with (
        patch(
            "debrix.mcp.resolve_runtime_control",
            return_value=ControlUnmanaged(),
        ),
        patch("debrix.mcp.resolve_mock", return_value=PASSTHROUGH),
    ):
        with pytest.raises(TypeError, match="real failure"):
            client.call_tool("query", {"sql": "bad"})
    assert inner.calls == 1


def test_async_managed_mcp_input_invokes_edited_arguments_once() -> None:
    class AsyncInputInner:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def call_tool(
            self, name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            self.calls.append(arguments)
            await asyncio.sleep(0)
            return {"name": name, "arguments": arguments}

    inner = AsyncInputInner()
    client = MockableClient(inner, server="demo")

    async def run() -> None:
        with (
            patch(
                "debrix.mcp.resolve_runtime_control_async",
                return_value=_invoke({"sql": "select edited"}),
            ),
            patch("debrix.mcp.resolve_mock") as mock_resolver,
        ):
            assert await client.call_tool(
                "query", {"sql": "select recorded"}
            ) == {
                "name": "query",
                "arguments": {"sql": "select edited"},
            }
        mock_resolver.assert_not_called()

    asyncio.run(run())
    assert inner.calls == [{"sql": "select edited"}]


def test_mcp_input_schema_is_captured_and_defended_before_invocation() -> None:
    inner = _FakeInner()
    client = MockableClient(
        inner,
        server="demo",
        input_schemas={
            "search": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "enum": ["allowed"]},
                },
                "required": ["q"],
                "additionalProperties": False,
            }
        },
    )
    with patch(
        "debrix.mcp.resolve_runtime_control",
        return_value=_invoke({"q": "forbidden"}),
    ):
        with pytest.raises(
            DebrixControlProtocolError,
            match=r"input\.q must match the captured enum",
        ):
            client.call_tool("search", {"q": "allowed"})
    assert inner.calls == []
