"""Tests for Tool Mocker resolve + @trace_tool interception."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from debrix import Attr, trace_tool
from debrix.mocks import (
    MockDecision,
    MockError,
    MockToolError,
    PASSTHROUGH,
    apply_mock_decision,
    resolve_mock,
)


def test_apply_fixed_result() -> None:
    d = MockDecision(action="mock", result={"ok": True})
    assert apply_mock_decision(d) == {"ok": True}


def test_apply_error_raises() -> None:
    d = MockDecision(
        action="mock",
        error=MockError(kind="timeout", message="mocked tool timeout"),
    )
    with pytest.raises(MockToolError, match="mocked tool timeout") as exc:
        apply_mock_decision(d)
    assert exc.value.kind == "timeout"


def test_resolve_passthrough_when_unreachable() -> None:
    decision = resolve_mock(
        kind="tool",
        name="lookup",
        arguments={"topic": "otlp"},
        endpoint="http://127.0.0.1:1",
        timeout=0.05,
    )
    assert decision.action == "passthrough"


def test_trace_tool_uses_mock(
    memory_exporter: InMemorySpanExporter,
) -> None:
    called = {"n": 0}

    @trace_tool(name="lookup")
    def lookup(topic: str) -> str:
        called["n"] += 1
        return f"real:{topic}"

    fake = MockDecision(action="mock", result="mocked-otlp")
    with patch("debrix.tracing.resolve_mock", return_value=fake):
        assert lookup("otlp") == "mocked-otlp"

    assert called["n"] == 0
    spans = list(memory_exporter.get_finished_spans())
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs[Attr.MOCKED] == "true"
    assert json.loads(attrs[Attr.REPLAY_INPUT]) == {"topic": "otlp"}
    assert json.loads(attrs[Attr.REPLAY_OUTPUT]) == "mocked-otlp"


def test_trace_tool_passthrough(
    memory_exporter: InMemorySpanExporter,
) -> None:
    @trace_tool(name="lookup")
    def lookup(topic: str) -> str:
        return f"real:{topic}"

    with patch("debrix.tracing.resolve_mock", return_value=PASSTHROUGH):
        assert lookup("debrix") == "real:debrix"

    attrs = memory_exporter.get_finished_spans()[0].attributes
    assert Attr.MOCKED not in attrs
    assert json.loads(attrs[Attr.REPLAY_OUTPUT]) == "real:debrix"


def test_trace_tool_mock_error(
    memory_exporter: InMemorySpanExporter,
) -> None:
    @trace_tool(name="lookup")
    def lookup(topic: str) -> str:
        return "never"

    fake = MockDecision(
        action="mock",
        error=MockError(kind="timeout", message="boom"),
    )
    with patch("debrix.tracing.resolve_mock", return_value=fake):
        with pytest.raises(MockToolError):
            lookup("otlp")

    attrs = memory_exporter.get_finished_spans()[0].attributes
    assert attrs[Attr.MOCKED] == "true"
    out = json.loads(attrs[Attr.REPLAY_OUTPUT])
    assert out["error"] == "timeout"


def test_resolve_parses_mock_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def read(self) -> bytes:
            return json.dumps(
                {
                    "action": "mock",
                    "rule_id": "r1",
                    "delay_ms": 0,
                    "result": "x",
                }
            ).encode()

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        "debrix.mocks.urllib.request.urlopen",
        lambda *a, **k: _Resp(),
    )
    d = resolve_mock(kind="tool", name="lookup", arguments={"topic": "a"})
    assert d.action == "mock"
    assert d.result == "x"
    assert d.rule_id == "r1"
