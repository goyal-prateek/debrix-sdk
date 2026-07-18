"""Tests for debrix.llm.complete (Mode B replay / mock resolve)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from debrix import Attr, SpanKind, Stub, trace_agent, trace_tool
from debrix.llm import complete
from debrix.mocks import MockDecision, PASSTHROUGH


def test_complete_replay_short_circuits(
    memory_exporter: InMemorySpanExporter,
) -> None:
    called = {"n": 0}

    def live(messages: list) -> tuple[str, dict[str, int], str]:
        called["n"] += 1
        return "live", {"input_tokens": 1, "output_tokens": 1}, "live-model"

    fake = MockDecision(
        action="replay",
        result={"content": "recorded", "model": "tape", "usage": {}},
    )
    with patch("debrix.llm.resolve_mock", return_value=fake):
        out = complete(
            [{"role": "user", "content": "hi"}],
            call=live,
        )
    assert out == "recorded"
    assert called["n"] == 0
    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes[Attr.SPAN_KIND] == SpanKind.LLM
    assert span.attributes[Attr.STUB] == Stub.REPLAY
    assert isinstance(span.attributes[Attr.REPLAY_SEQUENCE_INDEX], int)
    assert json.loads(span.attributes[Attr.REPLAY_OUTPUT])["content"] == "recorded"


def test_complete_passthrough_calls_live(
    memory_exporter: InMemorySpanExporter,
) -> None:
    def live(messages: list) -> tuple[str, dict[str, int], str]:
        return "from-live", {"input_tokens": 2, "output_tokens": 3}, "m"

    with patch("debrix.llm.resolve_mock", return_value=PASSTHROUGH):
        out = complete([{"role": "user", "content": "x"}], call=live)
    assert out == "from-live"
    attrs = memory_exporter.get_finished_spans()[0].attributes
    assert Attr.STUB not in attrs
    assert json.loads(attrs[Attr.REPLAY_OUTPUT])["content"] == "from-live"


def test_complete_requires_call_on_passthrough() -> None:
    with patch("debrix.llm.resolve_mock", return_value=PASSTHROUGH):
        with pytest.raises(RuntimeError, match="requires call="):
            complete([{"role": "user", "content": "x"}])


def test_complete_sequence_interleaved_with_tools(
    memory_exporter: InMemorySpanExporter,
) -> None:
    @trace_tool(name="lookup")
    def lookup() -> str:
        return "fact"

    @trace_agent(name="agent")
    def run() -> str:
        with patch("debrix.tracing.resolve_mock", return_value=PASSTHROUGH):
            lookup()
        with patch("debrix.llm.resolve_mock", return_value=PASSTHROUGH):

            def live(messages: list) -> tuple[str, dict[str, int], str]:
                return "ok", {}, "stub"

            return complete([{"role": "user", "content": "q"}], call=live)

    run()
    by_kind: dict[str, list] = {}
    for span in memory_exporter.get_finished_spans():
        kind = span.attributes.get(Attr.SPAN_KIND)
        by_kind.setdefault(kind, []).append(span)
    tool = by_kind[SpanKind.TOOL][0]
    llm = by_kind[SpanKind.LLM][0]
    assert tool.attributes[Attr.REPLAY_SEQUENCE_INDEX] == 0
    assert llm.attributes[Attr.REPLAY_SEQUENCE_INDEX] == 1
