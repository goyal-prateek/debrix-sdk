"""Tests for Phase 1A instrumentation: decorators, nesting, record_*."""

from __future__ import annotations

import asyncio
import json

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from debrix import (
    Attr,
    SpanKind,
    trace_agent,
    trace_span,
    trace_tool,
)
from debrix.span import DebrixSpan


def _by_name(exporter: InMemorySpanExporter) -> dict[str, object]:
    return {s.name: s for s in exporter.get_finished_spans()}


def test_agent_tool_llm_nesting(memory_exporter: InMemorySpanExporter) -> None:
    @trace_agent
    def run_agent(query: str) -> str:
        return research(query)

    @trace_tool(name="search")
    def research(query: str) -> str:
        with trace_span("complete", kind=SpanKind.LLM) as span:
            span.record_messages(
                [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": query},
                ]
            )
            answer = f"answer:{query}"
            span.record_response(
                {
                    "content": answer,
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            )
            return answer

    result = run_agent("hello")
    assert result == "answer:hello"

    spans = list(memory_exporter.get_finished_spans())
    assert len(spans) == 3

    by_name = _by_name(memory_exporter)
    agent = by_name["run_agent"]
    tool = by_name["search"]
    llm = by_name["complete"]

    assert agent.attributes[Attr.SPAN_KIND] == SpanKind.AGENT
    assert agent.attributes[Attr.AGENT_NAME] == "run_agent"
    assert tool.attributes[Attr.SPAN_KIND] == SpanKind.TOOL
    assert tool.attributes[Attr.TOOL_NAME] == "search"
    assert json.loads(tool.attributes[Attr.REPLAY_INPUT]) == {"query": "hello"}
    assert json.loads(tool.attributes[Attr.REPLAY_OUTPUT]) == "answer:hello"
    assert tool.attributes[Attr.REPLAY_SEQUENCE_INDEX] == 0
    assert Attr.REPLAY_INPUT not in agent.attributes
    assert llm.attributes[Attr.SPAN_KIND] == SpanKind.LLM

    assert tool.parent is not None
    assert tool.parent.span_id == agent.context.span_id
    assert llm.parent is not None
    assert llm.parent.span_id == tool.context.span_id
    assert (
        agent.context.trace_id
        == tool.context.trace_id
        == llm.context.trace_id
    )


def test_record_messages_and_response_round_trip(
    memory_exporter: InMemorySpanExporter,
) -> None:
    with trace_span("llm", kind=SpanKind.LLM) as span:
        assert isinstance(span, DebrixSpan)
        span.record_messages(
            [
                {"role": "user", "content": "hi"},
                {"role": "tool", "content": "{}", "name": "search"},
            ]
        )
        span.record_response(
            {"content": "yo", "model": "test", "usage": {"input_tokens": 1}}
        )

    finished = memory_exporter.get_finished_spans()
    assert len(finished) == 1
    attrs = finished[0].attributes
    messages = json.loads(attrs[Attr.MESSAGES])
    assert messages == [
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": "{}", "name": "search"},
    ]
    response = json.loads(attrs[Attr.RESPONSE])
    assert response["content"] == "yo"
    assert response["model"] == "test"
    assert response["usage"]["input_tokens"] == 1


def test_record_messages_rejects_bad_role(
    memory_exporter: InMemorySpanExporter,
) -> None:
    with trace_span("bad") as span:
        with pytest.raises(ValueError, match="role"):
            span.record_messages([{"role": "system_prompt", "content": "x"}])


def test_exception_sets_error_status_and_summary(
    memory_exporter: InMemorySpanExporter,
) -> None:
    @trace_tool
    def boom(reason: str) -> None:
        raise RuntimeError("tool failed")

    with pytest.raises(RuntimeError, match="tool failed"):
        boom("bad-arg")

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[Attr.ERROR_SUMMARY] == "RuntimeError: tool failed"
    assert span.attributes[Attr.SPAN_KIND] == SpanKind.TOOL
    assert span.attributes[Attr.TOOL_NAME] == "boom"
    assert json.loads(span.attributes[Attr.REPLAY_INPUT]) == {"reason": "bad-arg"}
    assert Attr.REPLAY_OUTPUT not in span.attributes
    assert any(e.name == "exception" for e in span.events)


def test_trace_tool_records_named_args_and_defaults(
    memory_exporter: InMemorySpanExporter,
) -> None:
    @trace_tool(name="lookup")
    def lookup(topic: str, *, limit: int = 3) -> dict[str, object]:
        return {"topic": topic, "limit": limit}

    assert lookup("Debrix") == {"topic": "Debrix", "limit": 3}

    span = memory_exporter.get_finished_spans()[0]
    assert json.loads(span.attributes[Attr.REPLAY_INPUT]) == {
        "topic": "Debrix",
        "limit": 3,
    }
    assert json.loads(span.attributes[Attr.REPLAY_OUTPUT]) == {
        "topic": "Debrix",
        "limit": 3,
    }


def test_trace_tool_skips_self_and_repr_non_json(
    memory_exporter: InMemorySpanExporter,
) -> None:
    class Box:
        def __init__(self, value: str) -> None:
            self.value = value

        def __repr__(self) -> str:
            return f"Box({self.value!r})"

    class Tools:
        @trace_tool
        def echo(self, box: Box) -> Box:
            return box

    result = Tools().echo(Box("x"))
    assert repr(result) == "Box('x')"

    span = memory_exporter.get_finished_spans()[0]
    assert json.loads(span.attributes[Attr.REPLAY_INPUT]) == {"box": "Box('x')"}
    assert json.loads(span.attributes[Attr.REPLAY_OUTPUT]) == "Box('x')"


def test_async_decorators(memory_exporter: InMemorySpanExporter) -> None:
    @trace_agent(name="async_agent")
    async def run() -> str:
        return await tool()

    @trace_tool
    async def tool() -> str:
        return "ok"

    assert asyncio.run(run()) == "ok"
    by_name = _by_name(memory_exporter)
    assert "async_agent" in by_name
    assert "tool" in by_name
    assert (
        by_name["tool"].parent.span_id
        == by_name["async_agent"].context.span_id
    )


def test_bare_and_named_decorators(
    memory_exporter: InMemorySpanExporter,
) -> None:
    @trace_agent
    def bare_agent() -> str:
        return "a"

    @trace_agent(name="named_agent")
    def named() -> str:
        return "b"

    assert bare_agent() == "a"
    assert named() == "b"

    by_name = _by_name(memory_exporter)
    assert by_name["bare_agent"].attributes[Attr.AGENT_NAME] == "bare_agent"
    assert by_name["named_agent"].attributes[Attr.AGENT_NAME] == "named_agent"


def test_context_manager_forms(memory_exporter: InMemorySpanExporter) -> None:
    with trace_agent("planner") as agent_span:
        agent_span.set_attribute("custom.key", "v")
        with trace_tool("lookup") as tool_span:
            tool_span.record_messages([{"role": "user", "content": "q"}])

    by_name = _by_name(memory_exporter)
    assert by_name["planner"].attributes[Attr.SPAN_KIND] == SpanKind.AGENT
    assert by_name["lookup"].attributes[Attr.SPAN_KIND] == SpanKind.TOOL
    assert (
        by_name["lookup"].parent.span_id
        == by_name["planner"].context.span_id
    )


def test_trace_span_default_kind_is_custom(
    memory_exporter: InMemorySpanExporter,
) -> None:
    with trace_span("misc"):
        pass
    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes[Attr.SPAN_KIND] == SpanKind.CUSTOM
