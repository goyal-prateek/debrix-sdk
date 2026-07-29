"""Tests for debrix.llm.complete (Mode B replay / mock resolve)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from debrix import Attr, SpanKind, Stub, trace_agent, trace_tool
from debrix.control import (
    ControlInvoke,
    ControlResolvedInput,
    ControlResolvedValue,
    ControlReturn,
    ControlUnmanaged,
    DebrixBreakpointCancelled,
)
from debrix.llm import acomplete, complete
from debrix.mocks import MockDecision, PASSTHROUGH


def message_invoke(value: object, provenance: str = "edited") -> ControlInvoke:
    return ControlInvoke(
        attempt_id="branch_attempt_123",
        branch_id="branch_123",
        occurrence_id="occurrence_123",
        decision_id="decision_123",
        input=ControlResolvedInput(
            provenance=provenance,  # type: ignore[arg-type]
            value=value,
        ),
        capture_live_result=True,
    )


def model_return(value: object, provenance: str = "edited") -> ControlReturn:
    return ControlReturn(
        attempt_id="branch_attempt_123",
        branch_id="branch_123",
        occurrence_id="occurrence_123",
        decision_id="decision_123",
        live_suffix=True,
        output=ControlResolvedValue(
            provenance=provenance,  # type: ignore[arg-type]
            kind="result",
            value=value,
        ),
    )


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
    with patch("debrix.llm.resolve_mock", return_value=fake) as resolve:
        out = complete(
            [{"role": "user", "content": "hi"}],
            call=live,
        )
    assert out == "recorded"
    assert called["n"] == 0
    span = memory_exporter.get_finished_spans()[0]
    assert resolve.call_args.kwargs["trace_id"] == format(
        span.context.trace_id, "032x"
    )
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


def test_complete_controlled_messages_call_provider_exactly_once(
    memory_exporter: InMemorySpanExporter,
) -> None:
    calls: list[list[dict[str, object]]] = []
    edited = {
        "messages": [
            {
                "role": "developer",
                "content": "Use approved tools",
                "metadata": {"priority": 3},
            },
            {
                "role": "assistant",
                "content": "Calling lookup",
                "tool_calls": [{"id": "call_2", "arguments": {"limit": 3}}],
            },
        ]
    }

    def live(
        messages: list[dict[str, object]],
    ) -> tuple[str, dict[str, int], str]:
        calls.append(messages)
        return "from-provider", {"input_tokens": 2, "output_tokens": 3}, "m"

    with (
        patch(
            "debrix.llm.resolve_runtime_control",
            side_effect=[message_invoke(edited), ControlUnmanaged()],
        ) as control,
        patch("debrix.llm.resolve_mock") as mock,
    ):
        out = complete(
            [
                {
                    "role": "developer",
                    "content": "Use tools",
                    "metadata": {"priority": 2},
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_1", "arguments": {"limit": 2}}
                    ],
                },
            ],
            call=live,
        )

    assert out == "from-provider"
    assert calls == [edited["messages"]]
    mock.assert_not_called()
    assert control.call_count == 2
    request = control.call_args_list[0].kwargs
    assert request["capabilities"] == ("messages", "model_output")
    assert request["input_descriptor"]["operationKind"] == "llm"
    post_request = control.call_args_list[1].kwargs
    assert post_request["capabilities"] == ("model_output",)
    assert post_request["input_value"]["value"]["content"] == "from-provider"
    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes[Attr.CONTROL_INPUT_PROVENANCE] == "edited"
    assert span.attributes[Attr.CONTROL_RESULT_PROVENANCE] == "live"
    assert json.loads(span.attributes[Attr.MESSAGES])[0]["content"] == (
        "Use approved tools"
    )


def test_complete_controlled_model_output_skips_provider_and_mock(
    memory_exporter: InMemorySpanExporter,
) -> None:
    calls = {"provider": 0}
    response = {
        "content": "controlled answer",
        "model": "provider-model",
        "usage": {"input_tokens": 2, "output_tokens": 4},
        "metadata": {"finish_reason": "length"},
    }

    def live(messages: list[dict[str, object]]) -> tuple[str, dict[str, int], str]:
        calls["provider"] += 1
        return "must-not-run", {}, "m"

    with (
        patch(
            "debrix.llm.resolve_runtime_control",
            return_value=model_return(response),
        ),
        patch("debrix.llm.resolve_mock") as mock,
    ):
        out = complete([{"role": "user", "content": "question"}], call=live)

    assert out == "controlled answer"
    assert calls["provider"] == 0
    mock.assert_not_called()
    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes[Attr.CONTROL_RESULT_PROVENANCE] == "edited"
    assert json.loads(span.attributes[Attr.RESPONSE]) == response


def test_downstream_model_output_runs_live_after_an_earlier_edit(
    memory_exporter: InMemorySpanExporter,
) -> None:
    provider_calls: list[list[dict[str, object]]] = []
    live_response = {
        "content": "live downstream answer",
        "model": "live-model",
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }

    def provider(
        messages: list[dict[str, object]],
    ) -> tuple[str, dict[str, int], str]:
        provider_calls.append(messages)
        return (
            str(live_response["content"]),
            live_response["usage"],  # type: ignore[arg-type]
            str(live_response["model"]),
        )

    @trace_agent(name="coordinator")
    def run() -> tuple[str, str]:
        edited = complete(
            [{"role": "user", "content": "policy"}],
            name="policy_interpret_result",
            call=provider,
        )
        downstream = complete(
            [{"role": "user", "content": edited}],
            name="decide_support_resolution",
            call=provider,
        )
        return edited, downstream

    with patch(
        "debrix.llm.resolve_runtime_control",
        side_effect=[
            model_return(
                {
                    "content": "edited policy",
                    "model": "baseline-model",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ),
            ControlUnmanaged(),
            model_return(live_response, provenance="live"),
        ],
    ) as control:
        assert run() == ("edited policy", "live downstream answer")

    assert len(provider_calls) == 1
    assert control.call_args_list[1].kwargs["capabilities"] == ("messages",)
    assert control.call_args_list[2].kwargs["capabilities"] == ("model_output",)
    assert control.call_args_list[2].kwargs["input_value"] == {
        "kind": "result",
        "value": live_response,
    }


def test_acomplete_controlled_messages_call_provider_exactly_once(
    memory_exporter: InMemorySpanExporter,
) -> None:
    calls: list[list[dict[str, object]]] = []
    edited = {
        "messages": [{"role": "user", "content": "edited question"}]
    }

    async def live(
        messages: list[dict[str, object]],
    ) -> tuple[str, dict[str, int], str]:
        calls.append(messages)
        return "async-live", {"input_tokens": 1, "output_tokens": 2}, "m"

    async def controlled(*args: object, **kwargs: object) -> ControlInvoke:
        return message_invoke(edited)

    with (
        patch(
            "debrix.llm.resolve_runtime_control_async",
            side_effect=controlled,
        ),
        patch("debrix.llm.resolve_mock") as mock,
    ):
        out = asyncio.run(
            acomplete(
                [{"role": "user", "content": "question"}],
                call=live,
            )
        )

    assert out == "async-live"
    assert calls == [edited["messages"]]
    mock.assert_not_called()
    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes[Attr.CONTROL_INPUT_PROVENANCE] == "edited"
    assert span.attributes[Attr.CONTROL_RESULT_PROVENANCE] == "live"


def test_acomplete_controlled_model_output_calls_provider_zero_times(
    memory_exporter: InMemorySpanExporter,
) -> None:
    calls = {"provider": 0}
    response = {
        "content": "async controlled",
        "model": "provider-model",
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }

    async def live(
        messages: list[dict[str, object]],
    ) -> tuple[str, dict[str, int], str]:
        calls["provider"] += 1
        return "must-not-run", {}, "m"

    async def controlled(*args: object, **kwargs: object) -> ControlReturn:
        return model_return(response)

    with (
        patch(
            "debrix.llm.resolve_runtime_control_async",
            side_effect=controlled,
        ),
        patch("debrix.llm.resolve_mock") as mock,
    ):
        out = asyncio.run(
            acomplete(
                [{"role": "user", "content": "question"}],
                call=live,
            )
        )

    assert out == "async controlled"
    assert calls["provider"] == 0
    mock.assert_not_called()
    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes[Attr.CONTROL_RESULT_PROVENANCE] == "edited"


def test_controlled_message_provider_exception_is_not_retried(
    memory_exporter: InMemorySpanExporter,
) -> None:
    calls = {"provider": 0}

    def live(messages: list[dict[str, object]]) -> tuple[str, dict[str, int], str]:
        calls["provider"] += 1
        raise RuntimeError("provider failed")

    with (
        patch(
            "debrix.llm.resolve_runtime_control",
            return_value=message_invoke(
                {"messages": [{"role": "user", "content": "edited"}]}
            ),
        ),
        patch("debrix.llm.resolve_mock") as mock,
    ):
        with pytest.raises(RuntimeError, match="provider failed"):
            complete(
                [{"role": "user", "content": "recorded"}],
                call=live,
            )

    assert calls["provider"] == 1
    mock.assert_not_called()
    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes[Attr.CONTROL_RESULT_PROVENANCE] == "live"
    assert "provider failed" in span.attributes[Attr.ERROR_SUMMARY]


def test_acomplete_provider_exception_is_not_retried_and_wait_yields(
    memory_exporter: InMemorySpanExporter,
) -> None:
    calls = {"provider": 0}
    heartbeat = {"ticks": 0}

    async def live(
        messages: list[dict[str, object]],
    ) -> tuple[str, dict[str, int], str]:
        calls["provider"] += 1
        raise RuntimeError("async provider failed")

    async def controlled(*args: object, **kwargs: object) -> ControlInvoke:
        await asyncio.sleep(0)
        return message_invoke(
            {"messages": [{"role": "user", "content": "edited"}]}
        )

    async def scenario() -> None:
        async def tick() -> None:
            heartbeat["ticks"] += 1
            await asyncio.sleep(0)
            heartbeat["ticks"] += 1

        with patch(
            "debrix.llm.resolve_runtime_control_async",
            side_effect=controlled,
        ):
            with pytest.raises(RuntimeError, match="async provider failed"):
                await asyncio.gather(
                    acomplete(
                        [{"role": "user", "content": "recorded"}],
                        call=live,
                    ),
                    tick(),
                )

    asyncio.run(scenario())
    assert calls["provider"] == 1
    assert heartbeat["ticks"] == 2
    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes[Attr.CONTROL_RESULT_PROVENANCE] == "live"


def test_control_cancellation_calls_neither_mock_nor_provider() -> None:
    calls = {"provider": 0}

    def live(messages: list[dict[str, object]]) -> tuple[str, dict[str, int], str]:
        calls["provider"] += 1
        return "must-not-run", {}, "m"

    with (
        patch(
            "debrix.llm.resolve_runtime_control",
            side_effect=DebrixBreakpointCancelled("cancelled_by_controller"),
        ),
        patch("debrix.llm.resolve_mock") as mock,
    ):
        with pytest.raises(
            DebrixBreakpointCancelled,
            match="cancelled_by_controller",
        ):
            complete(
                [{"role": "user", "content": "recorded"}],
                call=live,
            )

    assert calls["provider"] == 0
    mock.assert_not_called()


def test_large_controlled_llm_payloads_use_full_blob_capture_without_inline_replay(
    memory_exporter: InMemorySpanExporter,
) -> None:
    large_input = "i" * 70_000
    large_output = "o" * 70_000
    response = {
        "content": large_output,
        "model": "provider-model",
        "usage": {"input_tokens": 70_000, "output_tokens": 70_000},
    }

    with (
        patch(
            "debrix.llm.resolve_runtime_control",
            return_value=model_return(response, provenance="recorded"),
        ),
        patch("debrix.llm.resolve_mock") as mock,
    ):
        assert (
            complete([{"role": "user", "content": large_input}])
            == large_output
        )

    mock.assert_not_called()
    attrs = memory_exporter.get_finished_spans()[0].attributes
    assert Attr.MESSAGES not in attrs
    assert Attr.RESPONSE not in attrs
    assert Attr.REPLAY_INPUT not in attrs
    assert Attr.REPLAY_OUTPUT not in attrs
    assert attrs[Attr.MESSAGES_BLOB_REF].startswith("sha256:")
    assert attrs[Attr.RESPONSE_BLOB_REF].startswith("sha256:")
    assert attrs[Attr.MESSAGES_TRUNCATED] is True
    assert attrs[Attr.RESPONSE_TRUNCATED] is True
