"""End-to-end wrapper tests for FW2.3 result/error control decisions."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from debrix import (
    Attr,
    MockToolError,
    MockableClient,
    trace_agent,
    trace_span,
    trace_tool,
)
from debrix.control import (
    ControlErrorValue,
    ControlInvoke,
    ControlResolvedInput,
    ControlResolvedValue,
    ControlReturn,
    ControlUnmanaged,
    DebrixControlProtocolError,
)
from debrix.mocks import PASSTHROUGH
from debrix.runtime_control import (
    capture_bound_call,
    capture_llm_messages,
    resolve_runtime_control,
    validate_model_output,
)


def returned(
    *,
    provenance: str = "recorded",
    kind: str = "result",
    value: Any = None,
    error: ControlErrorValue | None = None,
    live_suffix: bool = True,
) -> ControlReturn:
    return ControlReturn(
        attempt_id="branch_attempt_123",
        branch_id="branch_123",
        occurrence_id="occurrence_123",
        decision_id="decision_123",
        live_suffix=live_suffix,
        output=ControlResolvedValue(
            provenance=provenance,  # type: ignore[arg-type]
            kind=kind,  # type: ignore[arg-type]
            value=value,
            error=error,
        ),
    )


def invoked(
    value: Any,
    *,
    provenance: str = "edited",
    capture_live_result: bool = True,
) -> ControlInvoke:
    return ControlInvoke(
        attempt_id="branch_attempt_123",
        branch_id="branch_123",
        occurrence_id="occurrence_123",
        decision_id="decision_123",
        input=ControlResolvedInput(
            provenance=provenance,  # type: ignore[arg-type]
            value=value,
        ),
        capture_live_result=capture_live_result,
    )


def test_bound_call_reconstructs_every_python_parameter_kind() -> None:
    def shaped(
        positional: int,
        /,
        regular: str = "default",
        *items: int,
        enabled: bool = True,
        **options: str,
    ) -> tuple[Any, ...]:
        return positional, regular, items, enabled, options

    captured = capture_bound_call(
        shaped,
        (1, "original", 2, 3),
        {"enabled": False, "region": "west"},
    )

    assert captured is not None
    assert captured.recorded_input == {
        "positional": 1,
        "regular": "original",
        "items": [2, 3],
        "enabled": False,
        "options": {"region": "west"},
    }
    assert [
        (parameter["name"], parameter["kind"], parameter["required"])
        for parameter in captured.descriptor["parameters"]
    ] == [
        ("positional", "positional_only", True),
        ("regular", "positional_or_keyword", False),
        ("items", "var_positional", False),
        ("enabled", "keyword_only", False),
        ("options", "var_keyword", False),
    ]

    args, kwargs = captured.reconstruct(
        {
            "positional": 10,
            "regular": "edited",
            "items": [20, 30],
            "enabled": True,
            "options": {"region": "east", "mode": "safe"},
        }
    )

    assert shaped(*args, **kwargs) == (
        10,
        "edited",
        (20, 30),
        True,
        {"region": "east", "mode": "safe"},
    )


def test_llm_messages_round_trip_complete_provider_shape() -> None:
    messages = [
        {
            "role": "system",
            "content": "Be concise.",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "lookup",
                        "arguments": {"query": "Paris", "limit": 2},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": {"temperature": 21},
            "tool_call_id": "call_1",
        },
        {
            "role": "provider_custom",
            "content": [{"type": "text", "text": "Continue"}],
            "metadata": {"safe": True},
        },
    ]
    captured = capture_llm_messages(messages)

    assert captured is not None
    assert captured.recorded_input == {"messages": messages}
    assert captured.descriptor == {
        "schemaVersion": 1,
        "operationKind": "llm",
        "jsonKind": "object",
        "parameters": [
            {
                "name": "messages",
                "kind": "sequence",
                "required": True,
                "hasDefault": False,
                "editable": True,
                "jsonKind": "array",
                "pythonType": "builtins.list",
            }
        ],
    }

    edited = {"messages": json.loads(json.dumps(messages))}
    edited["messages"][0]["content"] = "Be precise."
    edited["messages"][1]["tool_calls"][0]["function"]["arguments"]["limit"] = 3
    assert captured.reconstruct(edited) == edited["messages"]


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda value: value["messages"].append(
                {"role": "user", "content": "extra"}
            ),
            "entry count",
        ),
        (
            lambda value: value["messages"][0].update(role="assistant"),
            "role must remain system",
        ),
        (
            lambda value: value["messages"][0].update(extra=True),
            "keys must exactly match",
        ),
        (
            lambda value: value["messages"][0].update(content=["changed kind"]),
            "content must remain JSON string",
        ),
    ],
)
def test_llm_messages_reject_role_or_recursive_shape_changes(
    mutate: Any,
    match: str,
) -> None:
    captured = capture_llm_messages(
        [{"role": "system", "content": "recorded"}]
    )
    assert captured is not None
    edited = json.loads(json.dumps(captured.recorded_input))
    mutate(edited)

    with pytest.raises(ValueError, match=match):
        captured.reconstruct(edited)


def test_model_output_preserves_complete_recursive_shape() -> None:
    recorded = {
        "content": "recorded",
        "model": "provider-model",
        "usage": {"input_tokens": 2, "output_tokens": 3},
        "tool_calls": [
            {"id": "call_1", "arguments": {"query": "Paris"}},
        ],
        "metadata": {"finish_reason": "stop"},
    }
    edited = json.loads(json.dumps(recorded))
    edited["content"] = "edited"
    edited["usage"]["output_tokens"] = 4
    assert validate_model_output(recorded, edited) == edited

    wrong = json.loads(json.dumps(recorded))
    wrong["usage"]["output_tokens"] = "four"
    with pytest.raises(ValueError, match="output.usage.output_tokens"):
        validate_model_output(recorded, wrong)

    missing = json.loads(json.dumps(recorded))
    del missing["metadata"]
    with pytest.raises(ValueError, match="keys must exactly match"):
        validate_model_output(recorded, missing)


def test_bound_call_keeps_receivers_immutable_and_out_of_recorded_input() -> None:
    class Worker:
        def run(self, value: str, *, limit: int = 1) -> tuple[str, int]:
            return value, limit

    worker = Worker()
    captured = capture_bound_call(
        Worker.run,
        (worker, "recorded"),
        {"limit": 2},
    )

    assert captured is not None
    assert captured.recorded_input == {"value": "recorded", "limit": 2}
    assert captured.descriptor["parameters"][0] == {
        "name": "self",
        "kind": "positional_or_keyword",
        "required": True,
        "hasDefault": False,
        "editable": False,
        "jsonKind": None,
        "pythonType": f"{Worker.__module__}.{Worker.__qualname__}",
    }
    with pytest.raises(
        ValueError,
        match="input keys must exactly match editable parameters",
    ):
        captured.reconstruct({"self": "replacement", "value": "x", "limit": 2})

    args, kwargs = captured.reconstruct({"value": "edited", "limit": 5})
    assert args[0] is worker
    assert Worker.run(*args, **kwargs) == ("edited", 5)


@pytest.mark.parametrize(
    ("edited", "message"),
    [
        ({"value": "x"}, "input keys must exactly match editable parameters"),
        (
            {"value": "x", "limit": "many"},
            "input parameter limit must remain JSON number",
        ),
    ],
)
def test_bound_call_rejects_missing_or_wrong_kind_edits(
    edited: dict[str, Any],
    message: str,
) -> None:
    def lookup(value: str, *, limit: int = 1) -> str:
        return value * limit

    captured = capture_bound_call(lookup, ("recorded",), {})

    assert captured is not None
    with pytest.raises(ValueError, match=message):
        captured.reconstruct(edited)


def test_runtime_request_advertises_input_only_with_a_descriptor() -> None:
    descriptor = {
        "schemaVersion": 1,
        "operationKind": "tool",
        "jsonKind": "object",
        "parameters": [],
    }
    with (
        trace_span("lookup") as span,
        patch(
            "debrix.runtime_control.resolve_control",
            return_value=ControlUnmanaged(),
        ) as resolver,
    ):
        resolve_runtime_control(
            span,
            operation_kind="tool",
            operation_name="lookup",
            operation_server=None,
            agent_scope="planner",
            sequence_index=0,
            input_value={},
            input_descriptor=descriptor,
        )

    request = resolver.call_args.args[0]
    assert request.capabilities == ("input", "result", "error")
    assert request.input_descriptor == descriptor


def test_llm_runtime_request_advertises_message_and_output_capabilities() -> None:
    captured = capture_llm_messages(
        [{"role": "developer", "content": "Use tools carefully."}]
    )
    assert captured is not None
    with (
        trace_span("complete") as span,
        patch(
            "debrix.runtime_control.resolve_control",
            return_value=ControlUnmanaged(),
        ) as resolver,
    ):
        resolve_runtime_control(
            span,
            operation_kind="llm",
            operation_name="complete",
            operation_server=None,
            agent_scope="planner",
            sequence_index=2,
            input_value=captured.recorded_input,
            input_descriptor=captured.descriptor,
            capabilities=("messages", "model_output"),
        )

    request = resolver.call_args.args[0]
    assert request.capabilities == ("messages", "model_output")
    assert request.input_descriptor == captured.descriptor
    assert request.input_value == captured.recorded_input


def test_sync_tool_invoke_reconstructs_arguments_and_calls_real_once(
    memory_exporter: InMemorySpanExporter,
) -> None:
    calls: list[tuple[Any, ...]] = []

    @trace_tool(name="shaped")
    def shaped(
        positional: int,
        /,
        regular: str = "default",
        *items: int,
        enabled: bool = True,
        **options: str,
    ) -> str:
        calls.append((positional, regular, items, enabled, options))
        return "live result"

    with (
        patch(
            "debrix.tracing.resolve_runtime_control",
            side_effect=[
                invoked(
                    {
                        "positional": 10,
                        "regular": "edited",
                        "items": [20, 30],
                        "enabled": False,
                        "options": {"region": "east"},
                    }
                ),
                ControlUnmanaged(),
            ],
        ) as control,
        patch("debrix.tracing.resolve_mock") as mock_resolver,
    ):
        assert shaped(1, "original", 2, enabled=True, region="west") == (
            "live result"
        )

    assert calls == [(10, "edited", (20, 30), False, {"region": "east"})]
    mock_resolver.assert_not_called()
    request = control.call_args_list[0].kwargs
    assert request["input_descriptor"]["schemaVersion"] == 1
    assert request["input_value"]["positional"] == 1
    attrs = memory_exporter.get_finished_spans()[0].attributes
    assert json.loads(attrs[Attr.REPLAY_INPUT]) == {
        "positional": 10,
        "regular": "edited",
        "items": [20, 30],
        "enabled": False,
        "options": {"region": "east"},
    }
    assert attrs[Attr.CONTROL_INPUT_PROVENANCE] == "edited"
    assert attrs[Attr.CONTROL_RESULT_PROVENANCE] == "live"


def test_invalid_sdk_invoke_payload_never_calls_the_tool(
    memory_exporter: InMemorySpanExporter,
) -> None:
    calls = 0

    @trace_tool(name="lookup")
    def lookup(query: str) -> str:
        nonlocal calls
        calls += 1
        return query

    with patch(
        "debrix.tracing.resolve_runtime_control",
        return_value=invoked({"wrong": "value"}),
    ):
        with pytest.raises(
            DebrixControlProtocolError,
            match="input keys must exactly match editable parameters",
        ):
            lookup("recorded")

    assert calls == 0


def test_sync_method_input_keeps_self_and_class_receivers_immutable() -> None:
    class Worker:
        @trace_tool(name="instance_lookup")
        def lookup(self, query: str) -> str:
            return f"instance:{query}"

        @classmethod
        @trace_tool(name="class_lookup")
        def class_lookup(cls, query: str) -> str:
            return f"{cls.__name__}:{query}"

    with patch(
        "debrix.tracing.resolve_runtime_control",
            side_effect=[
                invoked({"query": "edited-instance"}),
                ControlUnmanaged(),
                invoked({"query": "edited-class"}),
                ControlUnmanaged(),
            ],
    ):
        assert Worker().lookup("recorded") == "instance:edited-instance"
        assert Worker.class_lookup("recorded") == "Worker:edited-class"


def test_async_tool_input_invokes_once_without_blocking_the_event_loop() -> None:
    calls: list[str] = []

    @trace_tool(name="async_lookup")
    async def lookup(query: str) -> str:
        calls.append(query)
        await asyncio.sleep(0)
        return f"live:{query}"

    async def run() -> None:
        ticked = False

        async def tick() -> None:
            nonlocal ticked
            await asyncio.sleep(0)
            ticked = True

        with patch(
            "debrix.tracing.resolve_runtime_control_async",
            side_effect=[
                invoked({"query": "edited"}),
                ControlUnmanaged(),
            ],
        ):
            result, _ = await asyncio.gather(lookup("recorded"), tick())
        assert result == "live:edited"
        assert ticked

    asyncio.run(run())
    assert calls == ["edited"]


def test_tool_input_schema_is_defended_before_invocation() -> None:
    calls = 0

    @trace_tool(
        name="lookup",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "enum": ["allowed"]},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    def lookup(query: str) -> str:
        nonlocal calls
        calls += 1
        return query

    with patch(
        "debrix.tracing.resolve_runtime_control",
        return_value=invoked({"query": "forbidden"}),
    ):
        with pytest.raises(
            DebrixControlProtocolError,
            match=r"input\.query must match the captured enum",
        ):
            lookup("allowed")
    assert calls == 0


def test_sync_tool_controlled_result_skips_real_and_mock(
    memory_exporter: InMemorySpanExporter,
) -> None:
    calls = 0

    @trace_tool(name="lookup")
    def lookup(topic: str) -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"answer": 0}

    with (
        patch(
            "debrix.tracing.resolve_runtime_control",
            return_value=returned(
                provenance="edited",
                value={"answer": 42},
            ),
        ),
        patch("debrix.tracing.resolve_mock") as mock_resolver,
    ):
        assert lookup("debrix") == {"answer": 42}

    assert calls == 0
    mock_resolver.assert_not_called()
    attrs = memory_exporter.get_finished_spans()[0].attributes
    assert attrs[Attr.CONTROL_PROVENANCE] == "edited"
    assert attrs[Attr.CONTROL_BRANCH_ID] == "branch_123"
    assert json.loads(attrs[Attr.REPLAY_OUTPUT]) == {"answer": 42}


def test_sync_tool_controlled_error_raises_supported_envelope(
    memory_exporter: InMemorySpanExporter,
) -> None:
    calls = 0

    @trace_tool(name="lookup")
    def lookup() -> str:
        nonlocal calls
        calls += 1
        return "never"

    with patch(
        "debrix.tracing.resolve_runtime_control",
        return_value=returned(
            kind="error",
            error=ControlErrorValue(kind="RuntimeError", message="recorded boom"),
        ),
    ):
        with pytest.raises(MockToolError, match="recorded boom") as error:
            lookup()

    assert error.value.kind == "RuntimeError"
    assert calls == 0
    attrs = memory_exporter.get_finished_spans()[0].attributes
    assert attrs[Attr.CONTROL_PROVENANCE] == "recorded"
    assert json.loads(attrs[Attr.REPLAY_OUTPUT]) == {
        "error": "RuntimeError",
        "message": "recorded boom",
    }


def test_live_suffix_marks_later_operation_and_runs_it_once(
    memory_exporter: InMemorySpanExporter,
) -> None:
    calls: list[str] = []

    @trace_tool(name="selected")
    def selected() -> str:
        calls.append("selected")
        return "never"

    @trace_tool(name="downstream")
    def downstream() -> str:
        calls.append("downstream")
        return "live"

    @trace_agent(name="planner")
    def run() -> tuple[str, str]:
        return selected(), downstream()

    with (
        patch(
            "debrix.tracing.resolve_runtime_control",
            side_effect=[
                returned(
                    provenance="edited",
                    value="recorded",
                    live_suffix=True,
                ),
                ControlUnmanaged(),
                ControlUnmanaged(),
            ],
        ),
        patch("debrix.tracing.resolve_mock", return_value=PASSTHROUGH),
    ):
        assert run() == ("recorded", "live")

    assert calls == ["downstream"]
    spans = {
        span.name: span
        for span in memory_exporter.get_finished_spans()
    }
    assert spans["selected"].attributes[Attr.CONTROL_PROVENANCE] == "edited"
    assert spans["downstream"].attributes[Attr.CONTROL_PROVENANCE] == "live"
    assert spans["downstream"].attributes[Attr.CONTROL_ATTEMPT_ID] == (
        "branch_attempt_123"
    )


class _SyncMcp:
    def __init__(self) -> None:
        self.calls = 0

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls += 1
        return f"live:{name}"


class _AsyncMcp:
    def __init__(self) -> None:
        self.calls = 0

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls += 1
        return f"live:{name}"


def test_sync_mcp_controlled_result_skips_server(
    memory_exporter: InMemorySpanExporter,
) -> None:
    inner = _SyncMcp()
    client = MockableClient(inner, server="demo")
    with patch(
        "debrix.mcp.resolve_runtime_control",
        return_value=returned(value={"rows": [1]}),
    ):
        assert client.call_tool("query", {"sql": "select 1"}) == {"rows": [1]}
    assert inner.calls == 0


def test_async_tool_and_mcp_control_do_not_invoke_real_operations(
    memory_exporter: InMemorySpanExporter,
) -> None:
    tool_calls = 0
    inner = _AsyncMcp()
    client = MockableClient(inner, server="demo")

    @trace_tool(name="lookup")
    async def lookup() -> str:
        nonlocal tool_calls
        tool_calls += 1
        return "live"

    async def run() -> None:
        with (
            patch(
                "debrix.tracing.resolve_runtime_control_async",
                return_value=returned(value="controlled"),
            ),
            patch(
                "debrix.mcp.resolve_runtime_control_async",
                return_value=returned(value={"rows": []}),
            ),
        ):
            assert await lookup() == "controlled"
            assert await client.call_tool("query", {"sql": "select 1"}) == {
                "rows": []
            }

    asyncio.run(run())
    assert tool_calls == 0
    assert inner.calls == 0
