"""Contract tests for the FW v3 managed no-override SDK channel."""

from __future__ import annotations

import asyncio
import json
import threading
import urllib.error
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import debrix
from debrix import Attr, trace_agent, trace_tool
from debrix.llm import complete
from debrix.mcp import MockableClient
from debrix.verification import (
    DebrixVerificationConfigurationError,
    DebrixVerificationControlLost,
    ENV_ATTEMPT_ID,
    ENV_RUN_TOKEN,
    launch_config_from_env,
    reset_for_tests,
)

ATTEMPT_ID = "verification_01arz3ndektsv4rrffq69g5fav"
RUN_TOKEN = "a" * 43


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None


@pytest.fixture(autouse=True)
def managed_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_for_tests()
    monkeypatch.delenv(ENV_ATTEMPT_ID, raising=False)
    monkeypatch.delenv(ENV_RUN_TOKEN, raising=False)
    yield
    reset_for_tests()


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_ATTEMPT_ID, ATTEMPT_ID)
    monkeypatch.setenv(ENV_RUN_TOKEN, RUN_TOKEN)


def _service(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fail_check: bool = False,
    purpose: str = "fix_verification",
) -> tuple[list[tuple[str, dict[str, Any]]], list[int]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    threads: list[int] = []

    def urlopen(request: Any, **_: Any) -> _Response:
        threads.append(threading.get_ident())
        payload = json.loads(request.data.decode())
        calls.append((request.full_url, payload))
        if request.full_url.endswith("/bind"):
            return _Response(
                {
                    "protocol_version": 1,
                    "mode": "managed",
                    "attempt_id": payload["attempt_id"],
                    "purpose": purpose,
                    "trace_id": payload["trace_id"],
                    "root_span_id": payload["runtime_span_id"],
                    "boundary_checks_required": True,
                }
            )
        if fail_check:
            raise urllib.error.URLError("desktop unavailable")
        return _Response(
            {
                "protocol_version": 1,
                "mode": "managed",
                "attempt_id": payload["attempt_id"],
                "purpose": purpose,
                "status": "bound",
                "trace_id": payload["trace_id"],
            }
        )

    monkeypatch.setattr(
        "debrix.verification.urllib.request.urlopen",
        urlopen,
    )
    return calls, threads


def _forbid_tool_resolvers(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError("diagnostic resolver was called")

    monkeypatch.setattr(
        "debrix.tracing.resolve_runtime_control",
        forbidden,
    )
    monkeypatch.setattr(
        "debrix.tracing.resolve_runtime_control_async",
        forbidden,
    )
    monkeypatch.setattr("debrix.tracing.resolve_mock", forbidden)


def test_launch_environment_is_all_or_nothing() -> None:
    assert launch_config_from_env({}) is None
    with pytest.raises(DebrixVerificationConfigurationError, match="together"):
        launch_config_from_env({ENV_ATTEMPT_ID: ATTEMPT_ID})
    with pytest.raises(DebrixVerificationConfigurationError, match="attempt ID"):
        launch_config_from_env(
            {
                ENV_ATTEMPT_ID: "verification_BAD",
                ENV_RUN_TOKEN: RUN_TOKEN,
            }
        )


def test_verification_errors_are_public() -> None:
    assert (
        debrix.DebrixVerificationControlLost
        is DebrixVerificationControlLost
    )
    assert (
        debrix.DebrixVerificationConfigurationError
        is DebrixVerificationConfigurationError
    )


@pytest.mark.parametrize(
    "purpose",
    ["fix_verification", "regression_rerun"],
)
def test_sync_agent_binds_once_and_tool_bypasses_all_resolvers(
    monkeypatch: pytest.MonkeyPatch,
    memory_exporter: InMemorySpanExporter,
    purpose: str,
) -> None:
    _enable(monkeypatch)
    calls, _ = _service(monkeypatch, purpose=purpose)
    _forbid_tool_resolvers(monkeypatch)
    invocations = 0

    @trace_tool
    def lookup(value: str) -> str:
        nonlocal invocations
        invocations += 1
        return value.upper()

    @trace_agent
    def run() -> str:
        return lookup("real")

    assert run() == "REAL"
    assert invocations == 1
    assert [url.rsplit("/", 1)[-1] for url, _ in calls] == [
        "bind",
        "check",
    ]
    bind = calls[0][1]
    assert bind["run_token"] == RUN_TOKEN
    assert bind["sdk"]["execution_model"] == "sync"

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 2
    trace_ids = {format(span.context.trace_id, "032x") for span in spans}
    assert trace_ids == {bind["trace_id"]}
    root_span_id = bind["runtime_span_id"]
    for span in spans:
        assert span.attributes[Attr.VERIFICATION_ATTEMPT_ID] == ATTEMPT_ID
        assert (
            span.attributes[Attr.VERIFICATION_PURPOSE]
            == purpose
        )
        assert span.attributes[Attr.VERIFICATION_PROTOCOL_VERSION] == 1
        assert (
            span.attributes[Attr.VERIFICATION_ROOT_SPAN_ID]
            == root_span_id
        )
        assert span.attributes[Attr.VERIFICATION_NO_OVERRIDES] is True
        assert Attr.STUB not in span.attributes
        assert Attr.CONTROL_ATTEMPT_ID not in span.attributes
        assert RUN_TOKEN not in json.dumps(dict(span.attributes))


@pytest.mark.parametrize(
    "purpose",
    ["fix_verification", "regression_rerun"],
)
def test_async_bind_and_check_do_not_block_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
    memory_exporter: InMemorySpanExporter,
    purpose: str,
) -> None:
    _enable(monkeypatch)
    calls, threads = _service(monkeypatch, purpose=purpose)
    _forbid_tool_resolvers(monkeypatch)
    main_thread = threading.get_ident()

    @trace_tool
    async def lookup() -> str:
        return "real"

    @trace_agent
    async def run() -> str:
        return await lookup()

    async def scenario() -> str:
        return await run()

    assert asyncio.run(scenario()) == "real"
    assert [url.rsplit("/", 1)[-1] for url, _ in calls] == [
        "bind",
        "check",
    ]
    assert all(thread_id != main_thread for thread_id in threads)
    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 2
    assert {
        span.attributes[Attr.VERIFICATION_PURPOSE] for span in spans
    } == {purpose}


def test_control_loss_fails_before_real_tool_execution(
    monkeypatch: pytest.MonkeyPatch,
    memory_exporter: InMemorySpanExporter,
) -> None:
    _enable(monkeypatch)
    _service(monkeypatch, fail_check=True)
    invocations = 0

    @trace_tool
    def lookup() -> str:
        nonlocal invocations
        invocations += 1
        return "unsafe"

    @trace_agent
    def run() -> str:
        return lookup()

    with pytest.raises(
        DebrixVerificationControlLost,
        match="service is unavailable",
    ):
        run()
    assert invocations == 0
    assert memory_exporter.get_finished_spans()


def test_managed_llm_bypasses_control_and_mock(
    monkeypatch: pytest.MonkeyPatch,
    memory_exporter: InMemorySpanExporter,
) -> None:
    _enable(monkeypatch)
    _service(monkeypatch)

    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError("diagnostic resolver was called")

    monkeypatch.setattr("debrix.llm.resolve_runtime_control", forbidden)
    monkeypatch.setattr("debrix.llm.resolve_mock", forbidden)
    assert (
        complete(
            [{"role": "user", "content": "hello"}],
            call=lambda _: ("real", {"input_tokens": 1}, "model"),
        )
        == "real"
    )
    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes[Attr.VERIFICATION_NO_OVERRIDES] is True
    assert Attr.STUB not in span.attributes


def test_managed_mcp_bypasses_control_and_mock(
    monkeypatch: pytest.MonkeyPatch,
    memory_exporter: InMemorySpanExporter,
) -> None:
    _enable(monkeypatch)
    _service(monkeypatch)

    def forbidden(*_: Any, **__: Any) -> Any:
        raise AssertionError("diagnostic resolver was called")

    monkeypatch.setattr("debrix.mcp.resolve_runtime_control", forbidden)
    monkeypatch.setattr("debrix.mcp.resolve_mock", forbidden)

    class Inner:
        def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            return {"name": name, "arguments": arguments}

    result = MockableClient(Inner()).call_tool("lookup", {"value": "real"})
    assert result == {
        "name": "lookup",
        "arguments": {"value": "real"},
    }
    span = memory_exporter.get_finished_spans()[0]
    assert span.attributes[Attr.VERIFICATION_NO_OVERRIDES] is True
    assert Attr.STUB not in span.attributes
