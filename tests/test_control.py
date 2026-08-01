"""Contract tests for the FW v2 SDK control channel."""

from __future__ import annotations

import asyncio
import io
import json
import threading
import urllib.error
from typing import Any

import debrix
import pytest

from debrix.control import (
    ControlDecision,
    ControlClaim,
    ControlInvoke,
    ControlOperation,
    ControlRequest,
    ControlReturn,
    ControlUnmanaged,
    DebrixBreakpointCancelled,
    DebrixControlLost,
    DebrixControlProtocolError,
    parse_probe_response,
    parse_resolve_response,
    resolve_control,
    resolve_control_async,
)


def request() -> ControlRequest:
    return ControlRequest(
        request_id="control_request_123",
        trace_id="0123456789abcdef0123456789abcdef",
        runtime_span_id="0123456789abcdef",
        parent_runtime_span_id=None,
        sequence_index=0,
        operation=ControlOperation(
            kind="tool",
            name="lookup",
            server=None,
            agent_scope="planner",
        ),
        input_value={"query": "current"},
        capabilities=("input", "result", "error"),
        execution_model="sync",
        sdk_version="0.1.0b1",
        input_descriptor={
            "schemaVersion": 1,
            "operationKind": "tool",
            "jsonKind": "object",
            "parameters": [],
        },
    )


def test_control_failures_are_part_of_the_public_sdk_api() -> None:
    assert debrix.DebrixBreakpointCancelled is DebrixBreakpointCancelled
    assert debrix.DebrixControlLost is DebrixControlLost


def test_request_wire_shape_is_stable_and_canonical() -> None:
    control_request = request()

    assert control_request.to_wire() == {
        "protocol_version": 1,
        "request_id": "control_request_123",
        "attempt_id": None,
        "trace_id": "0123456789abcdef0123456789abcdef",
        "runtime_span_id": "0123456789abcdef",
        "parent_runtime_span_id": None,
        "sequence_index": 0,
        "operation": {
            "kind": "tool",
            "name": "lookup",
            "server": None,
            "agent_scope": "planner",
        },
        "input": {"query": "current"},
        "input_descriptor": {
            "schemaVersion": 1,
            "operationKind": "tool",
            "jsonKind": "object",
            "parameters": [],
        },
        "capabilities": ["input", "result", "error"],
        "sdk": {
            "name": "debrix-python",
            "version": "0.1.0b1",
            "execution_model": "sync",
        },
    }
    assert json.dumps(
        control_request.to_wire(),
        sort_keys=True,
        separators=(",", ":"),
    ) == control_request.canonical_json()


def test_probe_response_is_a_closed_union() -> None:
    unmanaged = parse_probe_response(
        {"protocol_version": 1, "action": "unmanaged"}
    )
    claim = parse_probe_response(
        {
            "protocol_version": 1,
            "action": "claim",
            "claim_id": "claim_123",
            "claim_token": "opaque-token",
            "attempt_id": "branch_attempt_123",
            "expires_at": "2026-07-26T00:00:10.000Z",
        }
    )

    assert isinstance(unmanaged, ControlUnmanaged)
    assert claim == ControlClaim(
        claim_id="claim_123",
        claim_token="opaque-token",
        attempt_id="branch_attempt_123",
        expires_at="2026-07-26T00:00:10.000Z",
    )

    with pytest.raises(DebrixControlProtocolError, match="probe action"):
        parse_probe_response({"protocol_version": 1, "action": "surprise"})


def test_claim_rejects_wrong_protocol_or_missing_capability() -> None:
    with pytest.raises(DebrixControlProtocolError, match="protocol_version"):
        parse_probe_response(
            {
                "protocol_version": 2,
                "action": "claim",
                "claim_id": "claim_123",
                "claim_token": "opaque-token",
                "attempt_id": "branch_attempt_123",
                "expires_at": "2026-07-26T00:00:10.000Z",
            }
        )

    with pytest.raises(DebrixControlProtocolError, match="claim_token"):
        parse_probe_response(
            {
                "protocol_version": 1,
                "action": "claim",
                "claim_id": "claim_123",
                "attempt_id": "branch_attempt_123",
                "expires_at": "2026-07-26T00:00:10.000Z",
            }
        )


def test_terminal_return_and_invoke_preserve_provenance() -> None:
    returned = parse_resolve_response(
        {
            "protocol_version": 1,
            "action": "return",
            "attempt_id": "branch_attempt_123",
            "branch_id": "branch_123",
            "occurrence_id": "occurrence_123",
            "decision_id": "decision_123",
            "live_suffix": True,
            "output": {
                "provenance": "edited",
                "kind": "result",
                "value": {"answer": 42},
            },
        }
    )
    invoked = parse_resolve_response(
        {
            "protocol_version": 1,
            "action": "invoke",
            "attempt_id": "branch_attempt_123",
            "branch_id": "branch_123",
            "occurrence_id": "occurrence_123",
            "decision_id": "decision_124",
            "input": {
                "provenance": "edited",
                "value": {"query": "corrected"},
            },
            "capture_live_result": True,
        }
    )

    assert isinstance(returned, ControlReturn)
    assert returned.live_suffix is True
    assert returned.output.provenance == "edited"
    assert returned.output.value == {"answer": 42}
    assert isinstance(invoked, ControlInvoke)
    assert invoked.input.provenance == "edited"
    assert invoked.input.value == {"query": "corrected"}
    assert invoked.capture_live_result is True


@pytest.mark.parametrize(
    "payload",
    [
        {"protocol_version": 1, "action": "unmanaged"},
        {"protocol_version": 1, "action": "return"},
        {
            "protocol_version": 1,
            "action": "invoke",
            "attempt_id": "branch_attempt_123",
            "branch_id": "branch_123",
            "occurrence_id": "occurrence_123",
            "decision_id": "decision_123",
            "input": {"provenance": "mystery", "value": {}},
        },
    ],
)
def test_managed_response_rejects_malformed_or_unmanaged_payloads(
    payload: dict[str, Any],
) -> None:
    with pytest.raises(DebrixControlProtocolError):
        parse_resolve_response(payload)


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def scripted_urlopen(
    monkeypatch: pytest.MonkeyPatch,
    script: list[dict[str, Any] | BaseException],
) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    remaining = iter(script)

    def open_request(
        http_request: Any,
        **_: Any,
    ) -> _Response:
        calls.append(
            (
                http_request.full_url,
                json.loads(http_request.data.decode()),
            )
        )
        outcome = next(remaining)
        if isinstance(outcome, BaseException):
            raise outcome
        return _Response(outcome)

    monkeypatch.setattr(
        "debrix.control.urllib.request.urlopen",
        open_request,
    )
    return calls


def claim() -> dict[str, Any]:
    return {
        "protocol_version": 1,
        "action": "claim",
        "claim_id": "claim_123",
        "claim_token": "opaque-token",
        "attempt_id": "branch_attempt_123",
        "expires_at": "2026-07-26T00:00:10.000Z",
    }


def returned() -> dict[str, Any]:
    return {
        "protocol_version": 1,
        "action": "return",
        "attempt_id": "branch_attempt_123",
        "branch_id": "branch_123",
        "occurrence_id": "occurrence_123",
        "decision_id": "decision_123",
        "live_suffix": True,
        "output": {
            "provenance": "edited",
            "kind": "result",
            "value": {"answer": 42},
        },
    }


def test_unreachable_probe_is_safely_unmanaged() -> None:
    result = resolve_control(
        request(),
        endpoint="http://127.0.0.1:1",
        probe_timeout=0.01,
    )

    assert isinstance(result, ControlUnmanaged)


def test_structured_probe_rejection_is_not_treated_as_unmanaged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = json.dumps(
        {
            "error": {
                "code": "sdk_incompatible",
                "message": "descriptor mismatch",
            }
        }
    ).encode()
    scripted_urlopen(
        monkeypatch,
        [
            urllib.error.HTTPError(
                "http://127.0.0.1:17418/v1/control/probe",
                409,
                "Conflict",
                {},
                io.BytesIO(body),
            )
        ],
    )

    with pytest.raises(
        DebrixControlProtocolError,
        match="sdk_incompatible: descriptor mismatch",
    ):
        resolve_control(request(), endpoint="http://127.0.0.1:17418")


def test_claimed_resolution_retries_same_request_after_wait_and_transport_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = scripted_urlopen(
        monkeypatch,
        [
            claim(),
            {
                "protocol_version": 1,
                "action": "waiting",
                "attempt_id": "branch_attempt_123",
                "branch_id": "branch_123",
                "occurrence_id": "occurrence_123",
                "retry_after_ms": 0,
            },
            urllib.error.URLError("desktop restarting"),
            returned(),
        ],
    )

    result: ControlDecision | ControlUnmanaged = resolve_control(
        request(),
        endpoint="http://127.0.0.1:17418",
        poll_timeout=0.01,
        retry_initial=0,
        retry_max=0,
        total_timeout=1,
    )

    assert isinstance(result, ControlReturn)
    assert result.output.value == {"answer": 42}
    assert [path.rsplit("/", 1)[-1] for path, _ in calls] == [
        "probe",
        "resolve",
        "resolve",
        "resolve",
    ]
    assert {body["request_id"] for _, body in calls} == {
        "control_request_123"
    }
    for _, body in calls[1:]:
        assert body["claim_id"] == "claim_123"
        assert body["claim_token"] == "opaque-token"
        assert body["attempt_id"] == "branch_attempt_123"


def test_cancel_raises_typed_exception_after_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripted_urlopen(
        monkeypatch,
        [
            claim(),
            {
                "protocol_version": 1,
                "action": "cancel",
                "attempt_id": "branch_attempt_123",
                "branch_id": "branch_123",
                "occurrence_id": "occurrence_123",
                "reason": "cancelled_by_controller",
            },
        ],
    )

    with pytest.raises(
        DebrixBreakpointCancelled,
        match="cancelled_by_controller",
    ):
        resolve_control(
            request(),
            endpoint="http://127.0.0.1:17418",
        )


def test_terminal_expiry_abandons_and_raises_control_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = scripted_urlopen(
        monkeypatch,
        [
            claim(),
            {"protocol_version": 1, "action": "abandoned"},
        ],
    )

    with pytest.raises(DebrixControlLost, match="terminal expiry"):
        resolve_control(
            request(),
            endpoint="http://127.0.0.1:17418",
            total_timeout=0,
        )

    assert [path.rsplit("/", 1)[-1] for path, _ in calls] == [
        "probe",
        "abandon",
    ]
    assert calls[1][1]["attempt_id"] == "branch_attempt_123"


def test_async_control_wait_does_not_block_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_resolve = threading.Event()
    resolve_entered = threading.Event()

    def open_request(http_request: Any, **_: Any) -> _Response:
        if http_request.full_url.endswith("/probe"):
            return _Response(claim())
        resolve_entered.set()
        if not release_resolve.wait(0.5):
            raise AssertionError("control HTTP wait blocked the event loop")
        return _Response(returned())

    monkeypatch.setattr(
        "debrix.control.urllib.request.urlopen",
        open_request,
    )

    async def scenario() -> ControlDecision | ControlUnmanaged:
        task = asyncio.create_task(
            resolve_control_async(
                request(),
                endpoint="http://127.0.0.1:17418",
            )
        )
        while not resolve_entered.is_set():
            await asyncio.sleep(0)
        await asyncio.sleep(0)
        release_resolve.set()
        return await task

    result = asyncio.run(scenario())

    assert isinstance(result, ControlReturn)
