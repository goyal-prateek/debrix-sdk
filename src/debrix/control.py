"""Versioned FW v2 SDK control-channel types and parsing.

The control channel is deliberately separate from the fail-open Tool Mocker
resolver. A probe is unmanaged until Debrix returns a claim. Once a claim has
been observed, callers must use the managed resolver and fail closed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from debrix.config import get_otlp_endpoint

PROTOCOL_VERSION = 1
SDK_VERSION = "0.1.0b1"
logger = logging.getLogger("debrix.control")

ControlProvenance = Literal["recorded", "edited", "live"]
ControlValueKind = Literal["result", "error"]


class DebrixControlError(RuntimeError):
    """Base class for controlled-branch SDK failures."""


class DebrixControlProtocolError(DebrixControlError):
    """Raised when Debrix returns an invalid or incompatible control payload."""


class DebrixBreakpointCancelled(DebrixControlError):
    """Raised when a controller explicitly cancels a waiting branch."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class DebrixControlLost(DebrixControlError):
    """Raised when managed control cannot be recovered before terminal expiry."""


@dataclass(frozen=True)
class ControlOperation:
    kind: str
    name: str
    server: str | None = None
    agent_scope: str | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "server": self.server,
            "agent_scope": self.agent_scope,
        }


@dataclass(frozen=True)
class ControlRequest:
    request_id: str
    trace_id: str
    runtime_span_id: str
    parent_runtime_span_id: str | None
    sequence_index: int
    operation: ControlOperation
    input_value: Any
    capabilities: tuple[str, ...]
    execution_model: str
    sdk_version: str
    input_descriptor: Mapping[str, Any] | None = None
    attempt_id: str | None = None

    def to_wire(self) -> dict[str, Any]:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": self.request_id,
            "attempt_id": self.attempt_id,
            "trace_id": self.trace_id,
            "runtime_span_id": self.runtime_span_id,
            "parent_runtime_span_id": self.parent_runtime_span_id,
            "sequence_index": self.sequence_index,
            "operation": self.operation.to_wire(),
            "input": self.input_value,
            "capabilities": list(self.capabilities),
            "sdk": {
                "name": "debrix-python",
                "version": self.sdk_version,
                "execution_model": self.execution_model,
            },
        }
        if self.input_descriptor is not None:
            payload["input_descriptor"] = dict(self.input_descriptor)
        return payload

    def canonical_json(self) -> str:
        return json.dumps(self.to_wire(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ControlUnmanaged:
    pass


@dataclass(frozen=True)
class ControlClaim:
    claim_id: str
    claim_token: str
    attempt_id: str
    expires_at: str


@dataclass(frozen=True)
class ControlWaiting:
    attempt_id: str
    branch_id: str
    occurrence_id: str
    retry_after_ms: int


@dataclass(frozen=True)
class ControlErrorValue:
    kind: str
    message: str | None = None


@dataclass(frozen=True)
class ControlResolvedValue:
    provenance: ControlProvenance
    kind: ControlValueKind
    value: Any = None
    error: ControlErrorValue | None = None


@dataclass(frozen=True)
class ControlResolvedInput:
    provenance: ControlProvenance
    value: Any


@dataclass(frozen=True)
class ControlReturn:
    attempt_id: str
    branch_id: str
    occurrence_id: str
    decision_id: str
    live_suffix: bool
    output: ControlResolvedValue


@dataclass(frozen=True)
class ControlInvoke:
    attempt_id: str
    branch_id: str
    occurrence_id: str
    decision_id: str
    input: ControlResolvedInput
    capture_live_result: bool


@dataclass(frozen=True)
class ControlCancel:
    attempt_id: str
    branch_id: str
    occurrence_id: str
    reason: str


ControlProbeResponse = ControlUnmanaged | ControlClaim
ControlResolveResponse = (
    ControlWaiting | ControlReturn | ControlInvoke | ControlCancel
)
ControlDecision = ControlReturn | ControlInvoke


def _mapping(payload: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise DebrixControlProtocolError(f"{context} must be a JSON object")
    return payload


def _protocol(payload: Mapping[str, Any]) -> None:
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise DebrixControlProtocolError(
            f"protocol_version must be {PROTOCOL_VERSION}"
        )


def _string(
    payload: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DebrixControlProtocolError(
            f"{context}.{key} must be a non-empty string"
        )
    return value


def _integer(
    payload: Mapping[str, Any],
    key: str,
    *,
    context: str,
    minimum: int = 0,
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DebrixControlProtocolError(
            f"{context}.{key} must be an integer >= {minimum}"
        )
    return value


def parse_probe_response(payload: Any) -> ControlProbeResponse:
    body = _mapping(payload, "probe response")
    _protocol(body)
    action = body.get("action")
    if action == "unmanaged":
        return ControlUnmanaged()
    if action != "claim":
        raise DebrixControlProtocolError(
            "probe action must be unmanaged or claim"
        )
    return ControlClaim(
        claim_id=_string(body, "claim_id", context="claim"),
        claim_token=_string(body, "claim_token", context="claim"),
        attempt_id=_string(body, "attempt_id", context="claim"),
        expires_at=_string(body, "expires_at", context="claim"),
    )


def _resolved_value(
    payload: Any,
    *,
    context: str,
) -> ControlResolvedValue:
    body = _mapping(payload, context)
    provenance = body.get("provenance")
    if provenance not in {"recorded", "edited", "live"}:
        raise DebrixControlProtocolError(
            f"{context}.provenance must be recorded, edited, or live"
        )
    kind = body.get("kind")
    if kind not in {"result", "error"}:
        raise DebrixControlProtocolError(
            f"{context}.kind must be result or error"
        )
    if kind == "result":
        if "value" not in body:
            raise DebrixControlProtocolError(
                f"{context}.value is required for a result"
            )
        return ControlResolvedValue(
            provenance=provenance,
            kind=kind,
            value=body.get("value"),
        )

    error_body = _mapping(body.get("error"), f"{context}.error")
    error = ControlErrorValue(
        kind=_string(error_body, "kind", context=f"{context}.error"),
        message=(
            error_body.get("message")
            if isinstance(error_body.get("message"), str)
            else None
        ),
    )
    return ControlResolvedValue(
        provenance=provenance,
        kind=kind,
        error=error,
    )


def _resolved_input(payload: Any) -> ControlResolvedInput:
    body = _mapping(payload, "invoke.input")
    provenance = body.get("provenance")
    if provenance not in {"recorded", "edited", "live"}:
        raise DebrixControlProtocolError(
            "invoke.input.provenance must be recorded, edited, or live"
        )
    if "value" not in body:
        raise DebrixControlProtocolError("invoke.input.value is required")
    return ControlResolvedInput(
        provenance=provenance,
        value=body.get("value"),
    )


def parse_resolve_response(payload: Any) -> ControlResolveResponse:
    body = _mapping(payload, "resolve response")
    _protocol(body)
    action = body.get("action")
    if action == "waiting":
        return ControlWaiting(
            attempt_id=_string(body, "attempt_id", context="waiting"),
            branch_id=_string(body, "branch_id", context="waiting"),
            occurrence_id=_string(body, "occurrence_id", context="waiting"),
            retry_after_ms=_integer(
                body,
                "retry_after_ms",
                context="waiting",
            ),
        )
    if action == "cancel":
        return ControlCancel(
            attempt_id=_string(body, "attempt_id", context="cancel"),
            branch_id=_string(body, "branch_id", context="cancel"),
            occurrence_id=_string(body, "occurrence_id", context="cancel"),
            reason=_string(body, "reason", context="cancel"),
        )
    if action not in {"return", "invoke"}:
        raise DebrixControlProtocolError(
            "managed resolve action must be waiting, return, invoke, or cancel"
        )

    attempt_id = _string(body, "attempt_id", context=str(action))
    branch_id = _string(body, "branch_id", context=str(action))
    occurrence_id = _string(body, "occurrence_id", context=str(action))
    decision_id = _string(body, "decision_id", context=str(action))
    if action == "return":
        live_suffix = body.get("live_suffix")
        if not isinstance(live_suffix, bool):
            raise DebrixControlProtocolError(
                "return.live_suffix must be a boolean"
            )
        return ControlReturn(
            attempt_id=attempt_id,
            branch_id=branch_id,
            occurrence_id=occurrence_id,
            decision_id=decision_id,
            live_suffix=live_suffix,
            output=_resolved_value(body.get("output"), context="return.output"),
        )

    capture_live_result = body.get("capture_live_result")
    if not isinstance(capture_live_result, bool):
        raise DebrixControlProtocolError(
            "invoke.capture_live_result must be a boolean"
        )
    return ControlInvoke(
        attempt_id=attempt_id,
        branch_id=branch_id,
        occurrence_id=occurrence_id,
        decision_id=decision_id,
        input=_resolved_input(body.get("input")),
        capture_live_result=capture_live_result,
    )


class _ControlTransportError(Exception):
    """An HTTP exchange failed before a valid control payload was received."""


class _ControlHttpError(_ControlTransportError):
    def __init__(self, status: int, payload: Any = None) -> None:
        self.status = status
        self.payload = payload
        super().__init__(f"control endpoint returned HTTP {status}")


def _decode_response(response: Any) -> Mapping[str, Any]:
    try:
        payload = json.loads(response.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DebrixControlProtocolError(
            "control response must contain valid JSON"
        ) from error
    return _mapping(payload, "control response")


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    timeout: float,
) -> Mapping[str, Any]:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _decode_response(response)
    except urllib.error.HTTPError as error:
        error_payload: Any = None
        try:
            error_payload = json.loads(error.read().decode("utf-8"))
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            pass
        raise _ControlHttpError(error.code, error_payload) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise _ControlTransportError(str(error)) from error


def _managed_payload(
    request: ControlRequest,
    claim: ControlClaim,
) -> dict[str, Any]:
    payload = request.to_wire()
    payload.update(
        {
            "attempt_id": claim.attempt_id,
            "claim_id": claim.claim_id,
            "claim_token": claim.claim_token,
        }
    )
    return payload


def _structured_error(error: _ControlHttpError) -> str | None:
    if not isinstance(error.payload, Mapping):
        return None
    body = error.payload
    nested = body.get("error")
    if isinstance(nested, Mapping):
        code = nested.get("code")
        message = nested.get("message")
    else:
        code = body.get("code")
        message = body.get("message")
    if isinstance(code, str) and isinstance(message, str):
        return f"{code}: {message}"
    return None


def _abandon_control(
    endpoint: str,
    request: ControlRequest,
    claim: ControlClaim,
    *,
    timeout: float,
    reason: str,
) -> None:
    payload = _managed_payload(request, claim)
    payload["reason"] = reason
    try:
        _post_json(
            f"{endpoint}/v1/control/abandon",
            payload,
            timeout=timeout,
        )
    except (DebrixControlProtocolError, _ControlTransportError):
        logger.debug(
            "Best-effort control abandon could not be acknowledged",
            exc_info=True,
        )


def resolve_control(
    request: ControlRequest,
    *,
    endpoint: str | None = None,
    probe_timeout: float = 0.2,
    poll_timeout: float = 5.0,
    total_timeout: float | None = None,
    retry_initial: float = 0.05,
    retry_max: float = 1.0,
) -> ControlDecision | ControlUnmanaged:
    """Resolve one FW v2 control request.

    A missing or unreachable probe is safely unmanaged for compatibility with
    Debrix versions that predate FW v2. Observing a claim crosses the
    fail-closed boundary: transport loss is retried with the same request and
    claim identity until a terminal decision, cancellation, or configured
    expiry.
    """

    for name, value in (
        ("probe_timeout", probe_timeout),
        ("poll_timeout", poll_timeout),
        ("retry_initial", retry_initial),
        ("retry_max", retry_max),
    ):
        if value < 0:
            raise ValueError(f"{name} must be >= 0")
    if total_timeout is not None and total_timeout < 0:
        raise ValueError("total_timeout must be >= 0 or None")

    base_endpoint = (endpoint or get_otlp_endpoint()).rstrip("/")
    try:
        probe_payload = _post_json(
            f"{base_endpoint}/v1/control/probe",
            request.to_wire(),
            timeout=probe_timeout,
        )
    except _ControlHttpError as error:
        structured = _structured_error(error)
        if structured is not None or (
            400 <= error.status < 500 and error.status != 404
        ):
            raise DebrixControlProtocolError(
                structured or str(error)
            ) from error
        logger.debug(
            "Control probe endpoint is unavailable; continuing unmanaged",
            exc_info=True,
        )
        return ControlUnmanaged()
    except _ControlTransportError:
        logger.debug(
            "Control probe unavailable; continuing unmanaged",
            exc_info=True,
        )
        return ControlUnmanaged()

    probe = parse_probe_response(probe_payload)
    if isinstance(probe, ControlUnmanaged):
        return probe

    managed_payload = _managed_payload(request, probe)
    deadline = (
        None
        if total_timeout is None
        else time.monotonic() + total_timeout
    )
    retry_delay = retry_initial

    while True:
        if deadline is not None and time.monotonic() >= deadline:
            _abandon_control(
                base_endpoint,
                request,
                probe,
                timeout=probe_timeout,
                reason="terminal_expiry",
            )
            raise DebrixControlLost(
                "managed control reached terminal expiry before a decision"
            )

        try:
            response_payload = _post_json(
                f"{base_endpoint}/v1/control/resolve",
                managed_payload,
                timeout=poll_timeout,
            )
        except _ControlHttpError as error:
            structured = _structured_error(error)
            if structured is not None or 400 <= error.status < 500:
                raise DebrixControlProtocolError(
                    structured or str(error)
                ) from error
        except _ControlTransportError:
            pass
        else:
            response = parse_resolve_response(response_payload)
            if isinstance(response, ControlWaiting):
                wait_seconds = response.retry_after_ms / 1000
                if wait_seconds:
                    time.sleep(wait_seconds)
                retry_delay = retry_initial
                continue
            if isinstance(response, ControlCancel):
                raise DebrixBreakpointCancelled(response.reason)
            return response

        if retry_delay:
            time.sleep(retry_delay)
        retry_delay = min(
            retry_max,
            retry_delay * 2 if retry_delay else retry_initial,
        )


async def resolve_control_async(
    request: ControlRequest,
    **kwargs: Any,
) -> ControlDecision | ControlUnmanaged:
    """Resolve control without blocking an asyncio event loop."""

    return await asyncio.to_thread(resolve_control, request, **kwargs)
