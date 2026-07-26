"""Managed FW v3 verification channel.

The launch capability is supplied by Debrix through environment variables.
The first Debrix span binds exactly one trace. Once bound, supported
Tool/MCP/LLM boundaries re-check the local service and bypass every diagnostic
resolver before invoking the real operation.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import urllib.error
import urllib.request
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Literal, Mapping, cast

from debrix.config import get_otlp_endpoint
from debrix.control import SDK_VERSION
from debrix.semconv import Attr
from debrix.span import DebrixSpan

PROTOCOL_VERSION = 1
ENV_ATTEMPT_ID = "DEBRIX_VERIFICATION_ATTEMPT_ID"
ENV_RUN_TOKEN = "DEBRIX_VERIFICATION_TOKEN"
_HTTP_TIMEOUT_SECONDS = 2.0

ExecutionModel = Literal["sync", "asyncio"]
VerificationPurpose = Literal["fix_verification", "regression_rerun"]


class DebrixVerificationError(RuntimeError):
    """Base class for managed verification failures."""


class DebrixVerificationConfigurationError(DebrixVerificationError):
    """Raised when verification launch environment is incomplete or invalid."""


class DebrixVerificationProtocolError(DebrixVerificationError):
    """Raised when the local service returns an incompatible payload."""


class DebrixVerificationRejected(DebrixVerificationError):
    """Raised when Debrix rejects the initial trace binding."""


class DebrixVerificationControlLost(DebrixVerificationError):
    """Raised when an already-managed run can no longer fail closed safely."""


@dataclass(frozen=True)
class VerificationLaunchConfig:
    attempt_id: str
    run_token: str


@dataclass(frozen=True)
class ManagedVerification:
    attempt_id: str
    purpose: VerificationPurpose
    trace_id: str
    root_span_id: str
    protocol_version: int


_MANAGED_VERIFICATION: ContextVar[ManagedVerification | None] = ContextVar(
    "debrix.managed_verification",
    default=None,
)


def _valid_prefixed_id(value: str, prefix: str) -> bool:
    suffix = value.removeprefix(f"{prefix}_")
    alphabet = frozenset("0123456789abcdefghjkmnpqrstvwxyz")
    return (
        value.startswith(f"{prefix}_")
        and len(suffix) == 26
        and all(char in alphabet for char in suffix)
    )


def launch_config_from_env(
    environ: Mapping[str, str] | None = None,
) -> VerificationLaunchConfig | None:
    """Read the all-or-nothing managed verification launch environment."""

    values = os.environ if environ is None else environ
    attempt_id = values.get(ENV_ATTEMPT_ID)
    run_token = values.get(ENV_RUN_TOKEN)
    if attempt_id is None and run_token is None:
        return None
    if not attempt_id or not run_token:
        raise DebrixVerificationConfigurationError(
            f"{ENV_ATTEMPT_ID} and {ENV_RUN_TOKEN} must be set together"
        )
    if not _valid_prefixed_id(attempt_id, "verification"):
        raise DebrixVerificationConfigurationError(
            f"{ENV_ATTEMPT_ID} is not a valid verification attempt ID"
        )
    if len(run_token) != 43 or any(
        not (char.isascii() and (char.isalnum() or char in "-_"))
        for char in run_token
    ):
        raise DebrixVerificationConfigurationError(
            f"{ENV_RUN_TOKEN} is not a valid launch capability"
        )
    return VerificationLaunchConfig(
        attempt_id=attempt_id,
        run_token=run_token,
    )


def current_verification() -> ManagedVerification | None:
    """Return the current managed verification context, if any."""

    return _MANAGED_VERIFICATION.get()


def _post_json(path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    body = json.dumps(
        dict(payload),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{get_otlp_endpoint()}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=_HTTP_TIMEOUT_SECONDS,
        ) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            decoded = {}
        code = decoded.get("code")
        message = decoded.get("message")
        detail = message if isinstance(message, str) else f"HTTP {error.code}"
        if code == "SDK_INCOMPATIBLE":
            raise DebrixVerificationProtocolError(detail) from error
        if code == "VERIFICATION_CONTROL_LOST":
            raise DebrixVerificationControlLost(detail) from error
        raise DebrixVerificationRejected(detail) from error
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise DebrixVerificationControlLost(
            "Debrix verification service is unavailable"
        ) from error
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DebrixVerificationProtocolError(
            "Debrix returned invalid verification JSON"
        ) from error
    if not isinstance(decoded, dict):
        raise DebrixVerificationProtocolError(
            "Debrix returned an invalid verification response"
        )
    return cast(dict[str, Any], decoded)


def _validate_receipt(
    response: Mapping[str, Any],
    *,
    attempt_id: str,
    trace_id: str,
    root_span_id: str,
) -> ManagedVerification:
    if response.get("protocol_version") != PROTOCOL_VERSION:
        raise DebrixVerificationProtocolError(
            "Debrix returned an incompatible verification protocol"
        )
    if response.get("mode") != "managed":
        raise DebrixVerificationProtocolError(
            "Debrix did not confirm managed verification"
        )
    if response.get("attempt_id") != attempt_id:
        raise DebrixVerificationProtocolError(
            "Debrix returned a different verification attempt"
        )
    if response.get("trace_id") != trace_id:
        raise DebrixVerificationProtocolError(
            "Debrix returned a different verification trace"
        )
    if response.get("root_span_id") != root_span_id:
        raise DebrixVerificationProtocolError(
            "Debrix returned a different verification root span"
        )
    if response.get("boundary_checks_required") is not True:
        raise DebrixVerificationProtocolError(
            "Debrix did not require verification boundary checks"
        )
    purpose = response.get("purpose")
    if purpose not in ("fix_verification", "regression_rerun"):
        raise DebrixVerificationProtocolError(
            "Debrix returned an unsupported verification purpose"
        )
    return ManagedVerification(
        attempt_id=attempt_id,
        purpose=cast(VerificationPurpose, purpose),
        trace_id=trace_id,
        root_span_id=root_span_id,
        protocol_version=PROTOCOL_VERSION,
    )


def _bind(
    span: DebrixSpan,
    config: VerificationLaunchConfig,
    execution_model: ExecutionModel,
) -> ManagedVerification:
    request_id = secrets.token_urlsafe(24)
    response = _post_json(
        "/v1/verification/bind",
        {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "attempt_id": config.attempt_id,
            "run_token": config.run_token,
            "trace_id": span.trace_id_hex,
            "runtime_span_id": span.span_id_hex,
            "sdk": {
                "name": "debrix-python",
                "version": SDK_VERSION,
                "execution_model": execution_model,
            },
        },
    )
    return _validate_receipt(
        response,
        attempt_id=config.attempt_id,
        trace_id=span.trace_id_hex,
        root_span_id=span.span_id_hex,
    )


def _mark_span(span: DebrixSpan, managed: ManagedVerification) -> None:
    if span.trace_id_hex != managed.trace_id:
        raise DebrixVerificationControlLost(
            "managed verification escaped its bound trace"
        )
    span.set_attribute(Attr.VERIFICATION_ATTEMPT_ID, managed.attempt_id)
    span.set_attribute(Attr.VERIFICATION_PURPOSE, managed.purpose)
    span.set_attribute(
        Attr.VERIFICATION_PROTOCOL_VERSION,
        managed.protocol_version,
    )
    span.set_attribute(Attr.VERIFICATION_ROOT_SPAN_ID, managed.root_span_id)
    span.set_attribute(Attr.VERIFICATION_NO_OVERRIDES, True)


def prepare_span_sync(
    span: DebrixSpan,
) -> Token[ManagedVerification | None] | None:
    """Bind a sync root or mark a nested managed span."""

    managed = current_verification()
    if managed is not None:
        _mark_span(span, managed)
        return None
    config = launch_config_from_env()
    if config is None:
        return None
    managed = _bind(span, config, "sync")
    token = _MANAGED_VERIFICATION.set(managed)
    _mark_span(span, managed)
    return token


async def prepare_span_async(
    span: DebrixSpan,
) -> Token[ManagedVerification | None] | None:
    """Bind an asyncio root without blocking the event loop."""

    managed = current_verification()
    if managed is not None:
        _mark_span(span, managed)
        return None
    config = launch_config_from_env()
    if config is None:
        return None
    managed = await asyncio.to_thread(_bind, span, config, "asyncio")
    token = _MANAGED_VERIFICATION.set(managed)
    _mark_span(span, managed)
    return token


def reset_span_context(token: Token[ManagedVerification | None] | None) -> None:
    if token is not None:
        _MANAGED_VERIFICATION.reset(token)


def _check(span: DebrixSpan, execution_model: ExecutionModel) -> bool:
    managed = current_verification()
    if managed is None:
        return False
    _mark_span(span, managed)
    response = _post_json(
        "/v1/verification/check",
        {
            "protocol_version": PROTOCOL_VERSION,
            "attempt_id": managed.attempt_id,
            "trace_id": managed.trace_id,
            "runtime_span_id": span.span_id_hex,
            "execution_model": execution_model,
        },
    )
    if (
        response.get("protocol_version") != PROTOCOL_VERSION
        or response.get("mode") != "managed"
        or response.get("attempt_id") != managed.attempt_id
        or response.get("purpose") != managed.purpose
        or response.get("status") != "bound"
        or response.get("trace_id") != managed.trace_id
    ):
        raise DebrixVerificationProtocolError(
            "Debrix returned an invalid verification control receipt"
        )
    return True


def check_boundary_sync(span: DebrixSpan) -> bool:
    """Return True when this boundary must invoke the real operation directly."""

    return _check(span, "sync")


async def check_boundary_async(span: DebrixSpan) -> bool:
    """Async managed boundary check without blocking the event loop."""

    if current_verification() is None:
        return False
    return await asyncio.to_thread(_check, span, "asyncio")


def reset_for_tests() -> None:
    """Clear the current context in isolated SDK tests."""

    _MANAGED_VERIFICATION.set(None)
