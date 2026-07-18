"""Resolve Tool Mocker rules from the Debrix desktop app.

Calls ``POST {otlp_base}/mocks/resolve``. On any failure or timeout, returns
passthrough so the agent never blocks on the debugger.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("debrix.mocks")

# Keep resolve snappy — passthrough if Debrix is slow/down.
_RESOLVE_TIMEOUT_S = 0.2


def _json_safe(value: Any) -> Any:
    """Convert a value into something ``json.dumps`` can encode."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return repr(value)


@dataclass(frozen=True)
class MockError:
    kind: str
    message: str | None = None


@dataclass(frozen=True)
class MockDecision:
    """Outcome of a mock resolve call."""

    action: str  # "passthrough" | "mock"
    delay_ms: int | None = None
    result: Any = None
    error: MockError | None = None
    rule_id: str | None = None


PASSTHROUGH = MockDecision(action="passthrough")


class MockToolError(RuntimeError):
    """Raised when a mock rule returns an error / timeout."""

    def __init__(self, kind: str, message: str | None = None) -> None:
        self.kind = kind
        self.message = message or f"mocked tool {kind}"
        super().__init__(self.message)


def resolve_mock(
    *,
    kind: str,
    name: str,
    arguments: dict[str, Any] | None = None,
    server: str | None = None,
    endpoint: str | None = None,
    timeout: float = _RESOLVE_TIMEOUT_S,
) -> MockDecision:
    """Ask Debrix whether to mock this tool call.

    Returns ``PASSTHROUGH`` if Debrix is unreachable, times out, or has no rule.
    """
    from debrix.config import get_otlp_endpoint

    base = (endpoint or get_otlp_endpoint()).rstrip("/")
    url = f"{base}/mocks/resolve"
    body: dict[str, Any] = {
        "kind": kind,
        "name": name,
        "arguments": _json_safe(arguments or {}),
    }
    if server:
        body["server"] = server
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.debug("mock resolve passthrough (%s): %s", name, exc)
        return PASSTHROUGH
    except Exception as exc:  # noqa: BLE001
        logger.debug("mock resolve passthrough (%s): %s", name, exc)
        return PASSTHROUGH

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return PASSTHROUGH

    if not isinstance(payload, dict):
        return PASSTHROUGH
    action = payload.get("action")
    if action != "mock":
        return PASSTHROUGH

    delay_raw = payload.get("delay_ms")
    delay_ms = int(delay_raw) if isinstance(delay_raw, (int, float)) else None
    err_raw = payload.get("error")
    error: MockError | None = None
    if isinstance(err_raw, dict):
        error = MockError(
            kind=str(err_raw.get("kind") or "error"),
            message=err_raw.get("message") if isinstance(err_raw.get("message"), str) else None,
        )
    return MockDecision(
        action="mock",
        delay_ms=delay_ms,
        result=payload.get("result"),
        error=error,
        rule_id=payload.get("rule_id") if isinstance(payload.get("rule_id"), str) else None,
    )


def apply_mock_decision(decision: MockDecision) -> Any:
    """Apply delay / error / fixed result. Raises ``MockToolError`` on error modes.

    Callers must only invoke this when ``decision.action == "mock"``.
    """
    if decision.delay_ms and decision.delay_ms > 0:
        time.sleep(decision.delay_ms / 1000.0)
    if decision.error is not None:
        raise MockToolError(decision.error.kind, decision.error.message)
    return decision.result


async def apply_mock_decision_async(decision: MockDecision) -> Any:
    """Async variant of :func:`apply_mock_decision` (uses ``asyncio.sleep``)."""
    import asyncio

    if decision.delay_ms and decision.delay_ms > 0:
        await asyncio.sleep(decision.delay_ms / 1000.0)
    if decision.error is not None:
        raise MockToolError(decision.error.kind, decision.error.message)
    return decision.result
