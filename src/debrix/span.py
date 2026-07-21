"""Debrix span wrapper with opt-in message / response recording."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from opentelemetry.trace import Span, Status, StatusCode

from debrix.payloads import (
    PayloadJob,
    build_messages_preview,
    build_response_preview,
    ensure_worker,
    get_preview_chars,
    sha256_hex,
)
from debrix.semconv import Attr, Event

__all__ = ["DebrixSpan", "ALLOWED_MESSAGE_ROLES"]

ALLOWED_MESSAGE_ROLES: frozenset[str] = frozenset(
    {"system", "user", "assistant", "tool"}
)


def _normalize_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, Mapping):
            raise TypeError(
                f"messages[{i}] must be a mapping, got {type(msg).__name__}"
            )
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str) or role not in ALLOWED_MESSAGE_ROLES:
            raise ValueError(
                f"messages[{i}].role must be one of "
                f"{sorted(ALLOWED_MESSAGE_ROLES)}, got {role!r}"
            )
        if not isinstance(content, str):
            raise TypeError(
                f"messages[{i}].content must be a str, "
                f"got {type(content).__name__}"
            )
        entry: dict[str, str] = {"role": role, "content": content}
        name = msg.get("name")
        if name is not None:
            if not isinstance(name, str):
                raise TypeError(
                    f"messages[{i}].name must be a str when provided, "
                    f"got {type(name).__name__}"
                )
            entry["name"] = name
        normalized.append(entry)
    return normalized


def _service_name_from_span(span: Span) -> str:
    # Resource is on the tracer provider; fall back for tests.
    try:
        resource = span.resource  # type: ignore[attr-defined]
        if resource is not None:
            val = resource.attributes.get("service.name")
            if isinstance(val, str) and val:
                return val
    except Exception:  # noqa: BLE001
        pass
    return "debrix"


def _trace_id_hex(span: Span) -> str:
    ctx = span.get_span_context()
    return format(ctx.trace_id, "032x")


def _current_agent_name() -> str | None:
    # Lazy import: tracing → span → tracing would cycle at module load.
    from debrix.tracing import current_agent_name

    return current_agent_name()


class DebrixSpan:
    """Thin wrapper around an OpenTelemetry span with Debrix helpers."""

    def __init__(self, span: Span) -> None:
        self._span = span

    @property
    def otel_span(self) -> Span:
        """Underlying OpenTelemetry span."""
        return self._span

    def record_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
    ) -> None:
        """Record opt-in conversation messages on this span.

        Full bodies are uploaded asynchronously to Debrix ``/v1/payloads``.
        The span receives preview + stats + ``blob_ref`` immediately so the
        agent thread never waits on disk/network for large prompts.
        """
        normalized = _normalize_messages(messages)
        preview_chars = get_preview_chars()
        preview, truncated = build_messages_preview(normalized, preview_chars)
        body = json.dumps(normalized, separators=(",", ":")).encode("utf-8")
        digest = sha256_hex(body)

        self._span.set_attribute(
            Attr.MESSAGES_PREVIEW, json.dumps(preview, separators=(",", ":"))
        )
        self._span.set_attribute(Attr.MESSAGES_BYTES, len(body))
        self._span.set_attribute(Attr.MESSAGES_CHARS, len(body.decode("utf-8")))
        self._span.set_attribute(Attr.MESSAGES_COUNT, len(normalized))
        self._span.set_attribute(Attr.MESSAGES_TRUNCATED, truncated)

        blob_ref = f"sha256:{digest}"
        self._span.set_attribute(Attr.MESSAGES_BLOB_REF, blob_ref)
        # Small payloads: also keep legacy inline for older UIs / offline Debrix.
        if len(body) <= 64_000:
            self._span.set_attribute(Attr.MESSAGES, body.decode("utf-8"))

        from debrix.config import DEFAULT_OTLP_ENDPOINT, _resolved_endpoint

        worker = ensure_worker(_resolved_endpoint() or DEFAULT_OTLP_ENDPOINT)
        err = worker.try_enqueue(
            PayloadJob(
                kind="messages",
                body=body,
                sha256_hex=digest,
                span_context=self._span.get_span_context(),
                service_name=_service_name_from_span(self._span),
                trace_id_hex=_trace_id_hex(self._span),
                agent_name=_current_agent_name(),
            )
        )
        if err:
            self._span.set_attribute(Attr.MESSAGES_CAPTURE_ERROR, err)
        else:
            self._emit_payload_ready("messages", blob_ref)

    def record_response(self, response: Mapping[str, Any]) -> None:
        """Record opt-in model output / usage on this span.

        Expected loose shape (extra keys allowed)::

            {
              "content": "...",
              "model": "...",
              "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        """
        if not isinstance(response, Mapping):
            raise TypeError(
                f"response must be a mapping, got {type(response).__name__}"
            )
        payload = dict(response)
        preview_chars = get_preview_chars()
        preview, truncated = build_response_preview(payload, preview_chars)
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        digest = sha256_hex(body)

        self._span.set_attribute(
            Attr.RESPONSE_PREVIEW, json.dumps(preview, separators=(",", ":"))
        )
        self._span.set_attribute(Attr.RESPONSE_BYTES, len(body))
        self._span.set_attribute(Attr.RESPONSE_CHARS, len(body.decode("utf-8")))
        self._span.set_attribute(Attr.RESPONSE_TRUNCATED, truncated)

        blob_ref = f"sha256:{digest}"
        self._span.set_attribute(Attr.RESPONSE_BLOB_REF, blob_ref)
        if len(body) <= 64_000:
            self._span.set_attribute(Attr.RESPONSE, body.decode("utf-8"))

        from debrix.config import DEFAULT_OTLP_ENDPOINT, _resolved_endpoint

        worker = ensure_worker(_resolved_endpoint() or DEFAULT_OTLP_ENDPOINT)
        err = worker.try_enqueue(
            PayloadJob(
                kind="response",
                body=body,
                sha256_hex=digest,
                span_context=self._span.get_span_context(),
                service_name=_service_name_from_span(self._span),
                trace_id_hex=_trace_id_hex(self._span),
                agent_name=_current_agent_name(),
            )
        )
        if err:
            self._span.set_attribute(Attr.RESPONSE_CAPTURE_ERROR, err)
        else:
            self._emit_payload_ready("response", blob_ref)

    def _emit_payload_ready(self, kind: str, blob_ref: str) -> None:
        """Mark the content-addressed blob as queued (see Semantic Model)."""
        self._span.add_event(
            Event.PAYLOAD_READY,
            {
                Attr.PAYLOAD_KIND: kind,
                Attr.PAYLOAD_BLOB_REF: blob_ref,
            },
        )

    def record_exception(self, exc: BaseException) -> None:
        """Mark the span as errored and attach a short Debrix summary."""
        self._span.set_status(Status(StatusCode.ERROR, str(exc)))
        self._span.record_exception(exc)
        summary = f"{type(exc).__name__}: {exc}"
        if len(summary) > 200:
            summary = summary[:197] + "..."
        self._span.set_attribute(Attr.ERROR_SUMMARY, summary)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the underlying span."""
        self._span.set_attribute(key, value)
