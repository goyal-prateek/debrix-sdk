"""OTLP/HTTP exporter configuration for Debrix.

Default endpoint comes from ``ports.json`` (packaged ``ports.json``).
Override with ``configure(endpoint=...)`` or ``DEBRIX_OTLP_ENDPOINT`` — not the
shared ``OTEL_EXPORTER_OTLP_ENDPOINT`` (that would also redirect other OTel
exporters in the same process).
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Final

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

from debrix.payloads import ensure_worker, flush_payloads, get_capture_mode
from debrix.ports import DEFAULT_OTLP_ENDPOINT as _DEFAULT_OTLP_ENDPOINT

DEFAULT_OTLP_ENDPOINT: Final = _DEFAULT_OTLP_ENDPOINT
"""Debrix-only endpoint override (base URL, no ``/v1/traces`` suffix)."""
ENV_OTLP_ENDPOINT: Final = "DEBRIX_OTLP_ENDPOINT"

logger = logging.getLogger("debrix.config")

_configured = False
_endpoint: str = DEFAULT_OTLP_ENDPOINT


def _resolved_endpoint(endpoint: str | None = None) -> str:
    if endpoint:
        return endpoint.rstrip("/")
    return os.environ.get(ENV_OTLP_ENDPOINT, DEFAULT_OTLP_ENDPOINT).rstrip("/")


def force_flush(timeout_millis: int = 10_000) -> bool:
    """Flush OTLP spans and pending conversation payload uploads.

    Short-lived scripts must call this (or rely on atexit) before exit;
    otherwise ``blob_ref`` may land in Debrix without the payload body.
    """
    ok = True
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        ok = bool(provider.force_flush(timeout_millis=timeout_millis)) and ok
    ok = flush_payloads(timeout_millis=timeout_millis) and ok
    return ok


def configure(
    *,
    endpoint: str | None = None,
    service_name: str = "debrix",
    batch: bool = True,
) -> TracerProvider:
    """Configure a global TracerProvider that exports OTLP/HTTP to Debrix.

    Idempotent: subsequent calls return the existing provider without
    re-installing exporters.

    Args:
        endpoint: OTLP base URL (paths like ``/v1/traces`` are appended by the
            exporter). Defaults to the Debrix OTLP URL from ``ports.json``, or
            ``DEBRIX_OTLP_ENDPOINT`` when set.
        service_name: Value for ``service.name`` resource attribute.
        batch: Use ``BatchSpanProcessor`` when True (default); otherwise
            ``SimpleSpanProcessor`` (useful for short-lived scripts).
    """
    global _configured, _endpoint

    current = trace.get_tracer_provider()
    # OpenTelemetry forbids replacing an installed TracerProvider; reuse it.
    if isinstance(current, TracerProvider):
        _configured = True
        ensure_worker(_endpoint)
        return current

    resolved = _resolved_endpoint(endpoint)
    _endpoint = resolved
    # Ensure env protocol matches our contract when unset.
    os.environ.setdefault("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")

    if not batch and get_capture_mode() == "full":
        warnings.warn(
            "debrix.configure(batch=False) with full message capture may block "
            "the agent on span end while large exports flush; prefer batch=True "
            "for long-running agents.",
            UserWarning,
            stacklevel=2,
        )

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=f"{resolved}/v1/traces")
    processor = (
        BatchSpanProcessor(exporter) if batch else SimpleSpanProcessor(exporter)
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    _configured = True
    ensure_worker(resolved)
    return provider


def reset_for_tests() -> None:
    """Reset configuration flag (tests only)."""
    global _configured, _endpoint
    _configured = False
    _endpoint = DEFAULT_OTLP_ENDPOINT
    from debrix.payloads import reset_worker_for_tests

    reset_worker_for_tests()
