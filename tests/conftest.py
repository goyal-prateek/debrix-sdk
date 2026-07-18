"""Shared pytest fixtures for the Debrix SDK tests."""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from debrix.config import reset_for_tests

_exporter: InMemorySpanExporter | None = None


def pytest_configure() -> None:
    """Install a single in-memory TracerProvider for the whole test session.

    OpenTelemetry forbids replacing a TracerProvider once set, so tests share
    one provider and clear the exporter between cases.
    """
    global _exporter
    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        # A prior import/configure already installed a provider; attach exporter.
        _exporter = InMemorySpanExporter()
        current.add_span_processor(SimpleSpanProcessor(_exporter))
        return

    _exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(_exporter))
    trace.set_tracer_provider(provider)
    reset_for_tests()
    # Mark configured so lazy configure() does not attach an OTLP exporter.
    from debrix import configure

    configure(batch=False)


@pytest.fixture()
def memory_exporter() -> InMemorySpanExporter:
    assert _exporter is not None
    _exporter.clear()
    # Reset agent-scoped replay sequence so tests don't leak indices.
    from debrix.tracing import _REPLAY_SEQUENCE

    token = _REPLAY_SEQUENCE.set(0)
    yield _exporter
    _REPLAY_SEQUENCE.reset(token)
    _exporter.clear()
