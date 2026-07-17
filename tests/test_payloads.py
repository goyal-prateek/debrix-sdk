"""Tests for payload capture: preview, caps, upload, non-blocking enqueue."""

from __future__ import annotations

import gzip
import http.server
import json
import os
import threading
import time
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanContext, TraceFlags

from debrix import Attr, Event, SpanKind, trace_agent, trace_span
from debrix.payloads import (
    PayloadJob,
    PayloadWorker,
    build_messages_preview,
    build_response_preview,
    get_capture_mode,
    reset_worker_for_tests,
    sha256_hex,
)


def test_build_messages_preview_truncates_long_content() -> None:
    messages = [{"role": "system", "content": "x" * 100}]
    preview, truncated = build_messages_preview(messages, preview_chars=20)
    assert truncated is True
    assert "…[preview]…" in preview[0]["content"]
    assert len(preview[0]["content"]) < 100


def test_build_response_preview_truncates_content() -> None:
    preview, truncated = build_response_preview(
        {"content": "y" * 100, "model": "m"}, preview_chars=20
    )
    assert truncated is True
    assert "…[preview]…" in preview["content"]
    assert preview["model"] == "m"


def test_record_messages_sets_preview_and_blob_ref(
    memory_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEBRIX_CAPTURE_MESSAGES", "full")
    reset_worker_for_tests()

    with trace_span("llm", kind=SpanKind.LLM) as span:
        span.record_messages([{"role": "user", "content": "hello"}])

    finished = memory_exporter.get_finished_spans()[0]
    attrs = finished.attributes
    assert Attr.MESSAGES_PREVIEW in attrs
    assert Attr.MESSAGES_BLOB_REF in attrs
    assert attrs[Attr.MESSAGES_BLOB_REF].startswith("sha256:")
    assert attrs[Attr.MESSAGES_COUNT] == 1
    # Small body still sets legacy inline attribute
    assert json.loads(attrs[Attr.MESSAGES])[0]["content"] == "hello"
    ready = [e for e in finished.events if e.name == Event.PAYLOAD_READY]
    assert len(ready) == 1
    assert ready[0].attributes[Attr.PAYLOAD_KIND] == "messages"
    assert ready[0].attributes[Attr.PAYLOAD_BLOB_REF] == attrs[Attr.MESSAGES_BLOB_REF]


def test_record_response_sets_preview_and_blob_ref(
    memory_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEBRIX_CAPTURE_MESSAGES", "full")
    reset_worker_for_tests()

    with trace_span("llm", kind=SpanKind.LLM) as span:
        span.record_response(
            {
                "content": "hello back",
                "model": "test-model",
                "usage": {"input_tokens": 2, "output_tokens": 3},
            }
        )

    finished = memory_exporter.get_finished_spans()[0]
    attrs = finished.attributes
    assert Attr.RESPONSE_PREVIEW in attrs
    assert Attr.RESPONSE_BLOB_REF in attrs
    assert attrs[Attr.RESPONSE_BLOB_REF].startswith("sha256:")
    assert attrs[Attr.RESPONSE_TRUNCATED] is False
    assert json.loads(attrs[Attr.RESPONSE])["content"] == "hello back"
    ready = [e for e in finished.events if e.name == Event.PAYLOAD_READY]
    assert len(ready) == 1
    assert ready[0].attributes[Attr.PAYLOAD_KIND] == "response"
    assert ready[0].attributes[Attr.PAYLOAD_BLOB_REF] == attrs[Attr.RESPONSE_BLOB_REF]


def test_preview_mode_skips_blob_ref(
    memory_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEBRIX_CAPTURE_MESSAGES", "preview")
    reset_worker_for_tests()

    with trace_span("llm", kind=SpanKind.LLM) as span:
        span.record_messages([{"role": "user", "content": "hi"}])
        span.record_response({"content": "yo"})

    attrs = memory_exporter.get_finished_spans()[0].attributes
    assert Attr.MESSAGES_PREVIEW in attrs
    assert Attr.MESSAGES_BLOB_REF not in attrs
    assert Attr.MESSAGES in attrs
    assert Attr.RESPONSE_PREVIEW in attrs
    assert Attr.RESPONSE_BLOB_REF not in attrs
    assert Attr.RESPONSE in attrs
    events = memory_exporter.get_finished_spans()[0].events
    assert not any(e.name == Event.PAYLOAD_READY for e in events)


def test_off_mode_records_nothing(
    memory_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEBRIX_CAPTURE_MESSAGES", "off")
    reset_worker_for_tests()

    with trace_span("llm", kind=SpanKind.LLM) as span:
        span.record_messages([{"role": "user", "content": "hi"}])
        span.record_response({"content": "yo"})

    attrs = memory_exporter.get_finished_spans()[0].attributes
    assert Attr.MESSAGES not in attrs
    assert Attr.MESSAGES_PREVIEW not in attrs
    assert Attr.RESPONSE not in attrs
    assert Attr.RESPONSE_PREVIEW not in attrs


def test_over_cap_sets_capture_error_without_raising(
    memory_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEBRIX_CAPTURE_MESSAGES", "full")
    monkeypatch.setenv("DEBRIX_MAX_PAYLOAD_BYTES", "50")
    reset_worker_for_tests()

    big = "y" * 200
    with trace_span("llm", kind=SpanKind.LLM) as span:
        span.record_messages([{"role": "system", "content": big}])
        span.record_response({"content": big})

    attrs = memory_exporter.get_finished_spans()[0].attributes
    assert Attr.MESSAGES_CAPTURE_ERROR in attrs
    assert "exceeds max" in attrs[Attr.MESSAGES_CAPTURE_ERROR]
    assert Attr.RESPONSE_CAPTURE_ERROR in attrs
    assert "exceeds max" in attrs[Attr.RESPONSE_CAPTURE_ERROR]
    events = memory_exporter.get_finished_spans()[0].events
    assert not any(e.name == Event.PAYLOAD_READY for e in events)


def test_try_enqueue_returns_quickly_even_if_upload_slow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent path must not wait on the worker upload."""
    reset_worker_for_tests()

    class SlowWorker(PayloadWorker):
        def _upload(self, job: PayloadJob) -> None:  # type: ignore[override]
            time.sleep(0.2)

    worker = SlowWorker(endpoint="http://127.0.0.1:9", queue_size=8)
    worker.start()
    body = b'[{"role":"user","content":"x"}]'
    ctx = SpanContext(
        trace_id=1,
        span_id=2,
        is_remote=False,
        trace_flags=TraceFlags(0x01),
    )
    t0 = time.perf_counter()
    err = worker.try_enqueue(
        PayloadJob(
            kind="messages",
            body=body,
            sha256_hex=sha256_hex(body),
            span_context=ctx,
            service_name="t",
            trace_id_hex="01",
            agent_name="research_agent",
        )
    )
    elapsed = time.perf_counter() - t0
    assert err is None
    assert elapsed < 0.1
    assert worker.flush(timeout_millis=2_000) is True
    worker.stop()


def test_payload_worker_http_upload_headers_and_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wire-level: gzip body + Debrix headers hit /v1/payloads."""
    reset_worker_for_tests()
    received: dict[str, Any] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            received["path"] = self.path
            received["headers"] = {k.lower(): v for k, v in self.headers.items()}
            received["body"] = self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = b'[{"role":"user","content":"wire"}]'
        digest = sha256_hex(body)
        worker = PayloadWorker(endpoint=f"http://127.0.0.1:{port}", queue_size=4)
        worker.start()
        ctx = SpanContext(
            trace_id=0xABCD,
            span_id=0xEF,
            is_remote=False,
            trace_flags=TraceFlags(0x01),
        )
        err = worker.try_enqueue(
            PayloadJob(
                kind="response",
                body=body,
                sha256_hex=digest,
                span_context=ctx,
                service_name="svc",
                trace_id_hex=format(0xABCD, "032x"),
                agent_name="research_agent",
            )
        )
        assert err is None
        assert worker.flush(timeout_millis=5_000) is True
        worker.stop()
    finally:
        server.shutdown()

    assert received["path"] == "/v1/payloads"
    headers = received["headers"]
    assert headers["content-encoding"] == "gzip"
    assert headers["x-debrix-sha256"] == digest
    assert headers["x-debrix-kind"] == "response"
    assert headers["x-debrix-service-name"] == "svc"
    assert headers["x-debrix-agent-name"] == "research_agent"
    assert headers["x-debrix-trace-id"] == format(0xABCD, "032x")
    assert gzip.decompress(received["body"]) == body


def test_payload_upload_includes_agent_name_from_trace_agent(
    memory_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEBRIX_CAPTURE_MESSAGES", "full")
    reset_worker_for_tests()
    jobs: list[PayloadJob] = []

    class CaptureWorker(PayloadWorker):
        def try_enqueue(self, job: PayloadJob) -> str | None:  # type: ignore[override]
            jobs.append(job)
            return None

    monkeypatch.setattr(
        "debrix.span.ensure_worker",
        lambda _endpoint: CaptureWorker(endpoint="http://127.0.0.1:9"),
    )

    @trace_agent(name="research_agent")
    def run() -> None:
        with trace_span("complete", kind=SpanKind.LLM) as span:
            span.record_messages([{"role": "user", "content": "hi"}])
            span.record_response({"content": "ok"})

    run()
    assert len(jobs) == 2
    assert all(j.agent_name == "research_agent" for j in jobs)
    # silence unused fixture lint — exporter still needed for tracer setup
    assert memory_exporter.get_finished_spans()


def test_get_capture_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEBRIX_CAPTURE_MESSAGES", "preview")
    assert get_capture_mode() == "preview"
    monkeypatch.delenv("DEBRIX_CAPTURE_MESSAGES", raising=False)
    # Without settings file, default full
    if not any(
        os.path.isfile(p)
        for p in [
            os.path.expanduser(
                "~/Library/Application Support/com.debrix.app/settings.json"
            ),
            os.path.expanduser("~/.debrix/settings.json"),
        ]
    ):
        assert get_capture_mode() == "full"
