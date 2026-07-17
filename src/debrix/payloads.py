"""Background payload worker: full conversation bodies → Debrix /v1/payloads."""

from __future__ import annotations

import atexit
import gzip
import hashlib
import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

from opentelemetry.trace import SpanContext

logger = logging.getLogger("debrix.payloads")

CaptureMode = Literal["full", "preview", "off"]

DEFAULT_MAX_PAYLOAD_BYTES = 209_715_200  # ~200 MiB
DEFAULT_PREVIEW_CHARS = 4096
DEFAULT_QUEUE_SIZE = 64


@dataclass
class PayloadJob:
    kind: Literal["messages", "response"]
    body: bytes
    sha256_hex: str
    span_context: SpanContext
    service_name: str
    trace_id_hex: str
    agent_name: str | None = None


_worker: PayloadWorker | None = None
_worker_lock = threading.Lock()
_atexit_registered = False


def get_capture_mode() -> CaptureMode:
    env = os.environ.get("DEBRIX_CAPTURE_MESSAGES", "").strip().lower()
    if env in ("full", "preview", "off"):
        return env  # type: ignore[return-value]
    # Desktop Memory tab writes settings.json next to the app DB.
    for candidate in _settings_candidates():
        try:
            with open(candidate, encoding="utf-8") as f:
                data = json.load(f)
            mode = str(data.get("captureMessages", "")).lower()
            if mode in ("full", "preview", "off"):
                return mode  # type: ignore[return-value]
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    return "full"


def get_max_payload_bytes() -> int:
    raw = os.environ.get("DEBRIX_MAX_PAYLOAD_BYTES")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return DEFAULT_MAX_PAYLOAD_BYTES


def get_preview_chars() -> int:
    raw = os.environ.get("DEBRIX_PREVIEW_CHARS")
    if raw:
        try:
            return max(64, int(raw))
        except ValueError:
            pass
    return DEFAULT_PREVIEW_CHARS


def _settings_candidates() -> list[str]:
    home = os.path.expanduser("~")
    return [
        os.environ.get("DEBRIX_SETTINGS_PATH", ""),
        os.path.join(
            home,
            "Library",
            "Application Support",
            "com.debrix.app",
            "settings.json",
        ),
        os.path.join(home, ".debrix", "settings.json"),
    ]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def truncate_text(text: str, budget: int) -> tuple[str, bool]:
    if len(text) <= budget:
        return text, False
    head = budget // 2
    tail = budget - head
    return text[:head] + "…[preview]…" + text[-tail:], True


def build_messages_preview(
    messages: list[dict[str, str]], preview_chars: int
) -> tuple[list[dict[str, str]], bool]:
    truncated = False
    out: list[dict[str, str]] = []
    for msg in messages:
        entry = dict(msg)
        content, was = truncate_text(entry.get("content", ""), preview_chars)
        entry["content"] = content
        truncated = truncated or was
        out.append(entry)
    return out, truncated


def build_response_preview(
    response: dict[str, Any], preview_chars: int
) -> tuple[dict[str, Any], bool]:
    out = dict(response)
    truncated = False
    content = out.get("content")
    if isinstance(content, str):
        shortened, was = truncate_text(content, preview_chars)
        out["content"] = shortened
        truncated = was
    return out, truncated


class PayloadWorker:
    def __init__(
        self,
        *,
        endpoint: str,
        max_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        queue_size: int = DEFAULT_QUEUE_SIZE,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._max_bytes = max_bytes
        self._queue: queue.Queue[PayloadJob | None] = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(
            target=self._run, name="debrix-payload-worker", daemon=True
        )
        self._started = False
        self._pending = 0
        self._pending_lock = threading.Lock()
        self._idle = threading.Condition(self._pending_lock)

    def start(self) -> None:
        if not self._started:
            self._thread.start()
            self._started = True

    def stop(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def try_enqueue(self, job: PayloadJob) -> str | None:
        """Enqueue job. Returns error string on failure (overflow / over-cap)."""
        if len(job.body) > self._max_bytes:
            return f"payload exceeds max size ({len(job.body)} > {self._max_bytes} bytes)"
        with self._pending_lock:
            self._pending += 1
        try:
            self._queue.put_nowait(job)
            return None
        except queue.Full:
            with self._pending_lock:
                self._pending -= 1
                if self._pending <= 0:
                    self._idle.notify_all()
            return "payload queue full"

    def flush(self, timeout_millis: int = 10_000) -> bool:
        """Block until queued uploads finish (or timeout). Returns True if idle."""
        deadline = time.monotonic() + (timeout_millis / 1000.0)
        with self._idle:
            while self._pending > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "payload flush timed out with %s job(s) still pending",
                        self._pending,
                    )
                    return False
                self._idle.wait(timeout=remaining)
            return True

    def _job_done(self) -> None:
        with self._idle:
            self._pending = max(0, self._pending - 1)
            if self._pending == 0:
                self._idle.notify_all()

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                self._upload(job)
            except Exception:  # noqa: BLE001 — never kill worker
                logger.exception("payload upload failed")
            finally:
                self._job_done()

    def _upload(self, job: PayloadJob) -> None:
        compressed = gzip.compress(job.body)
        url = f"{self._endpoint}/v1/payloads"
        req = urllib.request.Request(url, data=compressed, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Content-Encoding", "gzip")
        req.add_header("X-Debrix-Sha256", job.sha256_hex)
        req.add_header("X-Debrix-Kind", job.kind)
        req.add_header("X-Debrix-Trace-Id", job.trace_id_hex)
        req.add_header("X-Debrix-Service-Name", job.service_name)
        if job.agent_name:
            req.add_header("X-Debrix-Agent-Name", job.agent_name)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp.read()
            logger.debug(
                "uploaded %s payload sha256:%s (%s bytes)",
                job.kind,
                job.sha256_hex,
                len(job.body),
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            logger.error("payload upload HTTP %s: %s", exc.code, body)
            return
        except urllib.error.URLError as exc:
            logger.error("payload upload failed: %s", exc)
            return


def ensure_worker(endpoint: str) -> PayloadWorker:
    global _worker, _atexit_registered
    with _worker_lock:
        if _worker is None:
            _worker = PayloadWorker(
                endpoint=endpoint, max_bytes=get_max_payload_bytes()
            )
            _worker.start()
            if not _atexit_registered:
                atexit.register(_atexit_flush)
                _atexit_registered = True
        return _worker


def flush_payloads(timeout_millis: int = 10_000) -> bool:
    """Flush the global payload worker if it exists."""
    with _worker_lock:
        worker = _worker
    if worker is None:
        return True
    return worker.flush(timeout_millis=timeout_millis)


def _atexit_flush() -> None:
    flush_payloads(timeout_millis=10_000)


def reset_worker_for_tests() -> None:
    global _worker
    with _worker_lock:
        if _worker is not None:
            _worker.stop()
            _worker.flush(timeout_millis=1_000)
        _worker = None
