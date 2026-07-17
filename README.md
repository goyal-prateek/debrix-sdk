# debrix

Open-source instrumentation SDK for [Debrix](https://github.com/goyal-prateek/debrix-sdk) — local-first AI Agent DevTools.

**Status:** alpha (`0.1.0a1`). APIs may change.

Requires the Debrix desktop app running locally to receive traces (OTLP/HTTP on `localhost:4318`).

## Install

```bash
pip install debrix
```

TestPyPI (pre-release smoke):

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ debrix==0.1.0a1
```

## Quick start

```python
from debrix import configure, force_flush, trace_agent, trace_tool, trace_span, SpanKind

configure(batch=False)  # OTLP/HTTP → http://127.0.0.1:4318

@trace_agent
def run_agent(query: str) -> str:
    return research(query)

@trace_tool(name="search")
def research(query: str) -> str:
    with trace_span("complete", kind=SpanKind.LLM) as span:
        span.record_messages([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": query},
        ])
        answer = "..."
        span.record_response({
            "content": answer,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })
        return answer

run_agent("hello")
force_flush()  # flush OTLP *and* conversation payload uploads before exit
```

Decorators also work as context managers:

```python
with trace_agent("planner") as span:
    with trace_tool("lookup"):
        ...
```

## Public API

| Symbol | Purpose |
| ------ | ------- |
| `configure()` | Install OTLP/HTTP exporter to Debrix (`:4318`) |
| `force_flush()` | Flush OTLP spans + pending `/v1/payloads` uploads (call before short scripts exit) |
| `trace_agent` | Agent boundary (decorator or `with trace_agent("name")`) |
| `trace_tool` | Tool call span; decorator records `debrix.replay.input` / `output` |
| `trace_span` | Generic / LLM / custom span context manager |
| `DebrixSpan.record_messages(...)` | Opt-in message payloads |
| `DebrixSpan.record_response(...)` | Opt-in model output / tokens |
| `SpanKind`, `Attr` | Semantic convention constants |

Nested calls propagate via OpenTelemetry context. On exception, spans are marked `ERROR` with `debrix.error.summary`.

## Develop

```bash
uv sync --group dev
uv run pytest
```

## License

MIT
