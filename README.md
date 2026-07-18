# debrix

Open-source instrumentation SDK for Debrix — local-first AI Agent DevTools.

**Status:** alpha (`0.1.0a3`). APIs may change.

Requires the Debrix desktop app running locally to receive traces (OTLP/HTTP on `localhost:17418`).

Source: [goyal-prateek/debrix-sdk](https://github.com/goyal-prateek/debrix-sdk)

## Install

```bash
pip install debrix
```

## Quick start

```python
from debrix import configure, force_flush, trace_agent, trace_tool, trace_span, SpanKind

configure(batch=False)  # OTLP/HTTP → http://127.0.0.1:17418

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
| `configure()` | Install OTLP/HTTP exporter to Debrix (`:17418`) |
| `force_flush()` | Flush OTLP spans + pending `/v1/payloads` uploads (call before short scripts exit) |
| `trace_agent` | Agent boundary (decorator or `with trace_agent("name")`) |
| `trace_tool` | Tool call span; records replay I/O + sequence index; consults Tool Mocker / Replay |
| `trace_span` | Generic / LLM / custom span context manager |
| `DebrixSpan.record_messages(...)` | Opt-in message payloads |
| `DebrixSpan.record_response(...)` | Opt-in model output / tokens |
| `MockableClient` | Opt-in MCP client wrapper for Tool Mocker (`debrix.mcp`) |
| `MockToolError` | Raised when a mock rule returns error/timeout |
| `SpanKind`, `Attr` | Semantic convention constants |

Nested calls propagate via OpenTelemetry context. On exception, spans are marked `ERROR` with `debrix.error.summary`.

## Tool Mocker & Deterministic Replay

When the Debrix desktop app is running, `@trace_tool` / `MockableClient` ask
`POST {otlp}/mocks/resolve` before calling the real function.

- **Tool Mocker:** rules from the app’s **Tool Mocks** panel → `action: mock`
- **Replay (Mode A):** an armed Observe **Replay** session → `action: replay`
  (recorded tool tape; LLM stays live)

If Debrix is down or times out (~200ms), the SDK **passthrough** to the real
tool.

```python
from debrix.mcp import MockableClient

client = MockableClient(real_mcp_client, server="demo-db")
result = await client.call_tool("query", {"sql": "select 1"})
```

Mocked spans set `debrix.mocked=true`; replayed spans set `debrix.replayed=true`.

## Develop

```bash
uv sync --group dev
uv run pytest
```

## License

MIT

## Release

Tag on `main` to publish to PyPI (GitHub Actions):

```bash
git tag -a v0.1.0a4 -m "debrix 0.1.0a4"
git push origin v0.1.0a4
```
