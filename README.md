# debrix

Open-source instrumentation SDK for Debrix — local-first AI Agent DevTools.

**Status:** beta (`0.1.0b1`). APIs may change.

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
with trace_agent("planner", arguments={"query": "hello"}) as span:
    with trace_tool("lookup"):
        ...
```

`@trace_agent` automatically records its bound function arguments, including
defaults, on `debrix.agent.arguments`. Context-managed agents can attach the
same JSON-safe field explicitly with `arguments={...}`. Agent return values are
not captured.

## Public API

| Symbol | Purpose |
| ------ | ------- |
| `configure()` | Install OTLP/HTTP exporter to Debrix (`:17418`) |
| `force_flush()` | Flush OTLP spans + pending `/v1/payloads` uploads (call before short scripts exit) |
| `trace_agent` | Agent boundary; decorators capture bound arguments, context managers accept `arguments={...}` |
| `trace_tool` | Tool call span; records replay I/O + sequence index; consults Tool Mocker / Replay |
| `trace_span` | Generic / LLM / custom span context manager |
| `DebrixSpan.record_messages(...)` | Opt-in message payloads |
| `DebrixSpan.record_response(...)` | Opt-in model output / tokens |
| `MockableClient` | Opt-in MCP client wrapper for Tool Mocker (`debrix.mcp`) |
| `MockToolError` | Raised when a mock rule returns error/timeout |
| `DebrixControlError`, `DebrixControlProtocolError`, `DebrixBreakpointCancelled`, `DebrixControlLost` | Typed fail-closed controlled-branch errors |
| `DebrixVerificationError`, `DebrixVerificationConfigurationError`, `DebrixVerificationProtocolError`, `DebrixVerificationRejected`, `DebrixVerificationControlLost` | Typed managed no-override verification errors |
| `SpanKind`, `Attr` | Semantic convention constants |

Calling `record_messages` or `record_response` stores the complete payload
locally in Debrix. Bounded previews remain on the span for fast inspection;
there are no partial or disabled capture modes.

Nested calls propagate via OpenTelemetry context. On exception, spans are marked `ERROR` with `debrix.error.summary`.

## Tool Mocker & Deterministic Replay

When the Debrix desktop app is running, `@trace_tool` / `MockableClient` ask
`POST {otlp}/mocks/resolve` before calling the real function.

- **Tool Mocker:** rules from the app’s **Tool Mocks** panel → `action: mock`
- **Replay (tools only):** armed Observe **Replay** → tools/MCP `action: replay`
- **Replay (tools + LLM stubs):** same session with **Tools + LLM**; use
  `debrix.llm.complete` so pinned LLM calls resolve as `action: replay`
  (`kind=llm`)

If Debrix is down or times out (~200ms), the SDK **passthrough** to the real
implementation.

```python
from debrix.mcp import MockableClient
from debrix.llm import acomplete, complete

client = MockableClient(real_mcp_client, server="demo-db")
result = await client.call_tool("query", {"sql": "select 1"})

answer = complete(
    messages,
    call=lambda msgs: my_provider(msgs),  # (content, usage, model)
)

answer = await acomplete(
    messages,
    call=my_async_provider,  # async (messages) -> (content, usage, model)
)
```

Stubbed spans set `debrix.stub` to `mock` (Tool Mocker) or `replay` (Deterministic Replay).

When a Debrix FW v2 branch is armed, `@trace_tool`, `MockableClient`,
`complete`, and `acomplete` check the fail-closed control channel before
ordinary mocks. Tool/MCP boundaries support controlled input, result, and
error decisions. A message breakpoint calls the provider exactly once with
the recorded or edited complete message list. A model-output breakpoint
returns the recorded or edited complete response without calling the
provider. Once Debrix claims a call, cancellation or connection loss raises a
typed control exception instead of silently running unmanaged.

## Managed no-override verification

Debrix FW v3 verifies the real project change through the existing
instrumented application. Starting a fix verification or regression rerun in
Debrix Desktop/MCP returns one attempt ID and one opaque token in this launch
environment:

```text
DEBRIX_VERIFICATION_ATTEMPT_ID=verification_…
DEBRIX_VERIFICATION_TOKEN=<one-time opaque capability>
```

Pass both values only to the next approved project-owned invocation. Do not
print, persist, trace, or copy the token into an artifact. No later public read
can recover the token. If the launch context is lost, cancel the durable
attempt and start a new one.

No extra SDK function starts the run. The first root `trace_agent` or
`trace_span` in that process automatically binds exactly one trace through
`debrix.verification.v1`. The managed context then:

- bypasses controlled branches, Tool Mocker, and Deterministic Replay before
  they can affect supported Tool/MCP/LLM boundaries;
- adds attempt, purpose, protocol, root-span, and no-override provenance but
  never the token;
- checks the local verification service fail-closed for sync and `asyncio`
  paths; and
- raises a typed verification error instead of falling through to diagnostic
  behavior when binding is rejected, the protocol is incompatible, or control
  is lost after binding.

The user or coding agent still chooses and invokes the existing project
command. Debrix does not execute tests, state predicates, judges, application
commands, or stored regression recipes. Call `force_flush()` before a
short-lived managed process exits so its trace and payload evidence can be
finalized.

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
git tag -a v0.1.0b1 -m "debrix 0.1.0b1"
git push origin v0.1.0b1
```
