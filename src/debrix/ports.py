"""Listen-port defaults from the packaged ``ports.json``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, TypedDict


class PortsFile(TypedDict):
    otlpPort: int
    mcpPort: int
    host: str


def _load() -> PortsFile:
    path = Path(__file__).with_name("ports.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "otlpPort": int(data["otlpPort"]),
        "mcpPort": int(data["mcpPort"]),
        "host": str(data["host"]),
    }


PORTS: Final[PortsFile] = _load()
DEFAULT_OTLP_ENDPOINT: Final[str] = f"http://{PORTS['host']}:{PORTS['otlpPort']}"
DEFAULT_MCP_URL: Final[str] = f"http://{PORTS['host']}:{PORTS['mcpPort']}/mcp"
