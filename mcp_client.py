"""
IT Support Chatbot - MCP client for the ticket server (mcp_server.py).

`MCPToolClient` starts mcp_server.py as a subprocess, connects to it over the
stdio transport with the official MCP SDK, discovers its tools and calls them.
The SDK is asynchronous while Streamlit is synchronous, so the client runs one
asyncio event loop in a background thread and keeps a single MCP session open:
    - start()       spawns the server, initializes the session, lists tools,
    - tool_schemas()   tool definitions in the Claude Messages API format,
    - execute()     calls a tool and never raises (errors become results),
    - close()       ends the session and the subprocess (also run at exit).
If the server crashes or a call times out, the session is closed and the next
call reconnects automatically.

The same client and server are used in live mode (Claude picks the tool) and
in mock mode (keyword rules pick the tool, see choose_mock_ticket_calls): in
both cases the ticket is created by a real MCP request, never by calling the
server functions directly.
"""

import asyncio
import atexit
import hashlib
import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

PROJECT_DIR = Path(__file__).resolve().parent
SERVER_SCRIPT = PROJECT_DIR / "mcp_server.py"
DEFAULT_TIMEOUT_SECONDS = 20
TICKET_TOOLS = ("create_ticket", "get_status_ticket")
# Parameters filled in by the application, never shown to Claude.
HIDDEN_PARAMETERS = {"create_ticket": ("idempotency_key",)}

MCP_INSTRUCTIONS = (
    "\n\nTickets: use create_ticket only when the user explicitly asks to open, create, log "
    "or raise a support ticket; write a short title, a description with the details the "
    "user gave, and a priority (low, medium, high or critical - default medium). Use "
    "get_status_ticket when the user asks about an existing ticket ID. Always tell the user "
    "the ticket ID and status returned by the tool; never invent ticket IDs."
)


class MCPClientError(Exception):
    """The MCP server could not be reached or did not answer; safe to show in the UI."""


@dataclass
class MCPToolResult:
    """Outcome of one MCP tool call."""

    output: str  # text sent back to Claude as the tool_result
    is_error: bool = False
    data: Optional[dict] = None  # parsed JSON result (ticket details) when available


def _env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


def load_mcp_config() -> dict:
    """Read the MCP settings from environment variables (call after load_dotenv)."""
    return {
        "enabled": _env_flag("MCP_ENABLED", "true"),
        "db_path": os.getenv("TICKET_DB_PATH", "tickets.db"),
        "timeout": float(os.getenv("MCP_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)),
    }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MCPToolClient:
    """Synchronous wrapper around one MCP stdio session (thread-safe)."""

    def __init__(self, db_path: str = "tickets.db", timeout: float = DEFAULT_TIMEOUT_SECONDS,
                 server_script: Path = SERVER_SCRIPT):
        self.db_path = db_path
        self.timeout = timeout
        self.server_script = Path(server_script)
        self.tools = []  # mcp.types.Tool objects from list_tools()
        self.last_error: Optional[str] = None
        self._lock = threading.RLock()
        self._loop = None
        self._thread = None
        self._session = None
        self._stop = None  # asyncio.Event that ends the session task
        self._task = None  # concurrent.futures.Future of the session task
        self._errlog = None
        self._atexit_registered = False

    # -- lifecycle ---------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._session is not None

    @property
    def tool_names(self) -> List[str]:
        return [tool.name for tool in self.tools]

    def start(self) -> "MCPToolClient":
        """Spawn the server and open the session (no-op if already connected)."""
        with self._lock:
            if self.connected:
                return self
            self.close()  # clean up a previous broken session
            if not self.server_script.is_file():
                self.last_error = "MCP server script not found."
                raise MCPClientError(self.last_error)

            ready = threading.Event()
            failure = []
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-client", daemon=True)
            self._thread.start()
            self._task = asyncio.run_coroutine_threadsafe(self._run_session(ready, failure), self._loop)

            if not ready.wait(self.timeout + 5) or failure or not self.connected:
                reason = failure[0] if failure else "the server did not start in time"
                self.close()
                self.last_error = f"Could not start the MCP ticket server: {reason}"
                raise MCPClientError(self.last_error)

            self.last_error = None
            if not self._atexit_registered:
                atexit.register(self.close)
                self._atexit_registered = True
            return self

    async def _run_session(self, ready: threading.Event, failure: list) -> None:
        """Open the stdio transport and session, then wait until close() is called.

        The transport is entered and exited in this single task, as the SDK requires.
        """
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import get_default_environment, stdio_client

        self._stop = asyncio.Event()
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(self.server_script)],
            env={**get_default_environment(), "TICKET_DB_PATH": self.db_path, "PYTHONIOENCODING": "utf-8"},
            cwd=str(self.server_script.parent),
        )
        # Server diagnostics are discarded: stdout carries the protocol, and some
        # hosts (pytest, Streamlit) replace sys.stderr with objects without a file handle.
        self._errlog = open(os.devnull, "w")
        try:
            async with stdio_client(params, errlog=self._errlog) as (read, write):
                async with ClientSession(read, write,
                                         read_timeout_seconds=timedelta(seconds=self.timeout)) as session:
                    await asyncio.wait_for(session.initialize(), self.timeout)
                    self.tools = (await asyncio.wait_for(session.list_tools(), self.timeout)).tools
                    self._session = session
                    ready.set()
                    await self._stop.wait()
        except BaseException as error:  # noqa: BLE001 - reported to start() / logged in last_error
            if not ready.is_set():
                failure.append(_describe(error))
            elif not self._stop.is_set():
                self.last_error = f"The MCP ticket server stopped: {_describe(error)}"
            if not isinstance(error, Exception):
                raise
        finally:
            self._session = None
            ready.set()

    def close(self) -> None:
        """End the session, stop the server subprocess and the event loop thread."""
        with self._lock:
            loop, task = self._loop, self._task
            if loop is not None and loop.is_running():
                if self._stop is not None:
                    loop.call_soon_threadsafe(self._stop.set)
                if task is not None:
                    try:
                        task.result(timeout=10)
                    except Exception:  # noqa: BLE001 - best effort during shutdown
                        task.cancel()
                loop.call_soon_threadsafe(loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=10)
            if loop is not None and not loop.is_running():
                loop.close()
            if self._errlog is not None:
                self._errlog.close()
            self._loop = self._thread = self._task = self._stop = self._session = self._errlog = None

    def __enter__(self) -> "MCPToolClient":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- tools -------------------------------------------------------------

    def tool_schemas(self) -> List[dict]:
        """Discovered tools as Claude tool definitions (sorted, hidden parameters removed)."""
        schemas = []
        for tool in sorted(self.tools, key=lambda item: item.name):
            schema = json.loads(json.dumps(tool.inputSchema or {"type": "object", "properties": {}}))
            for name in HIDDEN_PARAMETERS.get(tool.name, ()):
                schema.get("properties", {}).pop(name, None)
                if name in schema.get("required", []):
                    schema["required"].remove(name)
            schemas.append({
                "name": tool.name,
                "description": " ".join((tool.description or "").split()),
                "input_schema": schema,
            })
        return schemas

    def call_tool(self, name: str, arguments: dict):
        """Call an MCP tool and return the raw CallToolResult. Raises MCPClientError."""
        try:
            self.start()
            future = asyncio.run_coroutine_threadsafe(
                self._session.call_tool(name, arguments), self._loop)
            return future.result(timeout=self.timeout + 5)
        except MCPClientError:
            raise
        except Exception as error:  # noqa: BLE001 - connection lost, timeout...
            self.close()  # the next call reconnects
            self.last_error = f"MCP request failed: {_describe(error)}"
            raise MCPClientError(self.last_error) from error

    def execute(self, name: str, tool_input: dict, request_id: Optional[str] = None) -> MCPToolResult:
        """Run an MCP tool for the tool loop. Never raises: errors are returned as results."""
        arguments = {
            key: value for key, value in (tool_input or {}).items()
            if key not in HIDDEN_PARAMETERS.get(name, ())
        }
        if name == "create_ticket" and request_id:
            arguments["idempotency_key"] = idempotency_key(request_id, arguments)
        try:
            result = self.call_tool(name, arguments)
        except MCPClientError as error:
            return MCPToolResult(f"Ticket system unavailable: {error}", is_error=True)

        text = "\n".join(
            block.text for block in result.content or [] if getattr(block, "type", None) == "text"
        ).strip()
        data = result.structuredContent if isinstance(result.structuredContent, dict) else None
        if data is None and text.startswith("{"):
            try:
                data = json.loads(text)
            except ValueError:
                data = None
        if data is not None and set(data) == {"result"} and isinstance(data["result"], dict):
            data = data["result"]  # FastMCP wraps some return types in {"result": ...}
        output = json.dumps(data) if data is not None else (text or "No output.")
        return MCPToolResult(output, is_error=bool(result.isError), data=None if result.isError else data)


def idempotency_key(request_id: str, arguments: dict) -> str:
    """Same user message + same ticket content = same key, so a rerun cannot create a duplicate."""
    content = json.dumps(
        {key: arguments.get(key) for key in ("title", "description", "priority")}, sort_keys=True)
    return f"{request_id}:{hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]}"


def _describe(error: BaseException) -> str:
    """Short, readable error text (task groups wrap the real cause)."""
    while getattr(error, "exceptions", None):  # (Base)ExceptionGroup, also the 3.10 backport
        error = error.exceptions[0]
    message = str(error) or type(error).__name__
    return message.replace(str(PROJECT_DIR), ".")  # never show absolute local paths


# ---------------------------------------------------------------------------
# Mock mode: simulated tool choice (the MCP call itself is real)
# ---------------------------------------------------------------------------

TICKET_ID_PATTERN = re.compile(r"\bTCK-[0-9A-Za-z]+\b", re.I)
CREATE_PATTERN = re.compile(
    r"\b(create|open|raise|log|file|submit|new)\b.*\bticket\b|\bticket\b.*\b(create|open|raise|log)\b", re.I)
PRIORITY_WORDS = {"critical": "critical", "urgent": "critical", "high": "high",
                  "medium": "medium", "normal": "medium", "low": "low"}


def choose_mock_ticket_calls(question: str) -> List[tuple]:
    """
    Simulate Claude's choice of ticket tools with keyword rules.

    Returns a list of (tool_name, tool_input). The inputs are not validated
    here: the MCP server validates them, exactly as in live mode.
    """
    calls = []
    for ticket_id in dict.fromkeys(match.group(0) for match in TICKET_ID_PATTERN.finditer(question)):
        calls.append(("get_status_ticket", {"ticket_id": ticket_id}))
    if not calls and CREATE_PATTERN.search(question):
        priority = next(
            (value for word, value in PRIORITY_WORDS.items() if re.search(rf"\b{word}\b", question, re.I)),
            "medium")
        # "Create a ticket: VPN drops every hour" -> title "VPN drops every hour".
        summary = question.split(":", 1)[1].strip() if ":" in question else question.strip()
        title = summary if len(summary) <= 80 else summary[:80].rsplit(" ", 1)[0] + "..."
        calls.append(("create_ticket", {"title": title, "description": question.strip(), "priority": priority}))
    return calls


def describe_ticket_event(event: dict) -> str:
    """Markdown line summarising a ticket tool call, for the mock answer."""
    data = event.get("data") or {}
    if event.get("is_error"):
        return f"🎫 `{event['name']}` returned an error from the MCP server: {event['output']}"
    if event["name"] == "create_ticket":
        prefix = "already existed (duplicate request ignored)" if data.get("duplicate") else "was created"
        return (f"🎫 Ticket **{data.get('ticket_id')}** {prefix} via MCP: *{data.get('title')}* "
                f"(priority **{data.get('priority')}**, status **{data.get('status')}**).")
    if not data.get("found"):
        return f"🎫 {data.get('message', 'Ticket not found.')}"
    return (f"🎫 Ticket **{data['ticket_id']}** is **{data['status']}** "
            f"(priority {data['priority']}, created {data['created_at']}).")
