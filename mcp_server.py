"""
IT Support Chatbot - MCP server for IT support tickets (stdio transport).

A local Model Context Protocol server built with FastMCP (official Python MCP
SDK). Tickets are stored in a SQLite database, so they persist across restarts.

Tools:
    create_ticket(title, description, priority)
        Validates the input, generates a unique ticket ID (e.g. TCK-1A2B3C4D),
        stores status "open", the priority and the creation timestamp, and
        returns the created ticket.
    get_status_ticket(ticket_id)
        Returns the ticket details, or {"found": false, ...} if it does not exist.

The app starts this file as a subprocess and talks to it over stdin/stdout
(see mcp_client.py); you normally never run it by hand. To try it manually:
    python mcp_server.py          (waits for MCP messages on stdin)

Environment:
    TICKET_DB_PATH  SQLite file (default: tickets.db next to this file).
                    Relative paths are resolved against this file's folder.

stdout is reserved for MCP messages: this module never prints.
"""

import os
import re
import secrets
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = "tickets.db"

PRIORITIES = ("low", "medium", "high", "critical")
INITIAL_STATUS = "open"
TITLE_LENGTH = (3, 120)
DESCRIPTION_LENGTH = (10, 4000)
IDEMPOTENCY_KEY_MAX_LENGTH = 200
TICKET_ID_PATTERN = re.compile(r"^TCK-[0-9A-F]{8}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id       TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    description     TEXT NOT NULL,
    priority        TEXT NOT NULL,
    status          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    idempotency_key TEXT UNIQUE
)
"""
COLUMNS = ("ticket_id", "title", "description", "priority", "status", "created_at", "updated_at")


class TicketValidationError(ValueError):
    """Invalid tool input; the message is safe to show to the user."""


# ---------------------------------------------------------------------------
# Storage (SQLite)
# ---------------------------------------------------------------------------


def get_db_path() -> Path:
    """Database file from TICKET_DB_PATH (read on every call, so tests can change it)."""
    path = Path(os.getenv("TICKET_DB_PATH", DEFAULT_DB_PATH))
    return path if path.is_absolute() else PROJECT_DIR / path


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open the database and create the table if needed (caller closes it)."""
    path = db_path or get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute(SCHEMA)
    return connection


def _ticket(row: sqlite3.Row) -> dict:
    return {column: row[column] for column in COLUMNS}


def _clean_text(value, field: str, limits) -> str:
    if not isinstance(value, str):
        raise TicketValidationError(f"{field} must be text.")
    text = " ".join(value.split()) if field == "title" else value.strip()
    minimum, maximum = limits
    if len(text) < minimum:
        raise TicketValidationError(f"{field} must be at least {minimum} characters long.")
    if len(text) > maximum:
        raise TicketValidationError(f"{field} must be at most {maximum} characters long.")
    return text


def validate_ticket_input(title, description, priority) -> tuple:
    """Return the cleaned (title, description, priority) or raise TicketValidationError."""
    title = _clean_text(title, "title", TITLE_LENGTH)
    description = _clean_text(description, "description", DESCRIPTION_LENGTH)
    if not isinstance(priority, str) or priority.strip().lower() not in PRIORITIES:
        raise TicketValidationError(f"priority must be one of: {', '.join(PRIORITIES)}.")
    return title, description, priority.strip().lower()


def normalize_ticket_id(ticket_id) -> str:
    """Return the ticket ID in canonical form (TCK-XXXXXXXX) or raise TicketValidationError."""
    if not isinstance(ticket_id, str) or not TICKET_ID_PATTERN.match(ticket_id.strip().upper()):
        raise TicketValidationError(
            "ticket_id must look like TCK-1A2B3C4D (TCK- followed by 8 hexadecimal characters)."
        )
    return ticket_id.strip().upper()


def create_ticket_record(title, description, priority, idempotency_key: Optional[str] = None,
                         db_path: Optional[Path] = None) -> dict:
    """
    Validate and store a new ticket. Returns the ticket plus "duplicate".

    If `idempotency_key` was already used, the existing ticket is returned with
    duplicate=True instead of creating a second one (protects against retries
    and Streamlit reruns).
    """
    title, description, priority = validate_ticket_input(title, description, priority)
    if idempotency_key is not None:
        if not isinstance(idempotency_key, str) or len(idempotency_key) > IDEMPOTENCY_KEY_MAX_LENGTH:
            raise TicketValidationError("idempotency_key is invalid.")
        idempotency_key = idempotency_key.strip() or None

    with closing(connect(db_path)) as connection, connection:
        if idempotency_key:
            row = connection.execute(
                "SELECT * FROM tickets WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
            if row:
                return {**_ticket(row), "duplicate": True}

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for _ in range(5):  # 4 random bytes: a collision is very unlikely, but retry anyway
            ticket_id = f"TCK-{secrets.token_hex(4).upper()}"
            try:
                connection.execute(
                    "INSERT INTO tickets (ticket_id, title, description, priority, status, "
                    "created_at, updated_at, idempotency_key) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (ticket_id, title, description, priority, INITIAL_STATUS, now, now, idempotency_key),
                )
            except sqlite3.IntegrityError:
                continue
            row = connection.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
            return {**_ticket(row), "duplicate": False}
    raise RuntimeError("Could not generate a unique ticket ID. Please try again.")


def get_ticket_record(ticket_id, db_path: Optional[Path] = None) -> Optional[dict]:
    """Return the ticket, or None if it does not exist. Raises TicketValidationError on a bad ID."""
    ticket_id = normalize_ticket_id(ticket_id)
    with closing(connect(db_path)) as connection:
        row = connection.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
    return _ticket(row) if row else None


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("it-support-tickets", log_level="WARNING")


@mcp.tool()
def create_ticket(title: str, description: str, priority: str = "medium",
                  idempotency_key: Optional[str] = None) -> dict:
    """
    Create an IT support ticket and return its details (ticket_id, status, priority, created_at).

    Use it only when the user asks to open, create, log or raise a ticket.
    title: short summary of the issue (3-120 characters).
    description: what happens, since when, and what was already tried (10-4000 characters).
    priority: one of low, medium, high, critical.
    """
    # idempotency_key is filled in by the application (hidden from Claude) so that
    # re-sending the same request returns the existing ticket instead of a duplicate.
    try:
        return create_ticket_record(title, description, priority, idempotency_key)
    except TicketValidationError as error:
        raise ToolError(f"Invalid ticket: {error}") from error
    except sqlite3.Error as error:
        raise ToolError(f"Ticket database error: {error}") from error


@mcp.tool()
def get_status_ticket(ticket_id: str) -> dict:
    """
    Get the status and details of an IT support ticket by its ID (for example TCK-1A2B3C4D).

    Returns found=false with a message when no ticket has that ID.
    """
    try:
        ticket = get_ticket_record(ticket_id)
    except TicketValidationError as error:
        raise ToolError(f"Invalid ticket ID: {error}") from error
    except sqlite3.Error as error:
        raise ToolError(f"Ticket database error: {error}") from error
    if ticket is None:
        canonical = ticket_id.strip().upper()
        return {"found": False, "ticket_id": canonical, "message": f"No ticket found with ID {canonical}."}
    return {"found": True, **ticket}


if __name__ == "__main__":
    mcp.run()  # stdio transport (the default)
