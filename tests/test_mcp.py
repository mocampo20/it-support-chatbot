"""
Automated tests for MCP ticket management: the SQLite store, the real MCP
client/server communication over stdio, mock mode and the live tool loop.

The MCP tests start the real mcp_server.py subprocess with a temporary
database. No paid API calls are made: Claude responses are fake objects.
All ticket data is fictional.

Run:
    python -m pytest
"""

import json
from types import SimpleNamespace as NS

import pytest

import app_implementation
import mcp_client
import mcp_server
import tools

TOOLS_CONFIG = {
    "enabled": True, "web_search": True, "web_search_max_uses": 3,
    "web_search_version": "web_search_20260209", "max_iterations": 5,
}


def make_config(mock_mode=True, mcp_enabled=True, rag_enabled=False):
    return {"mock_mode": mock_mode, "model": "test", "max_tokens": 100, "refusal_fallback": False,
            "rag": {"enabled": rag_enabled, "top_k": 4, "min_similarity": 0.2},
            "tools": dict(TOOLS_CONFIG),
            "mcp": {"enabled": mcp_enabled, "db_path": "unused.db", "timeout": 20}}


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    return tmp_path_factory.mktemp("tickets") / "tickets.db"


@pytest.fixture(scope="module")
def client(db_path):
    """One real MCP server subprocess shared by the tests of this module."""
    with mcp_client.MCPToolClient(db_path=str(db_path)) as mcp:
        yield mcp


def create(client, request_id=None, **overrides):
    arguments = {"title": "Laptop cannot reach the VPN",
                 "description": "Since this morning the VPN client times out on a test laptop.",
                 "priority": "high", **overrides}
    return client.execute("create_ticket", arguments, request_id=request_id)


# ---------------------------------------------------------------------------
# Store and validation (server-side functions)
# ---------------------------------------------------------------------------


def test_store_creates_and_reads_ticket(tmp_path):
    ticket = mcp_server.create_ticket_record(
        "  Printer   offline ", "The shared test printer shows offline.", "LOW", db_path=tmp_path / "t.db")
    assert mcp_server.TICKET_ID_PATTERN.match(ticket["ticket_id"])
    assert ticket["title"] == "Printer offline" and ticket["priority"] == "low"
    assert ticket["status"] == "open" and ticket["created_at"].endswith("+00:00")
    assert not ticket["duplicate"]
    stored = mcp_server.get_ticket_record(ticket["ticket_id"].lower(), db_path=tmp_path / "t.db")
    assert stored["title"] == "Printer offline"


@pytest.mark.parametrize("title, description, priority, message", [
    ("ab", "A long enough description.", "low", "title must be at least"),
    ("x" * 121, "A long enough description.", "low", "title must be at most"),
    ("Valid title", "short", "low", "description must be at least"),
    ("Valid title", "A long enough description.", "urgent!", "priority must be one of"),
    (None, "A long enough description.", "low", "title must be text"),
])
def test_store_rejects_invalid_input(tmp_path, title, description, priority, message):
    with pytest.raises(mcp_server.TicketValidationError, match=message):
        mcp_server.create_ticket_record(title, description, priority, db_path=tmp_path / "t.db")
    assert not (tmp_path / "t.db").exists()  # validation happens before touching the database


def test_store_ticket_ids_are_unique(tmp_path):
    ids = {mcp_server.create_ticket_record("Disk full", "The test disk is full.", "medium",
                                           db_path=tmp_path / "t.db")["ticket_id"] for _ in range(50)}
    assert len(ids) == 50


# ---------------------------------------------------------------------------
# Real MCP client/server communication
# ---------------------------------------------------------------------------


def test_tools_are_discovered_with_claude_schemas(client):
    assert client.connected and client.tool_names == ["create_ticket", "get_status_ticket"]
    schemas = {schema["name"]: schema for schema in client.tool_schemas()}
    create_schema = schemas["create_ticket"]["input_schema"]
    assert set(create_schema["properties"]) == {"title", "description", "priority"}
    assert "idempotency_key" not in json.dumps(schemas)  # internal parameter hidden from Claude
    assert schemas["get_status_ticket"]["input_schema"]["required"] == ["ticket_id"]
    assert schemas["create_ticket"]["description"]


def test_create_and_get_ticket_over_mcp(client):
    created = create(client)
    assert not created.is_error
    ticket = created.data
    assert ticket["status"] == "open" and ticket["priority"] == "high" and ticket["created_at"]
    assert json.loads(created.output)["ticket_id"] == ticket["ticket_id"]

    status = client.execute("get_status_ticket", {"ticket_id": ticket["ticket_id"].lower()})
    assert not status.is_error
    assert status.data["found"] is True and status.data["ticket_id"] == ticket["ticket_id"]
    assert status.data["title"] == "Laptop cannot reach the VPN"


@pytest.mark.parametrize("overrides, message", [
    ({"title": "x"}, "title must be at least"),
    ({"description": "short"}, "description must be at least"),
    ({"priority": "whenever"}, "priority must be one of"),
])
def test_invalid_input_is_an_mcp_error_result(client, overrides, message):
    result = create(client, **overrides)
    assert result.is_error and message in result.output and result.data is None


def test_missing_ticket_is_handled_gracefully(client):
    result = client.execute("get_status_ticket", {"ticket_id": "TCK-00000000"})
    assert not result.is_error
    assert result.data == {"found": False, "ticket_id": "TCK-00000000",
                           "message": "No ticket found with ID TCK-00000000."}


def test_malformed_ticket_id_is_an_error(client):
    result = client.execute("get_status_ticket", {"ticket_id": "not-a-ticket"})
    assert result.is_error and "TCK-" in result.output


def test_same_request_id_never_creates_duplicates(client):
    first = create(client, request_id="msg-1")
    again = create(client, request_id="msg-1")
    other = create(client, request_id="msg-2")
    assert again.data["ticket_id"] == first.data["ticket_id"] and again.data["duplicate"]
    assert other.data["ticket_id"] != first.data["ticket_id"]


def test_claude_cannot_set_the_hidden_idempotency_key(client):
    first = create(client, request_id="msg-3")
    forged = create(client, request_id="msg-4", idempotency_key=f"msg-3:{'0' * 16}")
    assert forged.data["ticket_id"] != first.data["ticket_id"]


def test_tickets_persist_across_server_restarts(db_path):
    with mcp_client.MCPToolClient(db_path=str(db_path)) as first:
        ticket_id = create(first).data["ticket_id"]
    assert not first.connected  # subprocess and session closed
    with mcp_client.MCPToolClient(db_path=str(db_path)) as second:
        status = second.execute("get_status_ticket", {"ticket_id": ticket_id})
    assert status.data["found"] and status.data["ticket_id"] == ticket_id


def test_client_reconnects_after_close(db_path):
    mcp = mcp_client.MCPToolClient(db_path=str(db_path)).start()
    try:
        mcp.close()
        result = mcp.execute("get_status_ticket", {"ticket_id": "TCK-00000000"})
        assert not result.is_error and mcp.connected
    finally:
        mcp.close()


def test_server_errors_are_reported_not_raised(tmp_path):
    broken = tmp_path / "broken_server.py"
    broken.write_text("raise SystemExit(1)\n", encoding="utf-8")
    mcp = mcp_client.MCPToolClient(db_path=str(tmp_path / "t.db"), timeout=5, server_script=broken)
    with pytest.raises(mcp_client.MCPClientError):
        mcp.start()
    result = mcp.execute("get_status_ticket", {"ticket_id": "TCK-00000000"})
    assert result.is_error and "unavailable" in result.output
    assert not mcp.connected


# ---------------------------------------------------------------------------
# Mock mode: simulated tool choice, real MCP request
# ---------------------------------------------------------------------------


def test_mock_tool_choice_rules():
    [(name, args)] = mcp_client.choose_mock_ticket_calls(
        "Please create an urgent ticket: Wi-Fi is down on the test floor")
    assert name == "create_ticket" and args["priority"] == "critical"
    assert args["title"] == "Wi-Fi is down on the test floor"
    assert mcp_client.choose_mock_ticket_calls("What is the status of tck-1a2b3c4d?") == [
        ("get_status_ticket", {"ticket_id": "tck-1a2b3c4d"})]
    assert mcp_client.choose_mock_ticket_calls("My mouse is broken") == []


def test_mock_mode_creates_ticket_through_mcp_and_reruns_do_not_duplicate(client):
    messages = [{"role": "user", "id": "turn-1",
                 "content": "Open a high priority ticket: the test monitor flickers constantly"}]
    response = app_implementation.answer_question(messages, make_config(), mcp=client)

    [event] = response.tool_events
    assert event["kind"] == "mcp" and event["name"] == "create_ticket" and not event["is_error"]
    [ticket] = response.tickets
    assert ticket["priority"] == "high" and ticket["status"] == "open"
    assert ticket["ticket_id"] in response.answer and response.simulated_answer
    # The ticket really exists in the server's database (fetched over MCP).
    assert client.execute("get_status_ticket", {"ticket_id": ticket["ticket_id"]}).data["found"]

    rerun = app_implementation.answer_question(messages, make_config(), mcp=client)
    assert rerun.tickets[0]["ticket_id"] == ticket["ticket_id"] and rerun.tickets[0]["duplicate"]


def test_mock_mode_ticket_status_and_missing_ticket(client):
    ticket_id = create(client).data["ticket_id"]
    response = app_implementation.answer_question(
        [{"role": "user", "content": f"What is the status of {ticket_id}?"}], make_config(), mcp=client)
    assert response.tickets[0]["found"] and "open" in response.answer

    missing = app_implementation.answer_question(
        [{"role": "user", "content": "Status of TCK-FFFFFFFF please"}], make_config(), mcp=client)
    assert "No ticket found" in missing.answer and not missing.tool_events[0]["is_error"]


def test_mock_mode_invalid_ticket_input_is_shown_as_error(client):
    response = app_implementation.answer_question(
        [{"role": "user", "content": "create ticket: hi"}], make_config(), mcp=client)
    assert response.tool_events[0]["is_error"] and response.tickets == []
    assert "error" in response.answer


def test_mock_mode_without_mcp_client_or_when_disabled(client):
    question = [{"role": "user", "content": "Open a ticket: the test keyboard is missing keys"}]
    assert app_implementation.answer_question(question, make_config(), mcp=None).tool_events == []
    disabled = app_implementation.answer_question(question, make_config(mcp_enabled=False), mcp=client)
    assert disabled.tool_events == []


# ---------------------------------------------------------------------------
# RAG warning suppression
# ---------------------------------------------------------------------------


class EmptyKnowledgeBase:
    simulated = True

    def search(self, *args):
        return []


@pytest.mark.parametrize("question, warned", [
    ("What time is it?", False),
    ("Open a ticket: the test projector has no signal", False),
    ("My test printer is jammed", True),
    ("What time is it? Also any news about the latest outage?", True),  # web search: keep it
])
def test_no_results_warning_only_when_relevant(client, question, warned):
    response = app_implementation.answer_question(
        [{"role": "user", "content": question}], make_config(rag_enabled=True),
        knowledge_base=EmptyKnowledgeBase(), mcp=client)
    assert (response.notice is not None) is warned


# ---------------------------------------------------------------------------
# Live mode: Claude (fake) selects the tool, the MCP server executes it
# ---------------------------------------------------------------------------


class FakeClaude:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, **request):
        self.requests.append({**request, "messages": list(request["messages"])})
        return self.responses.pop(0)


def test_live_tool_loop_executes_mcp_tools(monkeypatch, client):
    tool_input = {"title": "Email sync fails", "priority": "medium",
                  "description": "The test mailbox stopped syncing after a password change."}
    fake = FakeClaude(
        NS(stop_reason="tool_use", content=[
            NS(type="tool_use", id="toolu_1", name="create_ticket", input=tool_input)]),
        NS(stop_reason="end_turn", content=[NS(type="text", text="Your ticket was created.", citations=None)]),
    )
    monkeypatch.setattr(app_implementation, "make_create_message", lambda cfg: fake)

    response = app_implementation.answer_question(
        [{"role": "user", "content": "Please log a ticket for my email", "id": "live-1"}],
        make_config(mock_mode=False), mcp=client)

    first = fake.requests[0]
    assert [t["name"] for t in first["tools"]] == [
        "get_current_datetime", "create_ticket", "get_status_ticket", "web_search"]
    assert "create_ticket" in first["system"]
    tool_result = fake.requests[1]["messages"][-1]["content"][0]
    created = json.loads(tool_result["content"])
    assert tool_result["tool_use_id"] == "toolu_1" and created["status"] == "open"
    assert response.tickets[0]["ticket_id"] == created["ticket_id"]
    assert response.tool_events[0]["kind"] == "mcp"


def test_live_tool_loop_reports_mcp_errors_to_claude(monkeypatch, client):
    fake = FakeClaude(
        NS(stop_reason="tool_use", content=[
            NS(type="tool_use", id="toolu_1", name="get_status_ticket", input={"ticket_id": "nope"})]),
        NS(stop_reason="end_turn", content=[NS(type="text", text="That ID looks wrong.", citations=None)]),
    )
    monkeypatch.setattr(app_implementation, "make_create_message", lambda cfg: fake)
    response = app_implementation.answer_question(
        [{"role": "user", "content": "status of nope"}], make_config(mock_mode=False), mcp=client)
    tool_result = fake.requests[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert response.answer == "That ID looks wrong." and response.tickets == []


def test_tool_definitions_without_mcp_are_unchanged():
    assert [t["name"] for t in tools.build_tools(TOOLS_CONFIG)] == ["get_current_datetime", "web_search"]
