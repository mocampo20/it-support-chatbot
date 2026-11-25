"""
Automated tests for the tools: get_current_datetime, the tool_use/tool_result
loop, web search parsing, the iteration limit and mock mode.

No paid API calls are made: Claude responses are fake objects.

Run:
    python -m pytest
"""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

import app_implementation
import tools

TOOLS_CONFIG = {
    "enabled": True, "web_search": True, "web_search_max_uses": 3,
    "web_search_version": "web_search_20260209", "max_iterations": 5,
}


def text(value, citations=None):
    return NS(type="text", text=value, citations=citations)


def tool_use(block_id="toolu_1", name="get_current_datetime", tool_input=None):
    return NS(type="tool_use", id=block_id, name=name, input=tool_input or {})


def response(stop_reason, *content):
    return NS(stop_reason=stop_reason, content=list(content))


class FakeClaude:
    """Returns the given responses in order and records every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, **request):
        self.requests.append({**request, "messages": list(request["messages"])})
        return self.responses.pop(0)


def base_request():
    return {"model": "test", "max_tokens": 100, "system": "sys",
            "messages": [{"role": "user", "content": "What time is it?"}],
            "tools": tools.build_tools(TOOLS_CONFIG)}


def config(**tools_overrides):
    return {"mock_mode": True, "model": "test", "max_tokens": 100, "refusal_fallback": False,
            "rag": {"enabled": False}, "tools": {**TOOLS_CONFIG, **tools_overrides}}


# ---------------------------------------------------------------------------
# Client tool
# ---------------------------------------------------------------------------


def test_get_current_datetime_returns_iso_8601_with_timezone():
    fixed = datetime(2026, 3, 1, 14, 30, 5, tzinfo=timezone(timedelta(hours=-5)))
    data = tools.get_current_datetime(fixed)
    assert data["datetime"] == "2026-03-01T14:30:05-05:00"
    assert data["date"] == "2026-03-01" and data["time"] == "14:30:05"
    assert data["utc_offset"] == "-05:00" and data["timezone"]


def test_get_current_datetime_uses_the_real_clock():
    data = tools.get_current_datetime()
    parsed = datetime.fromisoformat(data["datetime"])
    assert parsed.tzinfo is not None
    assert abs((datetime.now(timezone.utc) - parsed).total_seconds()) < 60


def test_unknown_client_tool_is_an_error_not_an_exception():
    output, is_error = tools.run_client_tool("delete_everything", {})
    assert is_error and "Unknown tool" in output


def test_tool_definitions():
    defined = tools.build_tools(TOOLS_CONFIG)
    assert [t["name"] for t in defined] == ["get_current_datetime", "web_search"]
    assert defined[1] == {"type": "web_search_20260209", "name": "web_search", "max_uses": 3}
    assert [t["name"] for t in tools.build_tools({**TOOLS_CONFIG, "web_search": False})] == [
        "get_current_datetime"]
    assert tools.build_tools({**TOOLS_CONFIG, "enabled": False}) == []


# ---------------------------------------------------------------------------
# Tool loop
# ---------------------------------------------------------------------------


def test_tool_loop_runs_client_tool_and_sends_tool_result():
    fake = FakeClaude(
        response("tool_use", tool_use()),
        response("end_turn", text("It is 14:30.")),
    )
    request = base_request()
    result = tools.run_tool_loop(fake, request, max_iterations=5)

    assert result.answer == "It is 14:30." and result.iterations == 2
    second = fake.requests[1]["messages"]
    assert second[1]["role"] == "assistant"
    tool_result = second[2]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "toolu_1"
    assert "datetime" in json.loads(tool_result["content"])
    assert result.tool_events[0]["name"] == "get_current_datetime"
    assert len(request["messages"]) == 1  # caller's list is not modified


def test_parallel_tool_calls_return_all_results_in_one_message():
    fake = FakeClaude(
        response("tool_use", tool_use("a"), tool_use("b", name="unknown_tool")),
        response("end_turn", text("Done.")),
    )
    tools.run_tool_loop(fake, base_request(), max_iterations=5)
    results = fake.requests[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert results[1]["is_error"] is True


def test_tool_loop_stops_at_max_iterations():
    fake = FakeClaude(*[response("tool_use", tool_use(f"t{i}")) for i in range(3)])
    result = tools.run_tool_loop(fake, base_request(), max_iterations=3)
    assert result.iterations == 3 and len(fake.requests) == 3
    assert "MAX_TOOL_ITERATIONS" in result.notice


def test_pause_turn_is_resumed_without_extra_user_message():
    fake = FakeClaude(
        response("pause_turn", NS(type="server_tool_use", id="srv_1", input={"query": "x"})),
        response("end_turn", text("Final.")),
    )
    result = tools.run_tool_loop(fake, base_request(), max_iterations=5)
    assert result.answer == "Final."
    assert fake.requests[1]["messages"][-1]["role"] == "assistant"


def test_web_search_results_and_citations_are_collected():
    cite = NS(type="web_search_result_location", url="https://example.com/a",
              title="Page A", cited_text="Version 2 was released.", encrypted_index="x")
    fake = FakeClaude(response(
        "end_turn",
        NS(type="server_tool_use", id="srv_1", name="web_search", input={"query": "latest version"}),
        NS(type="web_search_tool_result", tool_use_id="srv_1", content=[
            NS(type="web_search_result", title="Page A", url="https://example.com/a",
               page_age="2 days ago", encrypted_content="..."),
        ]),
        text("The latest version is 2", citations=[cite]),
        text("."),
    ))
    result = tools.run_tool_loop(fake, base_request(), max_iterations=5)

    assert result.answer == "The latest version is 2."
    assert result.web_results == [{"title": "Page A", "url": "https://example.com/a",
                                   "page_age": "2 days ago", "simulated": False}]
    assert result.citations == [{"title": "Page A", "url": "https://example.com/a",
                                 "cited_text": "Version 2 was released."}]
    assert result.tool_events[0]["input"] == {"query": "latest version"}


def test_web_search_error_is_reported_not_raised():
    fake = FakeClaude(response(
        "end_turn",
        NS(type="server_tool_use", id="srv_1", input={"query": "q"}),
        NS(type="web_search_tool_result", tool_use_id="srv_1",
           content=NS(type="web_search_tool_result_error", error_code="max_uses_exceeded")),
        text("Answer without search."),
    ))
    result = tools.run_tool_loop(fake, base_request(), max_iterations=5)
    assert result.tool_events[0]["is_error"] and "max_uses_exceeded" in result.tool_events[0]["output"]
    assert result.web_results == []


def test_refusal_stops_the_loop():
    result = tools.run_tool_loop(FakeClaude(response("refusal")), base_request(), max_iterations=5)
    assert "declined" in result.answer


# ---------------------------------------------------------------------------
# Mock mode and app integration
# ---------------------------------------------------------------------------


def test_mock_mode_runs_real_datetime_and_labels_simulated_search():
    response_ = app_implementation.answer_question(
        [{"role": "user", "content": "What time is it, and is there news about the latest outage?"}],
        config(),
    )
    names = [e["name"] for e in response_.tool_events]
    assert names == ["get_current_datetime", "web_search"]
    assert "datetime" in json.loads(response_.tool_events[0]["output"])
    assert response_.simulated_tools
    assert all(r["simulated"] and r["url"] is None for r in response_.web_results)
    assert response_.citations == []  # no invented citations
    assert "Simulated" in response_.answer and "get_current_datetime" in response_.answer


def test_mock_mode_without_tool_keywords_uses_no_tools():
    response_ = app_implementation.answer_question(
        [{"role": "user", "content": "My printer is jammed"}], config())
    assert response_.tool_events == [] and not response_.simulated_tools


def test_mock_mode_respects_disabled_web_search():
    response_ = app_implementation.answer_question(
        [{"role": "user", "content": "latest news"}], config(web_search=False))
    assert response_.web_results == []


def test_live_mode_uses_tool_loop(monkeypatch):
    fake = FakeClaude(response("tool_use", tool_use()), response("end_turn", text("It is noon.")))
    monkeypatch.setattr(app_implementation, "make_create_message", lambda cfg: fake)
    cfg = config()
    cfg["mock_mode"] = False
    history = [{"role": "user", "content": "What time is it?"}]

    response_ = app_implementation.answer_question(history, cfg)

    assert response_.answer == "It is noon." and not response_.simulated_answer
    first = fake.requests[0]
    assert [t["name"] for t in first["tools"]] == ["get_current_datetime", "web_search"]
    assert "get_current_datetime" in first["system"]
    assert history == [{"role": "user", "content": "What time is it?"}]


@pytest.mark.parametrize("error", [RuntimeError("ANTHROPIC_API_KEY is not set.")])
def test_live_mode_errors_become_chat_errors(monkeypatch, error):
    def fail(cfg):
        raise error
    monkeypatch.setattr(app_implementation, "make_create_message", fail)
    cfg = config()
    cfg["mock_mode"] = False
    with pytest.raises(app_implementation.ChatError):
        app_implementation.answer_question([{"role": "user", "content": "hi"}], cfg)
