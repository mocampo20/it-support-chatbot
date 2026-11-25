"""
IT Support Chatbot - backend logic for the Streamlit app.

This module has no Streamlit code, so it can be tested on its own. It reuses
the configuration and Claude API logic from the console chatbot (chatbot.py)
and adds what a multi-turn RAG chat needs:
    - a history-aware mock reply,
    - knowledge base retrieval (rag.py) injected into the Claude prompt,
    - tools (tools.py): Anthropic web search (server tool) and
      get_current_datetime (client tool) with a tool_use/tool_result loop,
    - MCP ticket tools (mcp_client.py -> mcp_server.py): create_ticket and
      get_status_ticket, discovered from the MCP server and executed through
      it in both live and mock mode,
    - a single `answer_question()` entry point used by the UI (returns the
      answer plus the retrieved sources),
    - friendly error messages returned as text instead of printed.
"""

import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import mcp_client
import rag
import tools
from chatbot import SYSTEM_PROMPT, ask_claude, make_create_message
from chatbot import load_config as load_base_config

__all__ = [
    "ChatError",
    "ChatResponse",
    "answer_question",
    "get_reply",
    "init_knowledge_base",
    "init_mcp_client",
    "load_config",
    "mode_label",
    "rag_label",
    "tools_label",
    "validate_config",
]


class ChatError(Exception):
    """An error with a message that is safe and friendly to show in the UI."""


@dataclass
class ChatResponse:
    """Everything the UI needs to display one assistant turn."""

    answer: str
    sources: List[dict] = field(default_factory=list)  # RetrievedChunk.to_dict() items
    simulated_answer: bool = False  # True when the answer comes from the mock, not Claude
    simulated_retrieval: bool = False  # True when sources come from mock embeddings
    notice: Optional[str] = None  # warning about retrieval (failed, no results...)
    tool_events: List[dict] = field(default_factory=list)  # tools.ToolEvent dicts
    web_results: List[dict] = field(default_factory=list)  # web search results
    citations: List[dict] = field(default_factory=list)  # web citations in the answer
    simulated_tools: bool = False  # True when web search was simulated (mock mode)
    tickets: List[dict] = field(default_factory=list)  # tickets created or looked up this turn


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def load_config() -> dict:
    """Load the chatbot configuration plus RAG ("rag"), tool ("tools") and MCP ("mcp") settings."""
    config = load_base_config()
    config["rag"] = rag.load_rag_config(mock_mode=config["mock_mode"])
    config["tools"] = tools.load_tools_config()
    config["mcp"] = mcp_client.load_mcp_config()
    return config


def mcp_enabled(config: dict) -> bool:
    """MCP ticket tools are used only when tools and MCP are both enabled."""
    return bool(config.get("tools", {}).get("enabled") and config.get("mcp", {}).get("enabled"))


def validate_config(config: dict) -> list:
    """Return a list of human-readable configuration problems (empty if OK)."""
    problems = []
    if not config["mock_mode"] and not os.getenv("ANTHROPIC_API_KEY"):
        problems.append(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file, "
            "or set MOCK_MODE=true to test without the API."
        )
    rag_config = config.get("rag", {})
    if rag_config.get("enabled") and not rag_config.get("mock_embeddings"):
        key = os.getenv("VOYAGE_API_KEY", "")
        if not key.strip() or key.startswith("your-"):
            problems.append(
                "VOYAGE_API_KEY is not set, so the knowledge base cannot be searched. "
                "Add it to your .env file, or set MOCK_EMBEDDINGS=true."
            )
    return problems


def mode_label(config: dict) -> str:
    """Short label describing the active mode."""
    return "Mock mode" if config["mock_mode"] else "Live mode"


def rag_label(config: dict) -> str:
    """Short label describing the retrieval mode."""
    rag_config = config.get("rag", {})
    if not rag_config.get("enabled"):
        return "RAG off"
    if rag_config.get("mock_embeddings"):
        return "RAG: mock embeddings"
    return f"RAG: Voyage {rag_config['embedding_model']}"


def tools_label(config: dict) -> str:
    """Short label describing the active tools."""
    tools_config = config.get("tools", {})
    if not tools_config.get("enabled"):
        return "Tools off"
    names = ["datetime"]
    if mcp_enabled(config):
        names.append("tickets (MCP)")
    if tools_config.get("web_search"):
        names.append("web search (simulated)" if config["mock_mode"] else "web search")
    return "Tools: " + ", ".join(names)


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------


def init_knowledge_base(config: dict):
    """
    Open the vector store and index the knowledge base (only new chunks are embedded).

    Returns (knowledge_base, IndexStats). Raises ChatError with a friendly message.
    """
    try:
        knowledge_base = rag.create_knowledge_base(config["rag"])
        stats = knowledge_base.ensure_index()
    except rag.RAGError as error:
        raise ChatError(str(error)) from error
    return knowledge_base, stats


def retrieve_sources(question: str, config: dict, knowledge_base):
    """
    Search the knowledge base for the question.

    Returns (chunks, notice, searched). `searched` is False when retrieval could
    not run; in that case the answer is not grounded and `notice` explains why.
    A failed live search is reported, never replaced with mock results.
    """
    rag_config = config["rag"]
    if knowledge_base is None:
        return [], "Knowledge base unavailable - this answer is not based on the IT documentation.", False
    try:
        chunks = knowledge_base.search(question, rag_config["top_k"], rag_config["min_similarity"])
    except rag.RAGError as error:
        return [], f"Knowledge base search failed: {error} This answer is not based on the IT documentation.", False
    if not chunks:
        return [], "No relevant knowledge base articles were found for this question.", True
    return chunks, None, True


# ---------------------------------------------------------------------------
# MCP ticket tools
# ---------------------------------------------------------------------------

# Tools whose requests never need the IT documentation: a knowledge base
# warning is irrelevant when a turn only used these.
NON_RAG_TOOLS = {"get_current_datetime", *mcp_client.TICKET_TOOLS}


def init_mcp_client(config: dict):
    """
    Start the MCP ticket server and connect to it (same server in mock and live mode).

    Returns (client, error_message). On error the chat still works without tickets;
    the client is still returned so that later calls can retry the connection.
    """
    if not mcp_enabled(config):
        return None, None
    client = mcp_client.MCPToolClient(db_path=config["mcp"]["db_path"], timeout=config["mcp"]["timeout"])
    try:
        client.start()
    except mcp_client.MCPClientError as error:
        return client, str(error)
    return client, None


def make_tool_executor(client, request_id: Optional[str] = None):
    """Tool executor for the tool loop: MCP tools go to the MCP server, others run locally."""

    def execute(name: str, tool_input: dict) -> tools.ToolEvent:
        if client is not None and name in mcp_client.TICKET_TOOLS:
            result = client.execute(name, tool_input, request_id=request_id)
            return tools.ToolEvent("mcp", name, dict(tool_input), result.output, result.is_error,
                                   data=result.data)
        return tools.default_tool_executor(name, tool_input)

    return execute


def run_mock_ticket_tools(question: str, client, request_id: Optional[str]) -> List[dict]:
    """Mock mode: keyword rules choose the ticket tool, the MCP server really executes it."""
    if client is None:
        return []
    execute = make_tool_executor(client, request_id)
    return [asdict(execute(name, tool_input))
            for name, tool_input in mcp_client.choose_mock_ticket_calls(question)]


def mcp_tool_schemas(client) -> list:
    """Tool definitions discovered from the MCP server (empty if it is unavailable)."""
    if client is None:
        return []
    try:
        client.start()  # reconnects if the server stopped
    except mcp_client.MCPClientError:
        return []
    return client.tool_schemas()


def tickets_from_events(events: List[dict]) -> List[dict]:
    """Ticket details returned by successful MCP calls (for the UI)."""
    return [
        {**event["data"], "action": event["name"]}
        for event in events
        if event.get("kind") == "mcp" and not event.get("is_error")
        and (event.get("data") or {}).get("ticket_id")
    ]


def only_non_rag_tools(events: List[dict]) -> bool:
    """True when the turn used tools and all of them were datetime or ticket tools."""
    return bool(events) and all(event["name"] in NON_RAG_TOOLS for event in events)


# ---------------------------------------------------------------------------
# Replies
# ---------------------------------------------------------------------------


def ask_mock(messages: list, sources: Optional[list] = None) -> str:
    """Return a canned, history-aware answer without calling the API."""
    question = messages[-1]["content"]
    turn = sum(1 for message in messages if message["role"] == "user")
    answer = (
        f"**[MOCK]** Simulated answer #{turn} to: *{question}*\n\n"
        f"The conversation so far has {len(messages)} message(s), "
        "so history is being kept. Set `MOCK_MODE=false` in `.env` to call Claude."
    )
    if sources:
        sections = ", ".join(f"*{chunk.section}*" for chunk in sources)
        excerpt = sources[0].text.split("\n", 1)[-1][:300].strip()
        answer += (
            f"\n\nRetrieved {len(sources)} knowledge base excerpt(s): {sections}.\n\n"
            f"> {excerpt}..."
        )
    return answer


def answer_question(messages: list, config: dict, knowledge_base=None, mcp=None) -> ChatResponse:
    """
    Return the assistant's reply (and its sources) for the full conversation.

    `messages` is the list of {"role": ..., "content": ...} dicts, oldest first,
    ending with the latest user message (extra keys such as "sources" are
    ignored). When RAG is enabled, the retrieved context is added to the latest
    user message only; earlier turns are sent as they were typed.
    `mcp` is an mcp_client.MCPToolClient (None = no ticket tools). The latest
    message's optional "id" makes ticket creation idempotent: the same message
    processed twice (e.g. after a Streamlit rerun) creates only one ticket.
    Raises ChatError with a friendly message if Claude cannot answer.
    """
    if not messages or messages[-1]["role"] != "user":
        raise ChatError("The conversation must end with a user message.")

    # The API only accepts role/content; drop UI-only keys.
    api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    question = api_messages[-1]["content"]
    request_id = messages[-1].get("id")
    client = mcp if mcp_enabled(config) else None

    chunks, notice, searched = [], None, False
    if config.get("rag", {}).get("enabled"):
        chunks, notice, searched = retrieve_sources(question, config, knowledge_base)

    system = SYSTEM_PROMPT
    if searched:
        api_messages[-1]["content"] = rag.augment_question(question, chunks)
        system = SYSTEM_PROMPT + rag.RAG_INSTRUCTIONS

    tools_config = config.get("tools", {})
    tool_result = tools.ToolLoopResult()
    if config["mock_mode"]:
        tool_result = tools.run_mock_tools(question, tools_config)
        tool_result.tool_events += run_mock_ticket_tools(question, client, request_id)
        answer = ask_mock(messages, chunks)
        tool_lines = [tools.describe_mock_tools(tool_result)] + [
            mcp_client.describe_ticket_event(event)
            for event in tool_result.tool_events if event["kind"] == "mcp"]
        tool_text = "\n\n".join(line for line in tool_lines if line)
        if tool_text:
            answer += "\n\n" + tool_text
    elif tools_config.get("enabled"):
        mcp_schemas = mcp_tool_schemas(client)
        if mcp_schemas:
            system += mcp_client.MCP_INSTRUCTIONS
        tool_result = call_claude_with_tools(
            api_messages, config, system + tools.TOOLS_INSTRUCTIONS,
            extra_tools=mcp_schemas, execute_tool=make_tool_executor(client, request_id))
        answer = tool_result.answer
    else:
        answer = call_claude(api_messages, config, system)

    if not chunks and only_non_rag_tools(tool_result.tool_events):
        notice = None  # e.g. "what time is it?" or "create a ticket": the docs are irrelevant
    notices = [text for text in (notice, tool_result.notice) if text]
    return ChatResponse(
        answer=answer,
        sources=[chunk.to_dict() for chunk in chunks],
        simulated_answer=config["mock_mode"],
        simulated_retrieval=bool(knowledge_base is not None and knowledge_base.simulated),
        notice=" ".join(notices) or None,
        tool_events=tool_result.tool_events,
        web_results=tool_result.web_results,
        citations=tool_result.citations,
        simulated_tools=any(event.get("simulated") for event in tool_result.tool_events),
        tickets=tickets_from_events(tool_result.tool_events),
    )


def get_reply(messages: list, config: dict, knowledge_base=None) -> str:
    """Return only the answer text (kept for backward compatibility)."""
    return answer_question(messages, config, knowledge_base).answer


def call_claude(messages: list, config: dict, system: str) -> str:
    """Call Claude (live mode, no tools) and map errors to ChatError."""
    return _with_friendly_errors(lambda: ask_claude(messages, config, system=system))


def call_claude_with_tools(messages: list, config: dict, system: str, extra_tools=None,
                           execute_tool=None) -> tools.ToolLoopResult:
    """Call Claude with the tools (plus discovered MCP tools) and run the tool loop (live mode)."""
    tools_config = config["tools"]

    def run():
        request = {
            "model": config["model"],
            "max_tokens": config["max_tokens"],
            "system": system,
            "messages": messages,
            "tools": tools.build_tools(tools_config, extra_tools),
        }
        return tools.run_tool_loop(
            make_create_message(config), request, tools_config["max_iterations"], execute_tool)

    return _with_friendly_errors(run)


def _with_friendly_errors(function):
    """Run an API call and turn any failure into a ChatError."""
    try:
        return function()
    except ImportError as error:
        raise ChatError(
            "The 'anthropic' package is not installed. Run: pip install -r requirements.txt"
        ) from error
    except RuntimeError as error:  # e.g. missing API key (raised by ask_claude)
        raise ChatError(str(error)) from error
    except Exception as error:  # noqa: BLE001 - mapped to a friendly message
        raise ChatError(friendly_api_error(error)) from error


def friendly_api_error(error: Exception) -> str:
    """Map Anthropic SDK errors to user-friendly text (most specific first)."""
    try:
        import anthropic
    except ImportError:
        return f"Unexpected error: {error}"

    if isinstance(error, anthropic.AuthenticationError):
        return "Invalid API key. Check ANTHROPIC_API_KEY in your .env file."
    if isinstance(error, anthropic.PermissionDeniedError):
        return "Your API key does not have permission to use this model or feature."
    if isinstance(error, anthropic.NotFoundError):
        return "Model not found. Check CLAUDE_MODEL in your .env file."
    if isinstance(error, anthropic.BadRequestError):
        return (
            f"Bad request: {error.message}. If the model does not support fallbacks, "
            "set ENABLE_REFUSAL_FALLBACK=false."
        )
    if isinstance(error, anthropic.RateLimitError):
        return "Rate limit reached. Please wait a moment and try again."
    if isinstance(error, anthropic.InternalServerError):
        return "The Anthropic API is temporarily unavailable. Please try again shortly."
    if isinstance(error, anthropic.APIStatusError):
        return f"The API returned status {error.status_code}: {error.message}"
    if isinstance(error, anthropic.APIConnectionError):
        return "Could not connect to the Anthropic API. Check your network connection."
    return f"Unexpected error: {error}"
