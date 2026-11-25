"""
IT Support Chatbot - Streamlit user interface (RAG + tools + MCP tickets).

The UI only handles display and session state; all LLM, retrieval, tool and
MCP logic lives in app_implementation.py, rag.py, tools.py and mcp_client.py.

Run:
    streamlit run app.py
"""

import uuid

import streamlit as st

from app_implementation import (
    ChatError,
    answer_question,
    init_knowledge_base,
    init_mcp_client,
    load_config,
    mode_label,
    rag_label,
    tools_label,
    validate_config,
)

WELCOME_MESSAGE = (
    "Hi! I'm your IT support assistant. Ask me about passwords, Wi-Fi, VPN, "
    "printers, email, software installs, or any other technical issue."
)

st.set_page_config(page_title="IT Support Chatbot", page_icon="🛠️", layout="centered")


# ---------------------------------------------------------------------------
# Cached resources and session state
# ---------------------------------------------------------------------------


@st.cache_resource
def get_config() -> dict:
    """Load configuration once per server process."""
    return load_config()


@st.cache_resource(show_spinner="Indexing the knowledge base...")
def get_knowledge_base():
    """
    Open and index the knowledge base once per server process.

    Returns (knowledge_base, stats, error_message). On error the chat still
    works, but answers are not grounded in the documentation.
    """
    config = get_config()
    if not config["rag"]["enabled"]:
        return None, None, None
    try:
        knowledge_base, stats = init_knowledge_base(config)
    except ChatError as error:
        return None, None, str(error)
    return knowledge_base, stats, None


@st.cache_resource(show_spinner="Starting the MCP ticket server...")
def get_mcp_client():
    """
    Start the MCP ticket server once per server process (shared by all sessions).

    Returns (client, error_message). The subprocess is stopped when Streamlit
    exits (atexit) or when "Reconnect" is clicked.
    """
    return init_mcp_client(get_config())


def reconnect_mcp() -> None:
    """Stop the MCP server subprocess and start a new one on the next run."""
    client, _ = get_mcp_client()
    if client is not None:
        client.close()
    get_mcp_client.clear()


def reload_knowledge_base() -> None:
    """Re-check the knowledge base file (only changed chunks are re-embedded)."""
    get_knowledge_base.clear()


def init_session() -> None:
    """Create the conversation history for this browser session."""
    if "messages" not in st.session_state:
        # Only real user/assistant turns are stored; the welcome text is UI-only.
        # Assistant turns also keep their "sources" and "notice" for display.
        st.session_state.messages = []
    if "tickets" not in st.session_state:
        # Tickets created or looked up in this session, keyed by ticket ID (no duplicates).
        st.session_state.tickets = {}


def clear_conversation() -> None:
    """Reset the conversation history (tickets stay in the database)."""
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def render_rag_status(config: dict, knowledge_base, stats, kb_error) -> None:
    """Sidebar section describing the knowledge base."""
    rag_config = config["rag"]
    st.subheader("📚 Knowledge base (RAG)")

    if not rag_config["enabled"]:
        st.info("RAG is disabled (`RAG_ENABLED=false`). Answers use general knowledge only.")
        return
    if kb_error:
        st.error(f"Knowledge base unavailable: {kb_error}")
    elif knowledge_base is not None:
        if knowledge_base.simulated:
            st.warning("🧪 **Simulated embeddings** – local mock vectors, no Voyage API calls.")
        else:
            st.success("🟢 **Voyage AI embeddings** – live retrieval.")
        st.markdown(f"- **Chunks indexed:** `{stats.total}`")
        st.markdown(f"- **Collection:** `{knowledge_base.collection_name}`")

    model = "mock-hash" if rag_config["mock_embeddings"] else rag_config["embedding_model"]
    st.markdown(f"- **Embedding model:** `{model}`")
    st.markdown(f"- **Source file:** `{rag_config['kb_path']}`")
    st.markdown(
        f"- **Top-k / min similarity:** `{rag_config['top_k']}` / `{rag_config['min_similarity']}`"
    )
    st.button("🔄 Reload knowledge base", on_click=reload_knowledge_base, width="stretch")


def render_tools_status(config: dict) -> None:
    """Sidebar section describing the tools."""
    tools_config = config["tools"]
    st.subheader("🧰 Tools")
    if not tools_config["enabled"]:
        st.info("Tools are disabled (`TOOLS_ENABLED=false`).")
        return
    st.markdown("- **Client tool:** `get_current_datetime` (runs locally)")
    if tools_config["web_search"]:
        search = "simulated, no network" if config["mock_mode"] else tools_config["web_search_version"]
        st.markdown(f"- **Server tool:** `web_search` ({search})")
        st.markdown(f"- **Max searches per turn:** `{tools_config['web_search_max_uses']}`")
    else:
        st.markdown("- **Server tool:** web search off (`WEB_SEARCH_ENABLED=false`)")
    st.markdown(f"- **Max tool iterations:** `{tools_config['max_iterations']}`")


def render_mcp_status(config: dict, client, mcp_error) -> None:
    """Sidebar section describing the MCP ticket server and this session's tickets."""
    st.subheader("🎫 Tickets (MCP)")
    if not (config["tools"]["enabled"] and config["mcp"]["enabled"]):
        st.info("Ticket tools are disabled (`MCP_ENABLED=false` or `TOOLS_ENABLED=false`).")
        return
    if client is not None and client.connected:
        st.success("🟢 **MCP server connected** (stdio, local SQLite).")
    else:
        reason = (client.last_error if client is not None else None) or mcp_error or "not connected"
        st.error(f"MCP server unavailable: {reason} It will retry on the next ticket request.")
    if client is not None and client.tool_names:
        st.markdown("- **Tools discovered:** " + ", ".join(f"`{name}`" for name in client.tool_names))
    st.markdown(f"- **Database:** `{config['mcp']['db_path']}`")
    if config["mock_mode"]:
        st.caption("🧪 Mock mode: tool choice is simulated, but tickets are real MCP requests.")
    st.button("🔁 Reconnect MCP server", on_click=reconnect_mcp, width="stretch")

    tickets = st.session_state.tickets
    if tickets:
        st.markdown(f"**Tickets in this session ({len(tickets)})**")
        for ticket in tickets.values():
            st.markdown(f"- `{ticket['ticket_id']}` · {ticket['priority']} · **{ticket['status']}**  \n"
                        f"  {ticket['title']}")


def render_sidebar(config: dict, knowledge_base, stats, kb_error, client=None, mcp_error=None) -> None:
    """Sidebar with mode, model information, RAG status and controls."""
    with st.sidebar:
        st.header("⚙️ Settings")

        if config["mock_mode"]:
            st.warning("🧪 **Mock mode is ON** – answers are simulated, no API calls.")
        else:
            st.success("🟢 **Live mode** – answers come from Claude.")

        st.subheader("Model")
        st.markdown(f"- **Model:** `{'mock' if config['mock_mode'] else config['model']}`")
        st.markdown(f"- **Max tokens:** `{config['max_tokens']}`")
        st.markdown(
            f"- **Refusal fallback:** `{'on' if config['refusal_fallback'] else 'off'}`"
        )
        st.markdown(f"- **Messages in chat:** `{len(st.session_state.messages)}`")

        st.divider()
        render_rag_status(config, knowledge_base, stats, kb_error)

        st.divider()
        render_tools_status(config)

        st.divider()
        render_mcp_status(config, client, mcp_error)

        st.divider()
        st.button(
            "🗑️ Clear conversation",
            on_click=clear_conversation,
            width="stretch",
            disabled=not st.session_state.messages,
        )


def render_header(config: dict) -> None:
    """Page title with visible mode badges."""
    st.title("🛠️ IT Support Chatbot")
    badge = "🧪 MOCK MODE" if config["mock_mode"] else "🟢 LIVE"
    st.caption(
        f"{badge} · Powered by Claude · {mode_label(config)} · {rag_label(config)} · "
        f"{tools_label(config)}"
    )


def render_sources(message: dict) -> None:
    """Show the retrieval notice and the knowledge base sources of an assistant turn."""
    if message.get("notice"):
        st.warning(f"⚠️ {message['notice']}")

    sources = message.get("sources") or []
    if not sources:
        return
    label = f"📚 Sources ({len(sources)})"
    if message.get("simulated_retrieval"):
        label += " · 🧪 simulated embeddings"
    with st.expander(label):
        for number, source in enumerate(sources, start=1):
            st.markdown(
                f"**{number}. {source['section']}** · `{source['source']}` · "
                f"similarity `{source['similarity']:.2f}`"
            )
            excerpt = source["text"].split("\n", 1)[-1]
            st.caption(excerpt[:400] + ("..." if len(excerpt) > 400 else ""))


def _link(title: str, url) -> str:
    """Markdown link, or plain text when there is no URL (simulated results)."""
    title = (title or "Untitled").replace("[", "(").replace("]", ")")
    return f"[{title}]({url})" if url else title


def render_tools(message: dict) -> None:
    """Show the tools used in an assistant turn, web results and citations."""
    events = message.get("tool_events") or []
    if not events:
        return
    label = f"🧰 Tools used ({len(events)})"
    if message.get("simulated_tools"):
        label += " · 🧪 simulated web search"
    with st.expander(label):
        for event in events:
            icon = {"server": "🌐", "mcp": "🎫"}.get(event["kind"], "🕒")
            where = {"server": "server tool", "mcp": "MCP tool"}.get(event["kind"], "client tool")
            tags = " · 🧪 simulated" if event.get("simulated") else ""
            tags += " · ⚠️ error" if event.get("is_error") else ""
            st.markdown(f"{icon} **{event['name']}** ({where}){tags}")
            if event["input"]:
                st.caption(f"Input: `{event['input']}`")
            st.code(event["output"], language="json" if event["output"].startswith("{") else None)

    web_results = message.get("web_results") or []
    if web_results:
        with st.expander(f"🌐 Web search results ({len(web_results)})"):
            for result in web_results:
                age = f" · {result['page_age']}" if result.get("page_age") else ""
                st.markdown(f"- {_link(result['title'], result.get('url'))}{age}")
            if any(result.get("simulated") for result in web_results):
                st.caption("🧪 Simulated results: no network request was made and there are no real sources.")

    citations = message.get("citations") or []
    if citations:
        with st.expander(f"🔗 Citations ({len(citations)})"):
            for number, citation in enumerate(citations, start=1):
                st.markdown(f"**{number}.** {_link(citation['title'], citation.get('url'))}")
                if citation.get("cited_text"):
                    st.caption(f"“{citation['cited_text'][:300]}”")


def render_tickets(message: dict) -> None:
    """Show the tickets created or looked up in an assistant turn."""
    for ticket in message.get("tickets") or []:
        if ticket["action"] == "create_ticket":
            heading = "Ticket already created" if ticket.get("duplicate") else "Ticket created"
            box = st.info if ticket.get("duplicate") else st.success
        elif ticket.get("found"):
            heading, box = "Ticket status", st.info
        else:
            st.warning(f"🎫 {ticket.get('message', 'Ticket not found.')}")
            continue
        box(
            f"🎫 **{heading}: `{ticket['ticket_id']}`**  \n"
            f"**{ticket['title']}**  \n"
            f"Status: **{ticket['status']}** · Priority: **{ticket['priority']}** · "
            f"Created: {ticket['created_at']}"
        )


def remember_tickets(tickets: list) -> None:
    """Add found/created tickets to the sidebar list (keyed by ID, so reruns never duplicate)."""
    for ticket in tickets:
        if ticket.get("found", True):
            st.session_state.tickets[ticket["ticket_id"]] = ticket


def render_history() -> None:
    """Show the welcome message and all previous turns."""
    with st.chat_message("assistant", avatar="🛠️"):
        st.markdown(WELCOME_MESSAGE)
    for message in st.session_state.messages:
        avatar = "🛠️" if message["role"] == "assistant" else "🧑‍💻"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_tickets(message)
                render_sources(message)
                render_tools(message)


def handle_user_input(config: dict, prompt: str, knowledge_base, client=None) -> bool:
    """Append the user's message, get a reply, and display it. Returns True on success."""
    # The ID makes ticket creation idempotent: if this message is ever processed
    # again (rerun, retry), the MCP server returns the existing ticket.
    st.session_state.messages.append({"role": "user", "content": prompt, "id": uuid.uuid4().hex})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🛠️"):
        try:
            with st.spinner("Searching the knowledge base, using tools and thinking..."):
                response = answer_question(st.session_state.messages, config, knowledge_base, mcp=client)
        except ChatError as error:
            # Drop the failed user turn so history stays valid (user/assistant alternation).
            st.session_state.messages.pop()
            st.error(f"⚠️ {error}")
            st.info("Your message was not saved. Fix the issue above and try again.")
            return False

        message = {
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
            "notice": response.notice,
            "simulated_retrieval": response.simulated_retrieval,
            "tool_events": response.tool_events,
            "web_results": response.web_results,
            "citations": response.citations,
            "simulated_tools": response.simulated_tools,
            "tickets": response.tickets,
        }
        st.markdown(message["content"])
        render_tickets(message)
        render_sources(message)
        render_tools(message)
    st.session_state.messages.append(message)
    remember_tickets(response.tickets)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    config = get_config()
    init_session()

    render_header(config)

    for problem in validate_config(config):
        st.error(f"⚠️ Configuration problem: {problem}")

    knowledge_base, stats, kb_error = get_knowledge_base()
    client, mcp_error = get_mcp_client()

    render_history()
    render_sidebar(config, knowledge_base, stats, kb_error, client, mcp_error)

    prompt = st.chat_input("Describe your IT issue or ask to open a ticket...")
    if prompt and prompt.strip():
        if handle_user_input(config, prompt.strip(), knowledge_base, client):
            # Refresh the sidebar counter and clear button (skipped on error so it stays visible).
            st.rerun()


main()
