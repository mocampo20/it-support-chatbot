# IT Support Chatbot

An IT help-desk assistant built with the **Claude API**, **Streamlit** and the
**Model Context Protocol (MCP)**. It answers technical questions from a knowledge base,
looks up recent information on the web, and opens and tracks support tickets.

| Capability | Implementation |
|------------|----------------|
| Conversational UI with history | Streamlit (`st.chat_input`, `st.chat_message`, `st.session_state`) |
| Retrieval Augmented Generation (RAG) | Voyage AI embeddings + ChromaDB (persistent, cosine similarity) |
| Recent / external information | Anthropic's native **web search** server tool |
| Date and time awareness | `get_current_datetime` client tool (runs locally) |
| Ticket management | Local **MCP server** (FastMCP, stdio) with `create_ticket` and `get_status_ticket`, stored in SQLite |
| Offline development | **Mock mode**: no API keys, no network, no credits |
| Minimal console client | Single-question chatbot (`chatbot.py`) |

The knowledge base (`it_sector.txt`) is **fictional sample documentation**. It describes no
real organization, and all its URLs use the reserved `example.com` domain.

---

## Architecture

```
                         ┌──────────────────────── Streamlit (app.py) ─────────────────────────┐
                         │  chat UI · session history · sources / tools / ticket cards · sidebar │
                         └───────────────────────────────┬──────────────────────────────────────┘
                                                         │ answer_question()
                                        ┌────────────────▼────────────────┐
                                        │  app_implementation.py          │
                                        │  config · prompt building ·     │
                                        │  mock/live routing · errors     │
                                        └──┬──────────────┬────────────┬──┘
                      retrieve context     │              │ tool loop  │ ticket tools
                  ┌────────────────────────▼──┐   ┌───────▼────────┐  ┌▼───────────────────────┐
                  │ rag.py                    │   │ tools.py       │  │ mcp_client.py           │
                  │ chunking · embeddings ·   │   │ tool_use /     │  │ background asyncio loop │
                  │ ChromaDB index & search   │   │ tool_result    │  │ one persistent session  │
                  └──────┬──────────────┬─────┘   │ loop, datetime │  └───────────┬─────────────┘
                         │              │         └───────┬────────┘              │ stdio (JSON-RPC)
                 Voyage AI API     chroma_db/             │                ┌──────▼──────────────┐
                 (or mock          (local)                │                │ mcp_server.py       │
                  embedder)                       Claude Messages API      │ FastMCP + SQLite    │
                                                  (+ web search tool)      │ (tickets.db)        │
                                                                           └─────────────────────┘
```

### Request flow (live mode)

1. The user's message is added to the session history.
2. **RAG:** the question is embedded (`input_type="query"`) and the top matching knowledge base
   chunks are placed in a `<knowledge_base>` block before the latest question.
3. **Claude** receives the system prompt, the full conversation and the tool definitions:
   `get_current_datetime`, the MCP ticket tools discovered via `list_tools`, and `web_search`.
4. **Tool loop:** while `stop_reason == "tool_use"`, client tools run locally and ticket tools
   are sent to the MCP server; all results go back in one message. `pause_turn` is resumed, and
   `MAX_TOOL_ITERATIONS` caps the loop.
5. The UI shows the answer with expandable **sources**, **tools used**, **web results**,
   **citations** and **ticket cards**.

### Project structure

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI: display and session state only |
| `app_implementation.py` | Backend for the UI: configuration, RAG prompt injection, mock/live routing, friendly errors (no Streamlit code) |
| `chatbot.py` | Console chatbot and shared Claude API logic (client, refusal fallback, error mapping) |
| `rag.py` | Chunking, Voyage AI / mock embeddings, ChromaDB indexing and retrieval |
| `tools.py` | Web search definition, `get_current_datetime`, tool loop, mock tool simulation |
| `mcp_server.py` | MCP server: input validation, ticket IDs, SQLite persistence |
| `mcp_client.py` | MCP client: starts the server subprocess, discovers and calls tools, mock tool-choice rules |
| `it_sector.txt` | Fictional IT knowledge base |
| `tests/` | Automated tests (offline) |
| `requirements.txt`, `pytest.ini`, `.env.example` | Dependencies, test settings, configuration template |

Generated at runtime and excluded from Git: `.env`, `.venv/`, `chroma_db/`, `tickets.db`,
caches and logs.

---

## Mock mode vs. live mode

Set `MOCK_MODE=true` to run everything offline. The table shows what is real and what is
simulated in each mode.

| Component | Mock mode (`MOCK_MODE=true`) | Live mode (`MOCK_MODE=false`) |
|-----------|------------------------------|-------------------------------|
| Claude answer | **Simulated** template answer, labelled `[MOCK]` | Real Claude Messages API call |
| Tool choice | **Simulated** with keyword rules | Decided by Claude |
| Embeddings | **Mock**: deterministic local bag-of-words vectors (keyword quality only) | Voyage AI (`VOYAGE_EMBEDDING_MODEL`) |
| ChromaDB indexing and retrieval | Real, in a separate `it_kb_mock_*` collection | Real, in an `it_kb_voyage_<model>` collection |
| `get_current_datetime` | Real (local clock) | Real (local clock) |
| Web search | **Simulated**: results labelled `[SIMULATED]`, no network, no URLs, no citations | Real Anthropic server tool (billed per search) |
| Ticket tools | Real MCP requests to the real server and SQLite database | Real MCP requests to the real server and SQLite database |
| API keys required | None | `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY` |

`MOCK_EMBEDDINGS` defaults to the value of `MOCK_MODE`. Set `MOCK_MODE=false` and
`MOCK_EMBEDDINGS=true` to use live Claude answers with free local embeddings.

The UI always shows which mode is active: badges under the title, banners in the sidebar, and
🧪 labels on simulated sources, tools and search results.

---

## Installation

Requirements: **Python 3.10+** (developed and tested on Python 3.10 on Windows 11).

**Windows (PowerShell)**

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your own `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY`, or set `MOCK_MODE=true`
to run without keys. Never commit `.env`.

---

## Configuration

All settings are read from `.env` or from environment variables. Restart the app after
changing them.

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | – | Claude API key (not needed in mock mode) |
| `CLAUDE_MODEL` | `claude-opus-5-5` | Claude model ID |
| `CLAUDE_MAX_TOKENS` | `16000` | Max tokens per response. Thinking counts toward this limit, so low values truncate answers |
| `MOCK_MODE` | `false` | `true` = simulated answers, no API calls |
| `ENABLE_REFUSAL_FALLBACK` | `true` | If the model declines a request, the API retries it on a fallback model (beta). Set `false` for models that do not support it |
| `VOYAGE_API_KEY` | – | Voyage AI key (not needed with mock embeddings) |
| `RAG_ENABLED` | `true` | Search the knowledge base before answering |
| `VOYAGE_EMBEDDING_MODEL` | `voyage-4` | Embedding model (changing it builds a new collection) |
| `MOCK_EMBEDDINGS` | value of `MOCK_MODE` | `true` = local mock embeddings |
| `RAG_KB_PATH` | `it_sector.txt` | Knowledge base file (relative paths resolve against the project folder) |
| `CHROMA_DB_PATH` | `chroma_db` | Vector database folder (relative to the project folder) |
| `RAG_TOP_K` | `4` | Max chunks sent to Claude |
| `RAG_MIN_SIMILARITY` | `0.2` | Minimum cosine similarity for a chunk to be used |
| `RAG_MAX_CHUNK_CHARS` | `1500` | Max characters per chunk |
| `TOOLS_ENABLED` | `true` | Give Claude tools in the Streamlit app |
| `WEB_SEARCH_ENABLED` | `true` | Enable the web search server tool |
| `WEB_SEARCH_MAX_USES` | `3` | Max web searches per answer |
| `WEB_SEARCH_TOOL_VERSION` | `web_search_20260209` | Use `web_search_20250305` for older models |
| `MAX_TOOL_ITERATIONS` | `5` | Max Claude calls per tool loop |
| `MCP_ENABLED` | `true` | Start the MCP ticket server (also requires `TOOLS_ENABLED=true`) |
| `TICKET_DB_PATH` | `tickets.db` | SQLite database (relative to the project folder; keep the `.db` extension so Git ignores it) |
| `MCP_TIMEOUT_SECONDS` | `20` | Timeout for starting the MCP server and for each tool call |

---

## Running

### Streamlit app

```powershell
streamlit run app.py
```

The app opens at <http://localhost:8501>. On the first live start, the knowledge base is embedded
once. Later starts embed only new or changed chunks.

Run it offline:

```powershell
$env:MOCK_MODE="true"; streamlit run app.py        # PowerShell
```
```bash
MOCK_MODE=true streamlit run app.py                 # macOS / Linux
```

Example prompts:

| Prompt | Expected behaviour |
|--------|--------------------|
| *How do I reset my password?* | Answer grounded in the knowledge base, with a 📚 Sources expander |
| *What time is it?* | `get_current_datetime` runs, shown under 🧰 Tools used |
| *Any news about the latest VPN client release?* | Web search (simulated and labelled in mock mode) |
| *Create a high priority ticket: VPN disconnects every 10 minutes* | Ticket created through MCP, 🎫 card shown and listed in the sidebar |
| *What is the status of TCK-…?* | Ticket status read from SQLite through MCP |
| *What is the status of TCK-00000000?* | "No ticket found" message (handled gracefully) |
| *Create a ticket: hi* | Server-side validation error, shown as a tool error |

### Console chatbot

A minimal single-turn client. It does not use RAG or tools.

```powershell
python chatbot.py
$env:MOCK_MODE="true"; python chatbot.py           # offline
```

### Retrieval check from the console

```powershell
$env:MOCK_MODE="true"; python rag.py "My VPN keeps disconnecting"
```

---

## Testing

```powershell
python -m pytest
```

The test suite is **fully offline** and makes no paid API calls:

- `tests/test_rag.py`: chunking, incremental indexing (no duplicates), retrieval, prompt
  building, error handling. Uses mock embeddings and a fake Voyage client.
- `tests/test_tools.py`: `get_current_datetime`, the tool loop (parallel calls, errors,
  `pause_turn`, iteration limit), web search result and citation parsing, mock mode. Uses fake
  Claude responses.
- `tests/test_mcp.py`: SQLite store and validation. It also starts the **real MCP server
  subprocess over stdio** with a temporary database to test tool discovery, ticket
  create/status, error results, idempotency, persistence across restarts, reconnection, and
  the mock and live tool loops (with fake Claude responses).

---

## Security and privacy

- API keys are read only from the environment or `.env`. They are never printed, logged or sent to
  the MCP server: the server subprocess gets a minimal default environment.
- `.gitignore` excludes `.env` / `.env.*` (except `.env.example`), virtual environments, caches,
  SQLite databases and journals, `chroma_db/`, Streamlit secrets and logs. `.env.example`
  contains placeholders only.
- The MCP server uses the stdio transport only and opens no network port. All inputs are validated and
  stored with parameterized SQL queries. Ticket IDs come from a cryptographically secure RNG.
- An internal idempotency key, hidden from Claude, prevents duplicate tickets when a message is
  processed twice.
- Only registered client tools can run locally. Unknown tool names return an error result.
- Retrieved excerpts and web content are passed to Claude as reference data, and the system
  prompt tells Claude not to follow instructions found inside them.
- MCP connection errors and knowledge base file errors are shown without absolute local paths.
  Errors raised by third-party libraries, such as ChromaDB, are shown as reported.

---

## Limitations

- **Live integrations were not exercised against the real APIs during the final review.** The
  live Claude calls, the Voyage AI embedding calls, the web search tool and the server-side refusal
  fallback (a beta feature) are covered by tests with fake clients and responses only. All
  mock-mode paths and the MCP client/server communication were run for real.
- **Costs in live mode:** each answer is a Claude API call (several calls when tools are used),
  web searches are billed per search, and the first indexing run calls Voyage AI.
- **No authentication or multi-user isolation.** The app is meant to run locally. Any user of
  a running instance can look up any ticket ID, and the MCP server is shared by all browser
  sessions of one Streamlit process.
- **Tickets cannot be updated.** Status stays `open`; there are no update, assignment or comment tools.
- **SQLite and ChromaDB are local files**, suitable for a single instance, not for horizontal
  scaling.
- **Conversation history lives in the browser session** and is lost on refresh. Only the final
  answer text of each turn is kept; tool calls and retrieved context from earlier turns are not
  resent.
- **Mock embeddings are keyword-based** and only meant to exercise the pipeline. Retrieval quality
  in mock mode does not reflect Voyage AI quality.
- **Mock tool selection uses simple keyword rules** and can differ from Claude's decisions.
- The MCP server is started with the same Python interpreter as the app. Run the app from the
  environment where `requirements.txt` is installed.

## Possible extensions

- Ticket update and comment tools on the MCP server.
- User authentication and per-user ticket ownership.
- Streaming responses in the UI.
