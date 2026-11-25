"""
IT Support Chatbot - console chatbot (single interaction).

Flow (single interaction):
    1. Load configuration from the .env file.
    2. Read one question from the user in the console.
    3. Build the `messages` object for the Claude Messages API.
    4. Invoke the LLM (or a mock, when MOCK_MODE=true).
    5. Print the question and the answer, then exit.

Run:
    python chatbot.py
"""

import os
import sys

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-opus-5-5"
# Thinking is always on for current Claude models and counts against max_tokens,
# so a small cap truncates answers (especially with RAG context and tool use).
DEFAULT_MAX_TOKENS = 16000

SYSTEM_PROMPT = (
    "You are a helpful IT support assistant. Answer the user's technical "
    "question clearly and concisely, with step-by-step instructions when useful."
)


def load_config() -> dict:
    """Load settings from the .env file and environment variables."""
    load_dotenv()  # Finds .env next to this file (or a parent folder); never prints values.

    return {
        "model": os.getenv("CLAUDE_MODEL", DEFAULT_MODEL),
        "max_tokens": int(os.getenv("CLAUDE_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        "mock_mode": os.getenv("MOCK_MODE", "false").strip().lower() in ("1", "true", "yes"),
        # Server-side refusal fallback: if the model declines a request, the API
        # retries it on a suitable fallback model within the same call.
        "refusal_fallback": os.getenv("ENABLE_REFUSAL_FALLBACK", "true").strip().lower()
        in ("1", "true", "yes"),
    }


# ---------------------------------------------------------------------------
# Core steps
# ---------------------------------------------------------------------------


def get_user_question() -> str:
    """Read a non-empty question from the console."""
    # Remove surrounding spaces and any invisible BOM character (added by some Windows terminals).
    question = input("Ask your IT support question: ").replace("﻿", "").strip()
    if not question:
        raise ValueError("The question cannot be empty.")
    return question


def build_messages(question: str) -> list:
    """Prepare the messages object expected by the Claude Messages API."""
    return [{"role": "user", "content": question}]


def ask_mock(messages: list) -> str:
    """Return a canned answer so the app can be tested without API credits."""
    question = messages[-1]["content"]
    return f"[MOCK] This is a simulated answer to: '{question}'. Set MOCK_MODE=false to call Claude."


def make_create_message(config: dict):
    """Return a function `create(**request)` that calls the Messages API.

    It applies the refusal-fallback setting, so every caller (console, app,
    tool loop) uses the same endpoint. Raises RuntimeError if no API key is set.
    """
    import anthropic  # Imported here so mock mode works even without the SDK.

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")

    client = anthropic.Anthropic()  # Reads ANTHROPIC_API_KEY from the environment.

    def create(**request):
        if config["refusal_fallback"]:
            return client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                **request,
            )
        return client.messages.create(**request)

    return create


def ask_claude(messages: list, config: dict, system: str = SYSTEM_PROMPT) -> str:
    """Send the messages to Claude and return the text of the answer.

    `system` defaults to the IT support prompt; the RAG app passes an extended one.
    """
    create = make_create_message(config)
    response = create(
        model=config["model"],
        max_tokens=config["max_tokens"],
        system=system,
        messages=messages,
    )

    # Always check why the model stopped before reading the content.
    if response.stop_reason == "refusal":
        return "Sorry, the assistant declined to answer this request."

    answer = "".join(block.text for block in response.content if block.type == "text")
    if response.stop_reason == "max_tokens":
        answer += "\n\n[Answer truncated: increase CLAUDE_MAX_TOKENS for longer answers.]"
    return answer


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run one question/answer interaction. Returns a process exit code."""
    config = load_config()

    try:
        question = get_user_question()
        messages = build_messages(question)

        if config["mock_mode"]:
            answer = ask_mock(messages)
        else:
            answer = ask_claude(messages, config)

    except (KeyboardInterrupt, EOFError):
        print("\nCancelled by user.")
        return 1
    except ImportError:
        print("Error: the 'anthropic' package is not installed. Run: pip install -r requirements.txt")
        return 1
    except (ValueError, RuntimeError) as error:
        print(f"Error: {error}")
        return 1
    except Exception as error:  # noqa: BLE001 - handled below by SDK error type
        return handle_api_error(error)

    print("\n" + "=" * 60)
    print(f"Model:    {'mock' if config['mock_mode'] else config['model']}")
    print(f"Question: {question}")
    print("-" * 60)
    print(f"Answer:\n{answer}")
    print("=" * 60)
    return 0


def handle_api_error(error: Exception) -> int:
    """Print a friendly message for Anthropic SDK errors (most specific first)."""
    import anthropic

    if isinstance(error, anthropic.AuthenticationError):
        print("Error: invalid API key. Check ANTHROPIC_API_KEY in your .env file.")
    elif isinstance(error, anthropic.NotFoundError):
        print("Error: model not found. Check CLAUDE_MODEL in your .env file.")
    elif isinstance(error, anthropic.BadRequestError):
        print(f"Error: bad request - {error.message}")
        print("Tip: if the model does not support fallbacks, set ENABLE_REFUSAL_FALLBACK=false.")
    elif isinstance(error, anthropic.RateLimitError):
        print("Error: rate limit reached. Please wait and try again.")
    elif isinstance(error, anthropic.APIStatusError):
        print(f"Error: API returned status {error.status_code} - {error.message}")
    elif isinstance(error, anthropic.APIConnectionError):
        print("Error: could not connect to the Anthropic API. Check your network.")
    else:
        print(f"Unexpected error: {error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
