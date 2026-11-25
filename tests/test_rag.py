"""
Basic automated tests for the RAG pipeline: chunking, indexing and retrieval.

No paid API calls are made: tests use the deterministic MockEmbedder, a fake
Voyage client, and a stubbed Claude call. Each test gets its own temporary
ChromaDB directory.

Run:
    python -m pytest
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import app_implementation
import rag

PROJECT_KB = Path(__file__).resolve().parent.parent / "it_sector.txt"

SAMPLE_KB = """# Sample KB

## Password reset
Go to the reset portal and verify your identity with MFA.

## VPN troubleshooting
If the VPN gateway is unreachable, check your internet connection and restart the router.

## Printers
Print to the secure queue and tap your badge at the printer to release the job.
"""


def make_kb(tmp_path, text=SAMPLE_KB, embedder=None):
    kb_file = tmp_path / "kb.txt"
    kb_file.write_text(text, encoding="utf-8")
    return rag.KnowledgeBase(embedder or rag.MockEmbedder(), str(kb_file), str(tmp_path / "chroma"))


def mock_config(**overrides):
    rag_config = {
        "enabled": True, "mock_embeddings": True, "embedding_model": "voyage-4",
        "kb_path": "kb.txt", "chroma_path": "chroma", "top_k": 3,
        "min_similarity": 0.2, "max_chunk_chars": 1500,
    }
    rag_config.update(overrides)
    return {"mock_mode": True, "model": "test", "max_tokens": 100,
            "refusal_fallback": False, "rag": rag_config}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunking_creates_one_chunk_per_section():
    chunks = rag.chunk_document(SAMPLE_KB, "kb.txt")
    assert [c.section for c in chunks] == ["Password reset", "VPN troubleshooting", "Printers"]
    assert all(c.text.startswith(f"Section: {c.section}\n") for c in chunks)
    assert all(c.source == "kb.txt" for c in chunks)
    assert [c.index for c in chunks] == [0, 1, 2]


def test_chunking_splits_long_sections_under_the_limit():
    paragraphs = "\n\n".join(f"Paragraph {i}: " + "word " * 60 for i in range(10))
    chunks = rag.chunk_document(f"## Long section\n{paragraphs}", "kb.txt", max_chars=500)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)
    assert all(c.section == "Long section" for c in chunks)


def test_chunk_ids_are_stable_and_unique():
    first = rag.chunk_document(SAMPLE_KB, "kb.txt")
    second = rag.chunk_document(SAMPLE_KB, "kb.txt")
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)


def test_project_knowledge_base_covers_required_topics():
    chunks = rag.chunk_document(PROJECT_KB.read_text(encoding="utf-8"), PROJECT_KB.name)
    sections = " ".join(c.section.lower() for c in chunks)
    for topic in ["password", "mfa", "vpn", "wi-fi", "outlook", "printer", "software",
                  "cybersecurity", "troubleshooting"]:
        assert topic in sections, topic


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def test_mock_embeddings_are_deterministic_and_normalized():
    embedder = rag.MockEmbedder()
    first = embedder.embed_query("Reset my password")
    assert first == rag.MockEmbedder().embed_query("Reset my password")
    assert len(first) == rag.MOCK_EMBEDDING_DIM
    assert abs(sum(v * v for v in first) - 1.0) < 1e-9


def test_voyage_embedder_uses_document_and_query_input_types():
    calls = []

    class FakeVoyageClient:
        def embed(self, texts, model=None, input_type=None):
            calls.append((len(texts), model, input_type))
            return SimpleNamespace(embeddings=[[0.1, 0.2] for _ in texts])

    embedder = rag.VoyageEmbedder("voyage-test", client=FakeVoyageClient())
    embedder.embed_documents(["a", "b"])
    embedder.embed_query("c")
    assert calls == [(2, "voyage-test", "document"), (1, "voyage-test", "query")]
    assert embedder.simulated is False


def test_voyage_errors_become_rag_errors():
    class FailingClient:
        def embed(self, *args, **kwargs):
            raise ConnectionError("network down")

    with pytest.raises(rag.RAGError):
        rag.VoyageEmbedder("voyage-test", client=FailingClient()).embed_query("hi")


def test_mock_and_real_embeddings_use_separate_collections():
    real = rag.VoyageEmbedder("voyage-4", client=object())
    assert rag.collection_name(rag.MockEmbedder()) != rag.collection_name(real)
    assert rag.collection_name(rag.MockEmbedder()).startswith("it_kb_mock")


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def test_indexing_twice_does_not_duplicate(tmp_path):
    kb = make_kb(tmp_path)
    first = kb.ensure_index()
    second = kb.ensure_index()
    assert (first.added, first.total) == (3, 3)
    assert (second.added, second.removed, second.total) == (0, 0, 3)


def test_index_persists_and_only_changed_chunks_are_reembedded(tmp_path):
    make_kb(tmp_path).ensure_index()

    edited = SAMPLE_KB.replace("tap your badge", "sign in with your PIN")
    reopened = make_kb(tmp_path, text=edited)  # new client, same directory
    stats = reopened.ensure_index()
    assert (stats.added, stats.removed, stats.total) == (1, 1, 3)


def test_missing_knowledge_base_file_raises_friendly_error(tmp_path):
    kb = rag.KnowledgeBase(rag.MockEmbedder(), str(tmp_path / "missing.txt"), str(tmp_path / "chroma"))
    with pytest.raises(rag.RAGError, match="not found"):
        kb.ensure_index()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("question, expected_section", [
    ("How do I reset my password?", "Password reset"),
    ("The VPN gateway is unreachable", "VPN troubleshooting"),
    ("How do I release my print job?", "Printers"),
])
def test_retrieval_returns_the_relevant_section_first(tmp_path, question, expected_section):
    kb = make_kb(tmp_path)
    kb.ensure_index()
    results = kb.search(question, top_k=3, min_similarity=0.0)
    assert results[0].section == expected_section
    assert results == sorted(results, key=lambda r: r.similarity, reverse=True)


def test_retrieval_on_the_project_knowledge_base(tmp_path):
    kb = rag.KnowledgeBase(rag.MockEmbedder(), str(PROJECT_KB), str(tmp_path / "chroma"))
    kb.ensure_index()
    assert "MFA" in kb.search("I lost my phone and cannot approve MFA")[0].section
    assert "Outlook" in kb.search("Outlook will not open")[0].section


def test_unrelated_or_empty_questions_return_no_results(tmp_path):
    kb = make_kb(tmp_path)
    kb.ensure_index()
    assert kb.search("best pizza recipe in town", min_similarity=0.2) == []
    assert kb.search("   ") == []


def test_search_on_empty_collection_returns_no_results(tmp_path):
    assert make_kb(tmp_path).search("password") == []


# ---------------------------------------------------------------------------
# Prompt injection into Claude and mock-mode behaviour
# ---------------------------------------------------------------------------


def test_context_is_injected_only_into_the_latest_user_message(tmp_path, monkeypatch):
    kb = make_kb(tmp_path)
    kb.ensure_index()
    captured = {}

    def fake_ask_claude(messages, config, system):
        captured.update(messages=messages, system=system)
        return "live answer"

    monkeypatch.setattr(app_implementation, "ask_claude", fake_ask_claude)
    config = mock_config()
    config["mock_mode"] = False
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!", "sources": [], "notice": None},
        {"role": "user", "content": "How do I reset my password?"},
    ]

    response = app_implementation.answer_question(history, config, kb)

    assert response.answer == "live answer"
    assert response.sources[0]["section"] == "Password reset"
    assert captured["messages"][0] == {"role": "user", "content": "Hello"}
    assert set(captured["messages"][1]) == {"role", "content"}  # UI-only keys removed
    assert "<knowledge_base>" in captured["messages"][-1]["content"]
    assert "knowledge_base" in captured["system"]
    assert history[-1]["content"] == "How do I reset my password?"  # history unchanged


def test_mock_mode_answer_is_labelled_as_simulated(tmp_path):
    kb = make_kb(tmp_path)
    kb.ensure_index()
    messages = [{"role": "user", "content": "How do I reset my password?"}]
    response = app_implementation.answer_question(messages, mock_config(), kb)
    assert response.simulated_answer and response.simulated_retrieval
    assert response.answer.startswith("**[MOCK]**")
    assert "Password reset" in response.answer


def test_failed_live_retrieval_is_reported_not_replaced_with_mock(tmp_path, monkeypatch):
    class QueryFailsClient:
        """Indexing works; the query call fails like a network error would."""

        def embed(self, texts, model=None, input_type=None):
            if input_type == "query":
                raise ConnectionError("network down")
            return SimpleNamespace(embeddings=[[1.0, float(i)] for i, _ in enumerate(texts)])

    kb = make_kb(tmp_path, embedder=rag.VoyageEmbedder("voyage-test", client=QueryFailsClient()))
    assert kb.ensure_index().total == 3
    monkeypatch.setattr(app_implementation, "ask_claude", lambda m, c, system: "live answer")
    config = mock_config(mock_embeddings=False)
    config["mock_mode"] = False

    response = app_implementation.answer_question(
        [{"role": "user", "content": "VPN help"}], config, kb
    )

    assert response.sources == []
    assert response.simulated_retrieval is False
    assert "search failed" in response.notice


def test_no_results_produces_a_notice(tmp_path):
    kb = make_kb(tmp_path)
    kb.ensure_index()
    response = app_implementation.answer_question(
        [{"role": "user", "content": "best pizza recipe"}], mock_config(), kb
    )
    assert response.sources == [] and "No relevant" in response.notice
