"""
tests/integration/test_end_to_end_real.py

Integration tests that run the full Archon pipeline (Nodes 1-4) against
REAL services: Qdrant vector store, Ollama embeddings (nomic-embed-text),
and Ollama LLM evaluation (qwen2.5-coder:7b).

Requirements:
  - Ollama running locally (http://localhost:11434)
  - Models pulled: nomic-embed-text, qwen2.5-coder:7b
  - The test index must be built first:
      python -m indexing.run_index --repo-path scratch/fake_repo \
             --db-path scratch/qdrant_test --full

Usage:
  pytest tests/integration/test_end_to_end_real.py -v -s

These tests are SLOW (~10-30s each) because they call real Ollama.
They are marked with @pytest.mark.integration so they can be skipped in CI:
  pytest -m "not integration"
"""

import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import List

import pytest
import requests

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.schemas import Candidate, ChangedCodeUnit, Finding
from shared.config import SIMILARITY_FLOOR, CONFIDENCE_THRESHOLD, EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Markers and skip conditions
# ---------------------------------------------------------------------------

def _ollama_reachable() -> bool:
    """Check if Ollama is running at localhost:11434."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _model_available(model_name: str) -> bool:
    """Check if a specific model is available in Ollama."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        return any(model_name in m for m in models)
    except Exception:
        return False



requires_ollama = pytest.mark.skipif(
    not _ollama_reachable(),
    reason="Ollama not running at http://localhost:11434",
)

requires_embed_model = pytest.mark.skipif(
    not _model_available("nomic-embed-text"),
    reason="nomic-embed-text model not available in Ollama",
)

requires_eval_model = pytest.mark.skipif(
    not _model_available("qwen2.5-coder:7b"),
    reason="qwen2.5-coder:7b model not available in Ollama",
)


# Composite marker for all real-service tests
integration = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_vector_store(tmp_path_factory):
    """Build the real Qdrant test index in a temporary directory, yield it, then close."""
    from indexing.vectorstore.store import QdrantVectorStore
    from indexing.incremental import IncrementalIndexer

    qdrant_path = str(tmp_path_factory.mktemp("qdrant_test"))
    repo_path = str(Path(__file__).resolve().parents[2] / "scratch" / "fake_repo")

    # Run the indexer to populate the temp store
    with IncrementalIndexer(repo_root=repo_path, db_path=qdrant_path) as indexer:
        indexer.run_indexing(force_full=True)

    store = QdrantVectorStore(path=qdrant_path)
    yield store
    store.close()


@pytest.fixture(scope="module")
def real_embed_fn():
    """Return the real embed_text function from Track A."""
    from indexing.embeddings.embedder import embed_text
    return embed_text


@pytest.fixture(scope="module")
def real_eval_fn():
    """Return the real Ollama evaluation function."""
    from agent.llm.ollama_eval import evaluate_with_ollama
    return evaluate_with_ollama


def _make_unit_id(file_path: str, symbol_name: str, sha: str) -> str:
    raw = f"{file_path}:{symbol_name}:{sha}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

DUPLICATE_DATE_PARSER = ChangedCodeUnit(
    unit_id=_make_unit_id(
        "src/orders/order_dates.py", "parse_order_date", "test_sha_001"
    ),
    file_path="src/orders/order_dates.py",
    symbol_name="parse_order_date",
    language="python",
    start_line=1,
    end_line=9,
    code=(
        'def parse_order_date(raw: str):\n'
        '    """Parse an order date string in YYYY-MM-DD format."""\n'
        '    parts = raw.split("-")\n'
        '    year = int(parts[0])\n'
        '    month = int(parts[1])\n'
        '    day = int(parts[2])\n'
        '    from datetime import datetime\n'
        '    return datetime(year, month, day)\n'
    ),
    diff_type="added",
)

NOVEL_SHIPPING_FN = ChangedCodeUnit(
    unit_id=_make_unit_id(
        "src/shipping/calculator.py", "compute_shipping_cost", "test_sha_002"
    ),
    file_path="src/shipping/calculator.py",
    symbol_name="compute_shipping_cost",
    language="python",
    start_line=1,
    end_line=15,
    code=(
        'def compute_shipping_cost(weight_kg: float, distance_km: float, express: bool = False) -> float:\n'
        '    """\n'
        '    Calculate shipping cost based on package weight, distance, and service tier.\n'
        '    Uses a tiered pricing model with express surcharge.\n'
        '    """\n'
        '    base_rate = 2.50\n'
        '    per_kg = 0.75\n'
        '    per_km = 0.02\n'
        '    cost = base_rate + (weight_kg * per_kg) + (distance_km * per_km)\n'
        '    if express:\n'
        '        cost *= 1.5\n'
        '    if weight_kg > 20:\n'
        '        cost += 10.0  # heavy package surcharge\n'
        '    return round(cost, 2)\n'
    ),
    diff_type="added",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@integration
@requires_ollama
@requires_embed_model
class TestEmbeddingContract:
    """Verify the real embed_text function meets the shared contract."""

    def test_embed_text_returns_correct_dimension(self, real_embed_fn):
        vec = real_embed_fn("Hello, world!")
        assert isinstance(vec, list)
        assert len(vec) == EMBEDDING_DIM
        assert all(isinstance(v, float) for v in vec)

    def test_embed_text_is_deterministic(self, real_embed_fn):
        vec1 = real_embed_fn("def foo(): pass")
        vec2 = real_embed_fn("def foo(): pass")
        assert vec1 == vec2

    def test_embed_text_different_inputs_differ(self, real_embed_fn):
        vec1 = real_embed_fn("def parse_date(raw): pass")
        vec2 = real_embed_fn("class ShippingCalculator: pass")
        assert vec1 != vec2


@integration
@requires_ollama
@requires_embed_model
class TestVectorStoreContract:
    """Verify the real QdrantVectorStore meets the shared VectorStore contract."""

    def test_index_has_chunks(self, real_vector_store):
        status = real_vector_store.get_index_status()
        assert int(status.get("chunk_count", 0)) > 0
        assert "last_indexed_sha" in status

    def test_query_returns_candidates(self, real_vector_store, real_embed_fn):
        vec = real_embed_fn("def parse_date(raw): parts = raw.split('-')")
        candidates = real_vector_store.query(vec, top_k=3)
        assert isinstance(candidates, list)
        assert len(candidates) > 0
        for c in candidates:
            assert isinstance(c, Candidate)
            assert hasattr(c, "chunk_id")
            assert hasattr(c, "file_path")
            assert hasattr(c, "symbol_name")
            assert hasattr(c, "code_snippet")
            assert hasattr(c, "source_type")
            assert hasattr(c, "similarity")
            assert c.source_type in ("code", "wiki")
            assert 0.0 <= c.similarity <= 1.0

    def test_query_sorted_descending(self, real_vector_store, real_embed_fn):
        vec = real_embed_fn("def parse_date(raw): pass")
        candidates = real_vector_store.query(vec, top_k=5)
        sims = [c.similarity for c in candidates]
        assert sims == sorted(sims, reverse=True)

    def test_query_respects_top_k(self, real_vector_store, real_embed_fn):
        vec = real_embed_fn("some code")
        candidates = real_vector_store.query(vec, top_k=2)
        assert len(candidates) <= 2


@integration
@requires_ollama
@requires_embed_model
class TestSemanticSearchNode:
    """Test Node 2 with real embeddings and real vector store."""

    def test_duplicate_code_finds_matches(self, real_vector_store, real_embed_fn):
        from agent.nodes.node2_semantic_search import semantic_search

        matches = semantic_search(
            units=[DUPLICATE_DATE_PARSER],
            store=real_vector_store,
            embed_fn=real_embed_fn,
            similarity_floor=SIMILARITY_FLOOR,
        )
        assert len(matches) == 1, "Expected 1 unit with matches for duplicate code"
        unit, candidates = matches[0]
        assert unit.symbol_name == "parse_order_date"
        assert len(candidates) >= 1
        # The top candidate should be parse_date from date_utils.py
        top = candidates[0]
        assert "date_utils" in top.file_path
        assert top.similarity >= SIMILARITY_FLOOR

    def test_novel_code_finds_no_matches(self, real_vector_store, real_embed_fn):
        from agent.nodes.node2_semantic_search import semantic_search

        matches = semantic_search(
            units=[NOVEL_SHIPPING_FN],
            store=real_vector_store,
            embed_fn=real_embed_fn,
            similarity_floor=SIMILARITY_FLOOR,
        )
        assert len(matches) == 0, (
            "Expected 0 matches for novel shipping code, but got "
            f"{len(matches)} -- possible false positive"
        )


@integration
@requires_ollama
@requires_embed_model
@requires_eval_model
class TestFullPipelineEndToEnd:
    """
    Full end-to-end integration test: Node 2 -> Node 3 -> Node 4.

    These tests call real Ollama for LLM evaluation and are SLOW (~10-30s each).
    """

    def test_true_positive_duplicate_detected(
        self, real_vector_store, real_embed_fn, real_eval_fn
    ):
        """
        A PR introduces parse_order_date() which duplicates the indexed
        parse_date(). The pipeline should detect it as a duplicate with
        high confidence and produce a rendered comment.
        """
        from agent.nodes.node2_semantic_search import semantic_search
        from agent.nodes.node3_evaluate import evaluate
        from agent.nodes.node4_comment import render_comment

        # Node 2
        matches = semantic_search(
            units=[DUPLICATE_DATE_PARSER],
            store=real_vector_store,
            embed_fn=real_embed_fn,
        )
        assert len(matches) >= 1, "Node 2: expected matches for duplicate code"

        # Node 3
        findings = evaluate(
            matches=matches,
            eval_fn=real_eval_fn,
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )
        assert len(findings) >= 1, "Node 3: expected at least 1 finding for duplicate"

        finding = findings[0]
        assert isinstance(finding, Finding)
        assert finding.is_duplicate is True
        assert finding.confidence >= CONFIDENCE_THRESHOLD
        assert len(finding.reasoning) > 0
        assert finding.unit.symbol_name == "parse_order_date"

        # Node 4
        comment = render_comment(findings)
        assert len(comment) > 0
        assert "Duplicated Logic" in comment
        assert "parse_order_date" in comment
        assert "pr-reviewer-agent-findings" in comment  # bot marker

    def test_true_negative_no_false_positives(
        self, real_vector_store, real_embed_fn
    ):
        """
        A PR introduces compute_shipping_cost() which is genuinely novel.
        The pipeline should produce zero findings.
        """
        from agent.nodes.node2_semantic_search import semantic_search

        # Node 2 should filter everything below SIMILARITY_FLOOR
        matches = semantic_search(
            units=[NOVEL_SHIPPING_FN],
            store=real_vector_store,
            embed_fn=real_embed_fn,
        )
        assert len(matches) == 0, (
            f"False positive detected: novel code matched {len(matches)} unit(s). "
            "This indicates the similarity floor or embeddings may need tuning."
        )
