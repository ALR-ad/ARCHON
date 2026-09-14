"""
Tests for agent/nodes/node2_semantic_search.py

Verifies:
1. semantic_search() returns List[Tuple[ChangedCodeUnit, List[Candidate]]]
2. Candidates below SIMILARITY_FLOOR are filtered out
3. Units with zero candidates after filtering are excluded
4. mock_embed_text() returns correct dimension
5. Integration with MockVectorStore works end-to-end
"""

from typing import Dict, List, Optional

from agent.nodes.node2_semantic_search import semantic_search, mock_embed_text
from agent.llm.mock_vectorstore import MockVectorStore
from shared.schemas import ChangedCodeUnit, Candidate
from shared.interfaces import VectorStore
from shared.config import EMBEDDING_DIM, SIMILARITY_FLOOR


# -- Fixtures ----------------------------------------------------------------

def _make_unit(symbol: str = "testFn", code: str = "function testFn() {}") -> ChangedCodeUnit:
    return ChangedCodeUnit(
        unit_id="test_unit_001",
        file_path="src/test.ts",
        symbol_name=symbol,
        language="typescript",
        start_line=1,
        end_line=5,
        code=code,
        diff_type="added",
    )


class HighSimilarityStore(VectorStore):
    """Returns candidates above the default floor."""
    def query(self, embedding: List[float], top_k: int = 8,
              filters: Optional[Dict[str, str]] = None) -> List[Candidate]:
        return [
            Candidate(
                chunk_id="high_1",
                file_path="src/utils.ts",
                symbol_name="utilFn",
                code_snippet="export function utilFn() { /* ... */ }",
                source_type="code",
                similarity=0.92,
            )
        ]

    def get_index_status(self) -> Dict[str, str]:
        return {"last_indexed_sha": "test", "chunk_count": "1"}


class LowSimilarityStore(VectorStore):
    """Returns candidates below the default floor."""
    def query(self, embedding: List[float], top_k: int = 8,
              filters: Optional[Dict[str, str]] = None) -> List[Candidate]:
        return [
            Candidate(
                chunk_id="low_1",
                file_path="src/unrelated.ts",
                symbol_name="unrelatedFn",
                code_snippet="export function unrelatedFn() {}",
                source_type="code",
                similarity=0.40,
            )
        ]

    def get_index_status(self) -> Dict[str, str]:
        return {"last_indexed_sha": "test", "chunk_count": "1"}


class EmptyStore(VectorStore):
    """Returns no candidates at all."""
    def query(self, embedding: List[float], top_k: int = 8,
              filters: Optional[Dict[str, str]] = None) -> List[Candidate]:
        return []

    def get_index_status(self) -> Dict[str, str]:
        return {"last_indexed_sha": "test", "chunk_count": "0"}


# -- Tests: mock_embed_text ---------------------------------------------------

def test_mock_embed_text_returns_correct_dimension():
    vec = mock_embed_text("some code here")
    assert isinstance(vec, list)
    assert len(vec) == EMBEDDING_DIM


def test_mock_embed_text_returns_floats():
    vec = mock_embed_text("code")
    assert all(isinstance(v, float) for v in vec)


def test_mock_embed_text_is_deterministic():
    v1 = mock_embed_text("same input")
    v2 = mock_embed_text("same input")
    assert v1 == v2


# -- Tests: semantic_search with MockVectorStore ------------------------------

def test_semantic_search_with_mock_store():
    """MockVectorStore returns similarity=0.86, which is above the default floor."""
    unit = _make_unit()
    store = MockVectorStore()
    results = semantic_search([unit], store)

    assert len(results) == 1
    returned_unit, candidates = results[0]
    assert returned_unit == unit
    assert len(candidates) == 1
    assert candidates[0].similarity >= SIMILARITY_FLOOR


# -- Tests: filtering --------------------------------------------------------

def test_high_similarity_candidates_are_kept():
    unit = _make_unit()
    results = semantic_search([unit], HighSimilarityStore())
    assert len(results) == 1
    _, candidates = results[0]
    assert all(c.similarity >= SIMILARITY_FLOOR for c in candidates)


def test_low_similarity_candidates_are_filtered_out():
    unit = _make_unit()
    results = semantic_search([unit], LowSimilarityStore())
    # All candidates are below the floor, so the unit should be excluded
    assert len(results) == 0


def test_empty_store_returns_no_results():
    unit = _make_unit()
    results = semantic_search([unit], EmptyStore())
    assert len(results) == 0


def test_multiple_units_filtered_independently():
    """Each unit is evaluated independently against the store."""
    units = [_make_unit(symbol="fn1"), _make_unit(symbol="fn2")]
    results = semantic_search(units, HighSimilarityStore())
    assert len(results) == 2


def test_custom_similarity_floor():
    """Passing a custom floor overrides the config default."""
    unit = _make_unit()
    # MockVectorStore returns 0.86. With floor=0.90, it should be filtered.
    results = semantic_search([unit], MockVectorStore(), similarity_floor=0.90)
    assert len(results) == 0

    # With floor=0.50, it should pass.
    results = semantic_search([unit], MockVectorStore(), similarity_floor=0.50)
    assert len(results) == 1
