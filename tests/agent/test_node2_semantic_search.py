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

# -- Tests: rename filtering --------------------------------------------------

class CustomStore(VectorStore):
    def __init__(self, candidates):
        self.candidates = candidates
    def query(self, embedding, top_k=8, filters=None):
        return self.candidates
    def get_index_status(self):
        return {}

def test_same_file_same_symbol_high_similarity_excluded():
    unit = _make_unit(symbol='testFn')
    c = Candidate(chunk_id='1', file_path='src/test.ts', symbol_name='testFn', code_snippet='', source_type='code', similarity=0.99)
    results = semantic_search([unit], CustomStore([c]))
    assert len(results) == 0

def test_same_file_different_symbol_similarity_above_threshold_excluded():
    from shared.config import RENAME_SIMILARITY_THRESHOLD
    unit = _make_unit(symbol='newFn')
    c = Candidate(chunk_id='1', file_path='src/test.ts', symbol_name='oldFn', code_snippet='', source_type='code', similarity=RENAME_SIMILARITY_THRESHOLD + 0.01)
    results = semantic_search([unit], CustomStore([c]))
    assert len(results) == 0

def test_same_file_different_symbol_similarity_below_threshold_included():
    from shared.config import RENAME_SIMILARITY_THRESHOLD
    unit = _make_unit(symbol='newFn')
    c = Candidate(chunk_id='1', file_path='src/test.ts', symbol_name='otherFn', code_snippet='', source_type='code', similarity=RENAME_SIMILARITY_THRESHOLD - 0.05)
    results = semantic_search([unit], CustomStore([c]))
    assert len(results) == 1

def test_different_file_same_symbol_high_similarity_included():
    from shared.config import RENAME_SIMILARITY_THRESHOLD
    unit = _make_unit(symbol='testFn')
    c = Candidate(chunk_id='1', file_path='src/other.ts', symbol_name='testFn', code_snippet='', source_type='code', similarity=0.99)
    results = semantic_search([unit], CustomStore([c]))
    assert len(results) == 1



# -- Tests: noise filters -----------------------------------------------------

def _cand(chunk_id, file_path, symbol, code="", similarity=0.9, source_type="code"):
    return Candidate(chunk_id=chunk_id, file_path=file_path, symbol_name=symbol,
                     code_snippet=code, source_type=source_type, similarity=similarity)


def test_test_file_candidates_dropped_by_default():
    unit = _make_unit()
    c = _cand("t1", "tests/test_utils.py", "test_util")
    assert semantic_search([unit], CustomStore([c])) == []
    # Explicit empty ignore list reviews tests too
    assert len(semantic_search([unit], CustomStore([c]), ignore_paths=[])) == 1


def test_wiki_candidates_never_path_filtered():
    unit = _make_unit()
    c = _cand("w1", "tests/STYLE.md", "", similarity=0.9, source_type="wiki")
    assert len(semantic_search([unit], CustomStore([c]))) == 1


def test_caller_candidate_dropped():
    unit = _make_unit(symbol="parseDate", code="function parseDate(raw) { return new Date(raw); }")
    caller = _cand("c1", "src/orders.ts", "loadOrder", code="function loadOrder(o) { return parseDate(o.date); }")
    assert semantic_search([unit], CustomStore([caller])) == []


def test_candidate_called_by_unit_dropped():
    unit = _make_unit(symbol="loadOrder", code="function loadOrder(o) { return parseDate(o.date); }")
    callee = _cand("c1", "src/dates.ts", "parseDate", code="function parseDate(raw) { return new Date(raw); }")
    assert semantic_search([unit], CustomStore([callee])) == []


class OldVsNewStore(VectorStore):
    """Returns different candidates depending on which version is queried."""
    def __init__(self, by_first_value):
        self.by_first_value = by_first_value
    def query(self, embedding, top_k=8, filters=None):
        return self.by_first_value[embedding[0]]
    def get_index_status(self):
        return {}


def test_matches_the_base_version_already_had_are_dropped():
    unit = _make_unit(symbol="validate", code="NEW")
    unit.diff_type = "modified"
    old_match = _cand("old", "src/a.ts", "checkA")
    new_match = _cand("new", "src/b.ts", "checkB")
    store = OldVsNewStore({1.0: [old_match, new_match], 0.0: [old_match]})
    embed = lambda text: [1.0] if text == "NEW" else [0.0]

    results = semantic_search([unit], store, embed_fn=embed, base_code={unit.unit_id: "OLD"})
    assert [c.chunk_id for c in results[0][1]] == ["new"]

    # Without a base version both matches are kept
    results = semantic_search([unit], store, embed_fn=embed)
    assert {c.chunk_id for c in results[0][1]} == {"old", "new"}


def test_base_match_just_below_floor_counts_as_preexisting():
    """An edit that nudges an old 0.80 match to 0.83 must not surface it at floor 0.82."""
    unit = _make_unit(symbol="validate", code="NEW")
    store = OldVsNewStore({
        1.0: [_cand("near", "src/a.ts", "checkA", similarity=0.83)],
        0.0: [_cand("near", "src/a.ts", "checkA", similarity=0.80)],
    })
    embed = lambda text: [1.0] if text == "NEW" else [0.0]
    results = semantic_search([unit], store, embed_fn=embed, similarity_floor=0.82,
                              base_code={unit.unit_id: "OLD"})
    assert results == []


def test_match_much_closer_than_base_version_is_kept():
    """If the PR makes the code much more similar (0.60 -> 0.90), that's new duplication."""
    unit = _make_unit(symbol="validate", code="NEW")
    store = OldVsNewStore({
        1.0: [_cand("grew", "src/a.ts", "checkA", similarity=0.90)],
        0.0: [_cand("grew", "src/a.ts", "checkA", similarity=0.60)],
    })
    embed = lambda text: [1.0] if text == "NEW" else [0.0]
    results = semantic_search([unit], store, embed_fn=embed, base_code={unit.unit_id: "OLD"})
    assert [c.chunk_id for c in results[0][1]] == ["grew"]
