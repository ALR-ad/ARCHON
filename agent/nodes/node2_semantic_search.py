"""
PERSON B: Node 2.
Input:  List[ChangedCodeUnit]
Calls:  embed_fn (the real indexing embed_text() in production,
        mock_embed_text() by default for tests) and VectorStore.query()
        (QdrantVectorStore in production, MockVectorStore in tests)
Output: List[Tuple[ChangedCodeUnit, List[Candidate]]] filtered by the similarity floor
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

from shared.schemas import ChangedCodeUnit, Candidate
from shared.interfaces import VectorStore
from shared.config import (
    SIMILARITY_FLOOR,
    EMBEDDING_DIM,
    RENAME_SIMILARITY_THRESHOLD,
    WIKI_QUERY_PREFIX,
    WIKI_TOP_K,
)
from indexing.chunking.code_chunker import TS_LANGUAGE_MAP
from agent.review_filters import is_caller_or_wrapper, is_review_ignored

logger = logging.getLogger("agent.nodes.node2")

# A match counts as pre-existing if the unit's base version already scored
# within this margin of the new score -- even below the floor, so a small edit
# that nudges an old similarity over the floor doesn't look new.
PREEXISTING_MARGIN = 0.05

# Rules are only looked up for real code, not for README / config changes
_CODE_LANGUAGES = set(TS_LANGUAGE_MAP) | set(TS_LANGUAGE_MAP.values())


def mock_embed_text(text: str) -> List[float]:
    """
    Stand-in for shared.interfaces.embed_text() until Person A's
    real embedding model is ready.

    Returns a fixed-length dummy vector (dimension matches EMBEDDING_DIM
    from shared/config.py) so the pipeline shape is correct.
    """
    # Simple deterministic dummy: use hash of text to seed values
    h = hash(text) & 0xFFFFFFFF
    return [float((h + i) % 1000) / 1000.0 for i in range(EMBEDDING_DIM)]


def semantic_search(
    units: List[ChangedCodeUnit],
    store: VectorStore,
    embed_fn: Callable[[str], List[float]] = mock_embed_text,
    similarity_floor: float = SIMILARITY_FLOOR,
    ignore_paths: Optional[List[str]] = None,
    base_code: Optional[Dict[str, str]] = None,
    wiki_top_k: int = WIKI_TOP_K,
) -> List[Tuple[ChangedCodeUnit, List[Candidate]]]:
    """
    Node 2 entry point.

    For each ChangedCodeUnit, embed its code and run two separate lookups:

    Wiki rules (architectural check): the wiki_top_k closest rules, with no
    similarity floor -- prose and code never score high against each other,
    so Node 3's LLM decides whether a rule is actually broken. Skipped for
    non-code files and when wiki_top_k is 0.

    Existing code (duplication check): candidates above the similarity
    floor, minus the ones that are noise rather than duplication:
      - below the similarity floor
      - the unit itself, or a likely rename of it (same file)
      - in review-ignored paths (tests by default)
      - callers / wrappers of the unit, or code the unit calls
      - already matched about as well (within PREEXISTING_MARGIN) by the
        unit's base-branch version (base_code), so a small edit to an old
        function doesn't resurface old similarities
    Units with no code candidates and no rules are dropped from the output.

    Args:
        units: Output of Node 1 (ingest_diff)
        store: VectorStore implementation (MockVectorStore for dev)
        embed_fn: Embedding function (mock_embed_text for dev)
        similarity_floor: Minimum similarity to keep a candidate
        ignore_paths: Candidate paths to drop (None = tests, see review_filters)
        base_code: unit_id -> code at the base commit (from Node 1)
        wiki_top_k: How many wiki rules to attach per unit (0 = none)

    Returns:
        List of (unit, filtered_candidates) tuples, only for units
        that have at least one candidate above the floor.
    """
    import concurrent.futures

    results: List[Tuple[ChangedCodeUnit, List[Candidate]]] = []
    base_code = base_code or {}

    # Texts to embed: every unit, plus the base version of modified units,
    # plus a prefixed copy of each code unit for the wiki-rule lookup.
    texts: List[str] = [u.code for u in units]
    base_index: Dict[str, int] = {}
    wiki_index: Dict[str, int] = {}
    for u in units:
        if u.unit_id in base_code:
            base_index[u.unit_id] = len(texts)
            texts.append(base_code[u.unit_id])
        if wiki_top_k > 0 and u.language.lower() in _CODE_LANGUAGES:
            wiki_index[u.unit_id] = len(texts)
            texts.append(WIKI_QUERY_PREFIX + u.code)

    logger.info("Node 2: preparing to embed %d units (+%d base versions)", len(units), len(base_index))

    # 1. Generate embeddings concurrently to hide network/processing latency
    embeddings: List[List[float]] = [[] for _ in texts]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(texts) or 1, 10)) as executor:
        future_to_index = {executor.submit(embed_fn, t): i for i, t in enumerate(texts)}
        for future in concurrent.futures.as_completed(future_to_index):
            i = future_to_index[future]
            try:
                embeddings[i] = future.result()
            except Exception as exc:
                logger.error("Node 2: embedding %d generated an exception: %s", i, exc)
                embeddings[i] = []  # Fallback for failure

    # 2. Query vector store sequentially (fast) and filter
    for i, unit in enumerate(units):
        embedding = embeddings[i]
        if not embedding:
            continue

        raw_candidates = [
            c for c in store.query(embedding, filters={"source_type": "code"}) if c.source_type == "code"
        ]

        rules: List[Candidate] = []
        if unit.unit_id in wiki_index and embeddings[wiki_index[unit.unit_id]]:
            rules = [
                c
                for c in store.query(
                    embeddings[wiki_index[unit.unit_id]], top_k=wiki_top_k, filters={"source_type": "wiki"}
                )
                if c.source_type == "wiki"
            ]

        old_scores: Dict[str, float] = {}
        if unit.unit_id in base_index and embeddings[base_index[unit.unit_id]]:
            old_candidates = store.query(embeddings[base_index[unit.unit_id]], filters={"source_type": "code"})
            old_scores = {c.chunk_id: c.similarity for c in old_candidates}

        filtered = []
        dropped = {"preexisting": 0, "ignored_path": 0, "caller_or_wrapper": 0}
        for c in raw_candidates:
            if c.similarity < similarity_floor:
                continue
            if c.file_path == unit.file_path:
                if c.symbol_name == unit.symbol_name:
                    continue
                if c.similarity >= RENAME_SIMILARITY_THRESHOLD:
                    continue
            if c.chunk_id in old_scores and old_scores[c.chunk_id] >= c.similarity - PREEXISTING_MARGIN:
                dropped["preexisting"] += 1
                continue
            if is_review_ignored(c.file_path, ignore_paths):
                dropped["ignored_path"] += 1
                continue
            if is_caller_or_wrapper(unit.code, unit.symbol_name, c.code_snippet, c.symbol_name):
                dropped["caller_or_wrapper"] += 1
                continue
            filtered.append(c)

        logger.info(
            "Node 2: %s::%s -> %d raw candidates, %d kept (floor %.2f; dropped %s), %d rule(s)",
            unit.file_path,
            unit.symbol_name,
            len(raw_candidates),
            len(filtered),
            similarity_floor,
            ", ".join(f"{k}={v}" for k, v in dropped.items() if v) or "none",
            len(rules),
        )

        if filtered or rules:
            results.append((unit, filtered + rules))

    logger.info("Node 2: %d unit(s) with matches out of %d total", len(results), len(units))
    return results
