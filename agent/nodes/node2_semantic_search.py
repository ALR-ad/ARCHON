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
from shared.config import SIMILARITY_FLOOR, EMBEDDING_DIM, RENAME_SIMILARITY_THRESHOLD
from agent.review_filters import is_caller_or_wrapper, is_review_ignored

logger = logging.getLogger("agent.nodes.node2")


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
) -> List[Tuple[ChangedCodeUnit, List[Candidate]]]:
    """
    Node 2 entry point.

    For each ChangedCodeUnit, embed its code, query the vector store,
    and filter out candidates that are noise rather than duplication:
      - below the similarity floor
      - the unit itself, or a likely rename of it (same file)
      - in review-ignored paths (tests by default)
      - callers / wrappers of the unit, or code the unit calls
      - already matched by the unit's base-branch version (base_code), so a
        small edit to an old function doesn't resurface old similarities
    Units with zero candidates after filtering are dropped from the output.

    Args:
        units: Output of Node 1 (ingest_diff)
        store: VectorStore implementation (MockVectorStore for dev)
        embed_fn: Embedding function (mock_embed_text for dev)
        similarity_floor: Minimum similarity to keep a candidate
        ignore_paths: Candidate paths to drop (None = tests, see review_filters)
        base_code: unit_id -> code at the base commit (from Node 1)

    Returns:
        List of (unit, filtered_candidates) tuples, only for units
        that have at least one candidate above the floor.
    """
    import concurrent.futures

    results: List[Tuple[ChangedCodeUnit, List[Candidate]]] = []
    base_code = base_code or {}

    # Texts to embed: every unit, plus the base version of modified units.
    texts: List[str] = [u.code for u in units]
    base_index: Dict[str, int] = {}
    for u in units:
        if u.unit_id in base_code:
            base_index[u.unit_id] = len(texts)
            texts.append(base_code[u.unit_id])

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

        raw_candidates = store.query(embedding)

        preexisting = set()
        if unit.unit_id in base_index and embeddings[base_index[unit.unit_id]]:
            old_candidates = store.query(embeddings[base_index[unit.unit_id]])
            preexisting = {c.chunk_id for c in old_candidates if c.similarity >= similarity_floor}

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
            if c.chunk_id in preexisting:
                dropped["preexisting"] += 1
                continue
            if c.source_type == "code":
                if is_review_ignored(c.file_path, ignore_paths):
                    dropped["ignored_path"] += 1
                    continue
                if is_caller_or_wrapper(unit.code, unit.symbol_name, c.code_snippet, c.symbol_name):
                    dropped["caller_or_wrapper"] += 1
                    continue
            filtered.append(c)

        logger.info(
            "Node 2: %s::%s -> %d raw candidates, %d kept (floor %.2f; dropped %s)",
            unit.file_path,
            unit.symbol_name,
            len(raw_candidates),
            len(filtered),
            similarity_floor,
            ", ".join(f"{k}={v}" for k, v in dropped.items() if v) or "none",
        )

        if filtered:
            results.append((unit, filtered))

    logger.info("Node 2: %d unit(s) with matches out of %d total", len(results), len(units))
    return results
