"""
PERSON B: Node 2.
Input:  List[ChangedCodeUnit]
Calls:  mock_embed_text() (stand-in for shared.interfaces.embed_text()),
        VectorStore.query() (via agent.llm.mock_vectorstore.MockVectorStore)
Output: List[Tuple[ChangedCodeUnit, List[Candidate]]] filtered by SIMILARITY_FLOOR

The real embed_text() is owned by Person A and raises NotImplementedError.
We use a local mock until Person A's implementation is ready to integrate.
"""

import logging
from typing import Callable, List, Tuple

from shared.schemas import ChangedCodeUnit, Candidate
from shared.interfaces import VectorStore
from shared.config import SIMILARITY_FLOOR, EMBEDDING_DIM

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
) -> List[Tuple[ChangedCodeUnit, List[Candidate]]]:
    """
    Node 2 entry point.

    For each ChangedCodeUnit, embed its code, query the vector store,
    and filter candidates below the similarity floor. Units with zero
    candidates after filtering are dropped from the output.

    Args:
        units: Output of Node 1 (ingest_diff)
        store: VectorStore implementation (MockVectorStore for dev)
        embed_fn: Embedding function (mock_embed_text for dev)
        similarity_floor: Minimum similarity to keep a candidate

    Returns:
        List of (unit, filtered_candidates) tuples, only for units
        that have at least one candidate above the floor.
    """
    import concurrent.futures

    results: List[Tuple[ChangedCodeUnit, List[Candidate]]] = []

    logger.info("Node 2: preparing to embed %d units", len(units))
    
    # 1. Generate embeddings concurrently to hide network/processing latency
    embeddings: List[List[float]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(units) or 1, 10)) as executor:
        # submit tasks in order
        future_to_index = {executor.submit(embed_fn, unit.code): i for i, unit in enumerate(units)}
        
        # We need embeddings in the same order as units. 
        # So we allocate a list and fill it.
        embeddings = [[]] * len(units)
        for future in concurrent.futures.as_completed(future_to_index):
            i = future_to_index[future]
            try:
                embeddings[i] = future.result()
            except Exception as exc:
                logger.error("Node 2: unit %d generated an exception: %s", i, exc)
                embeddings[i] = [] # Fallback for failure

    # 2. Query vector store sequentially (fast) and filter
    for i, unit in enumerate(units):
        embedding = embeddings[i]
        if not embedding:
            continue
            
        raw_candidates = store.query(embedding)

        # Filter by similarity floor
        filtered = [c for c in raw_candidates if c.similarity >= similarity_floor]

        logger.info(
            "Node 2: %s::%s -> %d raw candidates, %d above floor (%.2f)",
            unit.file_path,
            unit.symbol_name,
            len(raw_candidates),
            len(filtered),
            similarity_floor,
        )

        if filtered:
            results.append((unit, filtered))

    logger.info("Node 2: %d unit(s) with matches out of %d total", len(results), len(units))
    return results
