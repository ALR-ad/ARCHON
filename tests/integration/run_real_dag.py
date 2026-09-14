"""
tests/integration/run_real_dag.py

Integration entry point that wires the DAG with REAL implementations:
  - QdrantVectorStore from indexing/vectorstore/store.py
  - embed_text() from indexing/embeddings/embedder.py
  - evaluate_with_ollama from agent/llm/ollama_eval.py

This does NOT modify agent/dag.py or any existing mock-based tests.
Instead, it re-uses the same Strands GraphBuilder pattern with new
wrapper functions that inject the real dependencies.

Usage:
    python -m tests.integration.run_real_dag \
        --qdrant-path ./data/qdrant_test \
        --payload '{"pull_request": {...}, "repository": {...}}'

Or import build_real_dag() and run_real_pipeline() from test code.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add project root to path if needed
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.schemas import Candidate, ChangedCodeUnit, Finding
from shared.interfaces import VectorStore

# --- Node functions (same as agent/nodes/) ---
from agent.nodes.node1_ingest_diff import ingest_diff
from agent.nodes.node2_semantic_search import semantic_search
from agent.nodes.node3_evaluate import evaluate
from agent.nodes.node4_comment import render_comment

# --- REAL implementations from Track A ---
from indexing.vectorstore.store import QdrantVectorStore
from indexing.embeddings.embedder import embed_text as real_embed_text

# --- REAL Ollama evaluator from Track B ---
from agent.llm.ollama_eval import evaluate_with_ollama

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("integration.run_real_dag")


# ---------------------------------------------------------------------------
# Functional pipeline (no Strands framework needed for integration testing)
# ---------------------------------------------------------------------------

def run_real_pipeline(
    payload: Dict[str, Any],
    store: VectorStore,
    embed_fn: Callable[[str], List[float]] = real_embed_text,
    eval_fn: Optional[Callable] = None,
    units_override: Optional[List[ChangedCodeUnit]] = None,
) -> Dict[str, Any]:
    """
    Run the full 4-node pipeline with real implementations.

    Args:
        payload: GitHub webhook-style payload for Node 1.
                 Ignored if units_override is provided.
        store: A real VectorStore instance (e.g. QdrantVectorStore).
        embed_fn: The real embed_text function (default: indexing's).
        eval_fn: Evaluation function (default: evaluate_with_ollama).
        units_override: If provided, skip Node 1 and use these units
                        directly. Useful for integration tests where we
                        construct the ChangedCodeUnit ourselves.

    Returns:
        Dict with keys: units, matches, findings, comment
    """
    if eval_fn is None:
        eval_fn = evaluate_with_ollama

    state: Dict[str, Any] = {}

    # --- Node 1: Ingest ---
    if units_override is not None:
        state["units"] = units_override
        logger.info("Node 1: SKIPPED (using %d override units)", len(units_override))
    else:
        state["units"] = ingest_diff(payload)
        logger.info("Node 1: produced %d units", len(state["units"]))

    # --- Node 2: Semantic Search (REAL store + REAL embeddings) ---
    state["matches"] = semantic_search(
        state["units"],
        store=store,
        embed_fn=embed_fn,
    )
    logger.info("Node 2: %d unit(s) with matches", len(state["matches"]))

    # --- Node 3: Evaluate (REAL Ollama) ---
    state["findings"] = evaluate(
        state["matches"],
        eval_fn=eval_fn,
    )
    logger.info("Node 3: %d finding(s)", len(state["findings"]))

    # --- Node 4: Render comment ---
    state["comment"] = render_comment(state["findings"])
    logger.info("Node 4: comment length = %d chars", len(state["comment"]))

    return state


# ---------------------------------------------------------------------------
# Convenience: construct a QdrantVectorStore and run
# ---------------------------------------------------------------------------

def run_with_qdrant(
    payload: Dict[str, Any],
    qdrant_path: str = "./data/qdrant",
    units_override: Optional[List[ChangedCodeUnit]] = None,
    eval_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper that constructs a QdrantVectorStore from a local
    path and runs the full pipeline.
    """
    store = QdrantVectorStore(path=qdrant_path)
    try:
        status = store.get_index_status()
        logger.info(
            "Vector store status: %d chunks, last SHA: %s",
            int(status.get("chunk_count", 0)),
            status.get("last_indexed_sha", "N/A"),
        )
        return run_real_pipeline(
            payload=payload,
            store=store,
            units_override=units_override,
            eval_fn=eval_fn,
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _default_payload() -> Dict[str, Any]:
    """A minimal valid payload for testing Node 1."""
    return {
        "action": "opened",
        "number": 999,
        "pull_request": {
            "number": 999,
            "head": {"sha": "abc123integration"},
            "base": {"sha": "def456integration"},
        },
        "repository": {
            "owner": {"login": "test-org"},
            "name": "test-repo",
            "full_name": "test-org/test-repo",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the real integration DAG")
    parser.add_argument(
        "--qdrant-path",
        default="./data/qdrant",
        help="Path to local Qdrant storage directory",
    )
    parser.add_argument(
        "--payload",
        default=None,
        help="JSON string with GitHub webhook payload (optional)",
    )
    args = parser.parse_args()

    payload = json.loads(args.payload) if args.payload else _default_payload()

    result = run_with_qdrant(
        payload=payload,
        qdrant_path=args.qdrant_path,
    )

    print("\n" + "=" * 60)
    print("INTEGRATION PIPELINE RESULTS")
    print("=" * 60)
    print(f"Units:    {len(result['units'])}")
    print(f"Matches:  {len(result['matches'])}")
    print(f"Findings: {len(result['findings'])}")
    print(f"Comment:  {len(result['comment'])} chars")

    if result["findings"]:
        print("\n--- Findings ---")
        for f in result["findings"]:
            print(f"  [{f.finding_id}] {f.unit.file_path}::{f.unit.symbol_name}")
            print(f"    dup={f.is_duplicate} arch={f.is_architectural_violation} conf={f.confidence:.2f}")
            print(f"    reasoning: {f.reasoning}")

    if result["comment"]:
        print("\n--- Rendered Comment ---")
        print(result["comment"])
    else:
        print("\n--- No comment rendered (no findings) ---")
