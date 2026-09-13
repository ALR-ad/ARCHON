"""
PERSON B: Node 3 (the only LLM node, runs via Ollama locally).
Input:  List[Tuple[ChangedCodeUnit, List[Candidate]]]
Output: List[Finding]  (see shared/schemas.py), filtered by CONFIDENCE_THRESHOLD

Calls agent.llm.ollama_eval to get the LLM's assessment, then constructs
Finding objects and filters by confidence threshold.
"""

import hashlib
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from shared.schemas import Candidate, ChangedCodeUnit, Finding
from shared.config import CONFIDENCE_THRESHOLD
from agent.llm.ollama_eval import evaluate_with_ollama, fake_evaluate

logger = logging.getLogger("agent.nodes.node3")


def _make_finding_id(unit_id: str, chunk_ids: List[str]) -> str:
    """Deterministic ID for a finding based on the unit and candidate chunks."""
    raw = f"{unit_id}:{'|'.join(sorted(chunk_ids))}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def evaluate(
    matches: List[Tuple[ChangedCodeUnit, List[Candidate]]],
    eval_fn: Optional[Callable[[ChangedCodeUnit, List[Candidate]], Optional[Dict[str, Any]]]] = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> List[Finding]:
    """
    Node 3 entry point.

    For each (unit, candidates) pair from Node 2, call the evaluation
    function (Ollama by default) and construct Finding objects.
    Findings below the confidence threshold are filtered out.

    Args:
        matches: Output of Node 2 (semantic_search)
        eval_fn: Evaluation function. Defaults to evaluate_with_ollama.
                 Use fake_evaluate for testing without Ollama.
        confidence_threshold: Minimum confidence to keep a finding.

    Returns:
        List[Finding] with only high-confidence results.
    """
    if eval_fn is None:
        eval_fn = evaluate_with_ollama

    findings: List[Finding] = []

    for unit, candidates in matches:
        logger.info(
            "Node 3: evaluating %s::%s against %d candidates",
            unit.file_path,
            unit.symbol_name,
            len(candidates),
        )

        result = eval_fn(unit, candidates)

        if result is None:
            logger.warning(
                "Node 3: evaluation returned None for %s::%s, skipping",
                unit.file_path,
                unit.symbol_name,
            )
            continue

        confidence = float(result.get("confidence", 0.0))
        is_dup = bool(result.get("is_duplicate", False))
        is_arch = bool(result.get("is_architectural_violation", False))

        # Skip if neither duplicate nor architectural violation
        if not is_dup and not is_arch:
            logger.info(
                "Node 3: %s::%s - no issues found, skipping",
                unit.file_path,
                unit.symbol_name,
            )
            continue

        # Skip if below confidence threshold
        if confidence < confidence_threshold:
            logger.info(
                "Node 3: %s::%s - confidence %.2f below threshold %.2f, skipping",
                unit.file_path,
                unit.symbol_name,
                confidence,
                confidence_threshold,
            )
            continue

        finding = Finding(
            finding_id=_make_finding_id(unit.unit_id, [c.chunk_id for c in candidates]),
            unit=unit,
            candidates=candidates,
            is_duplicate=is_dup,
            is_architectural_violation=is_arch,
            confidence=confidence,
            reasoning=result.get("reasoning", ""),
            suggested_replacement=result.get("suggested_replacement"),
        )

        logger.info(
            "Node 3: FINDING for %s::%s (dup=%s, arch=%s, conf=%.2f)",
            unit.file_path,
            unit.symbol_name,
            is_dup,
            is_arch,
            confidence,
        )
        findings.append(finding)

    logger.info(
        "Node 3: produced %d finding(s) from %d match(es)",
        len(findings),
        len(matches),
    )
    return findings
