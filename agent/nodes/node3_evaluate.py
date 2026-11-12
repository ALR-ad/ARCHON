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


def _cited_rule(
    violated_rule: Any,
    candidates: List[Candidate],
    rule_candidates: List[Candidate],
) -> Optional[Candidate]:
    """
    Resolve the model's violated_rule (1-based number in the candidate list)
    to a wiki candidate. Returns None if it doesn't point at a wiki rule,
    unless there is only one rule it could have meant.
    """
    try:
        index = int(violated_rule)
    except (TypeError, ValueError):
        index = 0
    if 1 <= index <= len(candidates) and candidates[index - 1].source_type == "wiki":
        return candidates[index - 1]
    if len(rule_candidates) == 1:
        return rule_candidates[0]
    return None


def evaluate(
    matches: List[Tuple[ChangedCodeUnit, List[Candidate]]],
    eval_fn: Optional[Callable[[ChangedCodeUnit, List[Candidate]], Optional[Dict[str, Any]]]] = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    enabled_checks: Optional[Dict[str, bool]] = None,
    confirm_fn: Optional[Callable[[ChangedCodeUnit, Candidate], bool]] = None,
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
        enabled_checks: From .agent-review.yml, e.g.
                 {"duplication": True, "architectural_violation": False}.
                 A disabled check is never reported. None enables both.
        confirm_fn: Optional second opinion on a rule violation, called with
                 the unit and the cited rule. The violation is dropped
                 unless it returns True. None skips the second check.

    Returns:
        List[Finding] with only high-confidence results.
    """
    if eval_fn is None:
        eval_fn = evaluate_with_ollama
    checks = enabled_checks or {}
    check_duplication = bool(checks.get("duplication", True))
    check_architecture = bool(checks.get("architectural_violation", True))

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
        code_candidates = [c for c in candidates if c.source_type == "code"]
        rule_candidates = [c for c in candidates if c.source_type == "wiki"]

        # Duplication only counts against existing code
        is_dup = check_duplication and bool(result.get("is_duplicate", False)) and bool(code_candidates)

        # A violation only counts if the model names the wiki rule it breaks
        broken_rule = _cited_rule(result.get("violated_rule"), candidates, rule_candidates)
        is_arch = (
            check_architecture
            and bool(result.get("is_architectural_violation", False))
            and broken_rule is not None
        )

        # Second opinion, only for violations that would otherwise be reported
        if is_arch and confidence >= confidence_threshold and confirm_fn is not None:
            if not confirm_fn(unit, broken_rule):
                logger.info(
                    "Node 3: %s::%s - rule violation not confirmed, dropping it",
                    unit.file_path,
                    unit.symbol_name,
                )
                is_arch = False

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

        # Show only what the finding rests on: the matched code and/or the broken rule
        evidence = (code_candidates if is_dup else []) + ([broken_rule] if is_arch else [])

        finding = Finding(
            finding_id=_make_finding_id(unit.unit_id, [c.chunk_id for c in evidence]),
            unit=unit,
            candidates=evidence,
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
