"""
Tests for agent/llm/ollama_eval.py and agent/nodes/node3_evaluate.py

Verifies:
1. Prompt construction is correct
2. JSON parsing handles clean JSON, markdown-fenced JSON, and bad JSON
3. fake_evaluate returns valid shape
4. Node 3 evaluate() produces Finding objects with correct filtering
5. Confidence threshold filtering works
6. None results from eval_fn are handled gracefully
"""

import json

import pytest

from agent.llm.ollama_eval import (
    _build_prompt,
    _format_candidates,
    _parse_llm_response,
    fake_evaluate,
)
from agent.nodes.node3_evaluate import evaluate, _make_finding_id
from shared.schemas import Candidate, ChangedCodeUnit, Finding
from shared.config import CONFIDENCE_THRESHOLD


# -- Fixtures ----------------------------------------------------------------

def _make_unit() -> ChangedCodeUnit:
    return ChangedCodeUnit(
        unit_id="unit_abc123",
        file_path="src/orders/parseOrderDate.ts",
        symbol_name="parseOrderDate",
        language="typescript",
        start_line=12,
        end_line=28,
        code="export function parseOrderDate(raw: string): Date { /* ... */ }",
        diff_type="added",
    )


def _make_candidate() -> Candidate:
    return Candidate(
        chunk_id="chunk_xyz789",
        file_path="src/shared/formatDateUtils.ts",
        symbol_name="parse",
        code_snippet="export function parse(raw: string): Date { /* ... */ }",
        source_type="code",
        similarity=0.86,
    )


# -- Tests: prompt construction -----------------------------------------------

def test_build_prompt_contains_unit_info():
    prompt = _build_prompt(_make_unit(), [_make_candidate()])
    assert "parseOrderDate" in prompt
    assert "src/orders/parseOrderDate.ts" in prompt
    assert "typescript" in prompt


def test_format_candidates_labels_source_type():
    code_candidate = _make_candidate()
    wiki_candidate = Candidate(
        chunk_id="wiki_1",
        file_path="wiki/style-guide.md",
        symbol_name="",
        code_snippet="Always use the shared date parser.",
        source_type="wiki",
        similarity=0.82,
    )
    text = _format_candidates([code_candidate, wiki_candidate])
    assert "CODEBASE" in text
    assert "WIKI RULE" in text


# -- Tests: JSON parsing -----------------------------------------------------

def test_parse_clean_json():
    raw = '{"is_duplicate": true, "confidence": 0.9, "reasoning": "test"}'
    result = _parse_llm_response(raw)
    assert result["is_duplicate"] is True
    assert result["confidence"] == 0.9


def test_parse_markdown_fenced_json():
    raw = '```json\n{"is_duplicate": false, "confidence": 0.5}\n```'
    result = _parse_llm_response(raw)
    assert result["is_duplicate"] is False


def test_parse_bad_json_raises():
    with pytest.raises(json.JSONDecodeError):
        _parse_llm_response("not json at all")


# -- Tests: fake_evaluate ----------------------------------------------------

def test_fake_evaluate_returns_valid_shape():
    result = fake_evaluate(_make_unit(), [_make_candidate()])
    assert "is_duplicate" in result
    assert "is_architectural_violation" in result
    assert "confidence" in result
    assert "reasoning" in result
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0


# -- Tests: finding_id -------------------------------------------------------

def test_finding_id_is_deterministic():
    id1 = _make_finding_id("unit1", ["chunk_a", "chunk_b"])
    id2 = _make_finding_id("unit1", ["chunk_a", "chunk_b"])
    assert id1 == id2


def test_finding_id_order_independent():
    """chunk_ids are sorted internally, so order shouldn't matter."""
    id1 = _make_finding_id("unit1", ["chunk_b", "chunk_a"])
    id2 = _make_finding_id("unit1", ["chunk_a", "chunk_b"])
    assert id1 == id2


# -- Tests: Node 3 evaluate() ------------------------------------------------

def test_evaluate_with_fake_produces_findings():
    unit = _make_unit()
    candidates = [_make_candidate()]
    matches = [(unit, candidates)]

    findings = evaluate(matches, eval_fn=fake_evaluate)
    assert len(findings) == 1
    assert isinstance(findings[0], Finding)


def test_evaluate_finding_fields_populated():
    unit = _make_unit()
    candidates = [_make_candidate()]
    findings = evaluate([(unit, candidates)], eval_fn=fake_evaluate)

    f = findings[0]
    assert f.finding_id  # non-empty
    assert f.unit == unit
    assert f.candidates == candidates
    assert f.is_duplicate is True
    assert isinstance(f.confidence, float)
    assert f.reasoning  # non-empty
    assert f.suggested_replacement is not None


def test_evaluate_filters_by_confidence():
    """With a high threshold, the fake finding (0.88) should be excluded."""
    unit = _make_unit()
    candidates = [_make_candidate()]

    findings = evaluate(
        [(unit, candidates)],
        eval_fn=fake_evaluate,
        confidence_threshold=0.95,
    )
    assert len(findings) == 0


def test_evaluate_keeps_findings_above_threshold():
    """With a low threshold, the fake finding (0.88) should be kept."""
    unit = _make_unit()
    candidates = [_make_candidate()]

    findings = evaluate(
        [(unit, candidates)],
        eval_fn=fake_evaluate,
        confidence_threshold=0.50,
    )
    assert len(findings) == 1


def test_evaluate_skips_none_results():
    """If eval_fn returns None, the match is skipped (no crash)."""
    unit = _make_unit()
    candidates = [_make_candidate()]

    def always_none(u, c):
        return None

    findings = evaluate([(unit, candidates)], eval_fn=always_none)
    assert len(findings) == 0


def test_evaluate_skips_non_issues():
    """If neither duplicate nor violation, finding is skipped."""
    unit = _make_unit()
    candidates = [_make_candidate()]

    def no_issues(u, c):
        return {
            "is_duplicate": False,
            "is_architectural_violation": False,
            "confidence": 0.90,
            "reasoning": "Nothing wrong here.",
        }

    findings = evaluate([(unit, candidates)], eval_fn=no_issues)
    assert len(findings) == 0


def test_evaluate_multiple_matches():
    unit1 = _make_unit()
    unit2 = ChangedCodeUnit(
        unit_id="unit_def456",
        file_path="src/orders/validateOrder.ts",
        symbol_name="validateOrderPayload",
        language="typescript",
        start_line=5,
        end_line=22,
        code="export function validateOrderPayload(order: Order): boolean { /* ... */ }",
        diff_type="modified",
    )
    candidates = [_make_candidate()]
    matches = [(unit1, candidates), (unit2, candidates)]

    findings = evaluate(matches, eval_fn=fake_evaluate)
    assert len(findings) == 2
    # Each finding should reference its own unit
    assert findings[0].unit.symbol_name == "parseOrderDate"
    assert findings[1].unit.symbol_name == "validateOrderPayload"


def test_evaluate_drops_disabled_duplication_check():
    """enabled_checks.duplication=false means a duplicate-only result is not reported."""
    findings = evaluate(
        [(_make_unit(), [_make_candidate()])],
        eval_fn=fake_evaluate,
        enabled_checks={"duplication": False, "architectural_violation": True},
    )
    assert findings == []


def test_evaluate_keeps_enabled_check_when_other_disabled():
    def both_issues(unit, candidates):
        return {
            "is_duplicate": True,
            "is_architectural_violation": True,
            "violated_rule": 2,
            "confidence": 0.9,
            "reasoning": "Both.",
        }

    findings = evaluate(
        [(_make_unit(), [_make_candidate(), _make_rule()])],
        eval_fn=both_issues,
        enabled_checks={"duplication": False},
    )
    assert len(findings) == 1
    assert findings[0].is_duplicate is False
    assert findings[0].is_architectural_violation is True


# -- Tests: architectural violations need a cited wiki rule -------------------

def _make_rule(chunk_id: str = "rule_001", text: str = "[Data access] Never run raw SQL in handlers.") -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        file_path="docs/ARCHITECTURE.md",
        symbol_name="",
        code_snippet=text,
        source_type="wiki",
        similarity=0.6,
    )


def _verdict(**overrides):
    result = {
        "is_duplicate": False,
        "is_architectural_violation": True,
        "violated_rule": None,
        "confidence": 0.9,
        "reasoning": "Breaks the rule.",
    }
    result.update(overrides)
    return lambda unit, candidates: result


def test_violation_reports_only_the_cited_rule():
    rules = [_make_rule("rule_a"), _make_rule("rule_b", "[HTTP] Use the shared client.")]
    candidates = [_make_candidate()] + rules
    findings = evaluate([(_make_unit(), candidates)], eval_fn=_verdict(violated_rule=3))
    assert len(findings) == 1
    assert findings[0].is_architectural_violation is True
    assert findings[0].is_duplicate is False
    assert [c.chunk_id for c in findings[0].candidates] == ["rule_b"]


def test_violation_without_any_rule_candidate_is_dropped():
    """The model can't claim a violation when no wiki rule was shown to it."""
    findings = evaluate([(_make_unit(), [_make_candidate()])], eval_fn=_verdict(violated_rule=1))
    assert findings == []


def test_violation_citing_a_code_candidate_is_dropped_when_rules_are_ambiguous():
    candidates = [_make_candidate(), _make_rule("rule_a"), _make_rule("rule_b")]
    findings = evaluate([(_make_unit(), candidates)], eval_fn=_verdict(violated_rule=1))
    assert findings == []


def test_violation_with_missing_citation_uses_the_only_rule():
    candidates = [_make_candidate(), _make_rule("rule_a")]
    findings = evaluate([(_make_unit(), candidates)], eval_fn=_verdict(violated_rule=None))
    assert [c.chunk_id for c in findings[0].candidates] == ["rule_a"]


def test_duplicate_claim_with_only_rules_is_dropped():
    """Duplication is judged against code only; rules can't be 'duplicated'."""
    findings = evaluate(
        [(_make_unit(), [_make_rule()])],
        eval_fn=_verdict(is_duplicate=True, is_architectural_violation=False),
    )
    assert findings == []


def test_duplicate_and_violation_keep_code_and_rule():
    candidates = [_make_candidate(), _make_rule("rule_a")]
    findings = evaluate([(_make_unit(), candidates)], eval_fn=_verdict(is_duplicate=True, violated_rule=2))
    assert [c.source_type for c in findings[0].candidates] == ["code", "wiki"]


# -- Tests: second opinion on rule violations ---------------------------------

def test_unconfirmed_violation_is_dropped():
    candidates = [_make_rule("rule_a")]
    findings = evaluate([(_make_unit(), candidates)], eval_fn=_verdict(violated_rule=1),
                        confirm_fn=lambda unit, rule: False)
    assert findings == []


def test_confirmed_violation_is_kept_and_checked_against_the_cited_rule():
    seen = []
    def confirm(unit, rule):
        seen.append(rule.chunk_id)
        return True

    candidates = [_make_rule("rule_a"), _make_rule("rule_b")]
    findings = evaluate([(_make_unit(), candidates)], eval_fn=_verdict(violated_rule=2), confirm_fn=confirm)
    assert len(findings) == 1
    assert seen == ["rule_b"]


def test_unconfirmed_violation_keeps_a_real_duplicate():
    candidates = [_make_candidate(), _make_rule("rule_a")]
    findings = evaluate([(_make_unit(), candidates)],
                        eval_fn=_verdict(is_duplicate=True, violated_rule=2),
                        confirm_fn=lambda unit, rule: False)
    assert len(findings) == 1
    assert findings[0].is_duplicate is True
    assert findings[0].is_architectural_violation is False
    assert [c.source_type for c in findings[0].candidates] == ["code"]


def test_confirm_not_called_for_duplicates_or_low_confidence():
    def must_not_run(unit, rule):
        raise AssertionError("confirm_fn should not be called")

    evaluate([(_make_unit(), [_make_candidate()])], eval_fn=fake_evaluate, confirm_fn=must_not_run)
    evaluate([(_make_unit(), [_make_rule()])], eval_fn=_verdict(violated_rule=1, confidence=0.2),
             confirm_fn=must_not_run)


def test_quote_must_really_appear_in_the_code():
    from agent.llm.ollama_eval import _quote_appears_in

    code = 'def f(currency):\n    resp = requests.get(f"https://x/{currency}")\n    return resp.json()'
    assert _quote_appears_in("requests.get(f'https://x/{currency}')", code)      # quote style changed
    assert _quote_appears_in("  resp   =  requests.get(", code)                  # re-spaced fragment
    assert _quote_appears_in("made_up_call()\nreturn resp.json()", code)         # one of two lines is real
    assert not _quote_appears_in("conn.execute(sql)", code)                      # invented
    assert not _quote_appears_in("", code)
