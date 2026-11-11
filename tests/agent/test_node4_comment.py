"""
Tests for agent/nodes/node4_comment.py

Verifies:
1. render_comment() returns empty string for no findings
2. Rendered markdown contains expected sections (header, file groups, badges)
3. Suggested-fix block is included when suggested_replacement exists
4. Suggested-fix block is absent when suggested_replacement is None
5. Hidden HTML comment with finding IDs is present
6. Findings are grouped by file path
7. BOT_COMMENT_MARKER is present for idempotent updates
"""

from agent.nodes.node4_comment import render_comment, BOT_COMMENT_MARKER
from shared.schemas import Candidate, ChangedCodeUnit, Finding


# -- Fixtures ----------------------------------------------------------------

def _make_unit(
    file_path: str = "src/orders/parseOrderDate.ts",
    symbol_name: str = "parseOrderDate",
) -> ChangedCodeUnit:
    return ChangedCodeUnit(
        unit_id="unit_001",
        file_path=file_path,
        symbol_name=symbol_name,
        language="typescript",
        start_line=12,
        end_line=28,
        code="export function parseOrderDate(raw: string): Date { /* ... */ }",
        diff_type="added",
    )


def _make_candidate() -> Candidate:
    return Candidate(
        chunk_id="chunk_001",
        file_path="src/shared/formatDateUtils.ts",
        symbol_name="parse",
        code_snippet="export function parse(raw: string): Date { /* ... */ }",
        source_type="code",
        similarity=0.86,
    )


def _make_finding(
    file_path: str = "src/orders/parseOrderDate.ts",
    symbol_name: str = "parseOrderDate",
    is_duplicate: bool = True,
    is_arch: bool = False,
    confidence: float = 0.88,
    suggested_replacement: str | None = "import { parse } from 'src/shared/formatDateUtils';",
    finding_id: str = "finding_001",
) -> Finding:
    return Finding(
        finding_id=finding_id,
        unit=_make_unit(file_path, symbol_name),
        candidates=[_make_candidate()],
        is_duplicate=is_duplicate,
        is_architectural_violation=is_arch,
        confidence=confidence,
        reasoning="This function duplicates logic found in formatDateUtils.ts.",
        suggested_replacement=suggested_replacement,
    )


# -- Tests: empty findings ---------------------------------------------------

def test_render_empty_findings():
    result = render_comment([])
    assert result == ""


# -- Tests: basic rendering ---------------------------------------------------

def test_render_contains_header():
    result = render_comment([_make_finding()])
    assert "Tech Debt & Architecture Review" in result
    assert "Found **1** issue" in result


def test_render_contains_file_path():
    result = render_comment([_make_finding()])
    assert "src/orders/parseOrderDate.ts" in result


def test_render_contains_symbol_name():
    result = render_comment([_make_finding()])
    assert "parseOrderDate" in result


def test_render_contains_reasoning():
    result = render_comment([_make_finding()])
    assert "duplicates logic" in result


def test_render_contains_confidence():
    result = render_comment([_make_finding(confidence=0.88)])
    assert "88%" in result


# -- Tests: badges -----------------------------------------------------------

def test_render_duplicate_badge():
    result = render_comment([_make_finding(is_duplicate=True, is_arch=False)])
    assert "Duplicated Logic" in result


def test_render_architectural_badge():
    result = render_comment([_make_finding(is_duplicate=False, is_arch=True)])
    assert "Architectural Violation" in result


def test_render_both_badges():
    result = render_comment([_make_finding(is_duplicate=True, is_arch=True)])
    assert "Duplicated Logic" in result
    assert "Architectural Violation" in result


# -- Tests: suggestion block --------------------------------------------------

def test_render_includes_suggestion_when_present():
    result = render_comment([_make_finding(suggested_replacement="import { parse }")])
    assert "**Suggested fix:**\n```typescript\nimport { parse }\n```" in result
    # ```suggestion only works in line-level review comments, not issue comments
    assert "```suggestion" not in result


def test_render_omits_suggestion_when_none():
    result = render_comment([_make_finding(suggested_replacement=None)])
    assert "Suggested fix" not in result


# -- Tests: similar code references -------------------------------------------

def test_render_contains_candidate_info():
    result = render_comment([_make_finding()])
    assert "formatDateUtils.ts" in result
    assert "86%" in result
    assert "Similar existing code" in result


# -- Tests: hidden HTML comment -----------------------------------------------

def test_render_contains_bot_marker():
    result = render_comment([_make_finding()])
    assert BOT_COMMENT_MARKER in result


def test_render_contains_finding_ids():
    result = render_comment([
        _make_finding(finding_id="id_aaa"),
        _make_finding(
            file_path="src/other.ts",
            symbol_name="other",
            finding_id="id_bbb",
        ),
    ])
    assert "id_aaa" in result
    assert "id_bbb" in result


# -- Tests: grouping by file -------------------------------------------------

def test_render_groups_by_file():
    findings = [
        _make_finding(file_path="src/a.ts", symbol_name="fn1", finding_id="f1"),
        _make_finding(file_path="src/a.ts", symbol_name="fn2", finding_id="f2"),
        _make_finding(file_path="src/b.ts", symbol_name="fn3", finding_id="f3"),
    ]
    result = render_comment(findings)
    # Should have file headers for both files
    assert "src/a.ts" in result
    assert "src/b.ts" in result
    # File a appears before file b (sorted)
    assert result.index("src/a.ts") < result.index("src/b.ts")


# -- Tests: summary counts ---------------------------------------------------

def test_render_summary_counts():
    findings = [
        _make_finding(is_duplicate=True, is_arch=False, finding_id="f1"),
        _make_finding(
            file_path="src/b.ts",
            is_duplicate=False,
            is_arch=True,
            finding_id="f2",
        ),
    ]
    result = render_comment(findings)
    assert "Found **2** issues" in result
    assert "1 duplicated logic" in result
    assert "1 architectural violation" in result


# -- Tests: size limits -------------------------------------------------------

def _big_candidate(i: int, lines: int = 400) -> Candidate:
    return Candidate(
        chunk_id=f"big_{i}",
        file_path=f"src/big_{i}.ts",
        symbol_name=f"big{i}",
        code_snippet="\n".join(f"const line{n} = {n};" for n in range(lines)),
        source_type="code",
        similarity=0.9,
    )


def test_long_snippets_are_clipped_and_stay_indented():
    from agent.nodes.node4_comment import MAX_SNIPPET_LINES

    finding = _make_finding()
    finding.candidates = [_big_candidate(0)]
    result = render_comment([finding])
    assert f"const line{MAX_SNIPPET_LINES - 1} = " in result
    assert f"const line{MAX_SNIPPET_LINES} = " not in result
    assert f"... ({400 - MAX_SNIPPET_LINES} more lines)" in result
    # Every snippet line is indented so the code block stays inside the list item
    assert "\n  const line1 = 1;" in result


def test_extra_candidates_are_summarised():
    from agent.nodes.node4_comment import MAX_CANDIDATES_SHOWN

    finding = _make_finding()
    finding.candidates = [_big_candidate(i, lines=2) for i in range(MAX_CANDIDATES_SHOWN + 2)]
    result = render_comment([finding])
    assert f"src/big_{MAX_CANDIDATES_SHOWN - 1}.ts" in result
    assert f"src/big_{MAX_CANDIDATES_SHOWN}.ts" not in result
    assert "...and 2 more" in result


def test_comment_stays_under_github_limit_and_keeps_most_confident():
    from agent.nodes.node4_comment import MAX_COMMENT_CHARS, MAX_FINDINGS

    findings = []
    for i in range(60):
        f = _make_finding(file_path=f"src/file_{i}.ts", finding_id=f"finding_{i:03d}",
                          confidence=0.99 if i == 59 else 0.80,
                          suggested_replacement="x" * 3000)
        findings.append(f)
    result = render_comment(findings)

    assert len(result) <= MAX_COMMENT_CHARS
    assert "Found **60** issues" in result            # header counts everything
    assert "src/file_59.ts" in result                 # highest confidence is shown
    shown = result.count("### 📁")
    assert 0 < shown <= MAX_FINDINGS
    assert f"_{60 - shown} lower-confidence findings not shown" in result
    assert BOT_COMMENT_MARKER in result
    assert "finding_000" in result and "finding_059" in result   # all IDs kept in the footer
