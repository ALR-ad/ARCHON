"""
PERSON B: Node 4.
Input:  List[Finding]
Output: Rendered markdown string (one consolidated PR comment)

The comment is grouped by file, with a suggestion block if
suggested_replacement exists, and a hidden HTML comment with finding IDs
for idempotent updates.

Does NOT post to GitHub — that's agent/github_client.py's job (Task 5).
"""

import logging
from collections import defaultdict
from typing import Dict, List

from shared.schemas import Finding

logger = logging.getLogger("agent.nodes.node4")

# Marker used to identify bot comments for idempotent updates
BOT_COMMENT_MARKER = "<!-- pr-reviewer-agent-findings -->"


def _render_finding(finding: Finding) -> str:
    """Render a single Finding into a markdown block."""
    parts: List[str] = []

    # Issue type badge
    labels: List[str] = []
    if finding.is_duplicate:
        labels.append("🔁 **Duplicated Logic**")
    if finding.is_architectural_violation:
        labels.append("🏗️ **Architectural Violation**")

    badge_line = " · ".join(labels)
    confidence_pct = int(finding.confidence * 100)
    parts.append(f"{badge_line} (confidence: {confidence_pct}%)")
    parts.append("")

    # Symbol and location
    parts.append(
        f"**`{finding.unit.symbol_name}`** "
        f"(lines {finding.unit.start_line}–{finding.unit.end_line})"
    )
    parts.append("")

    # Reasoning
    parts.append(f"> {finding.reasoning}")
    parts.append("")

    # Similar code references
    if finding.candidates:
        parts.append("<details>")
        parts.append(f"<summary>Similar existing code ({len(finding.candidates)} match{'es' if len(finding.candidates) != 1 else ''})</summary>")
        parts.append("")
        for c in finding.candidates:
            source_label = "📖 wiki" if c.source_type == "wiki" else "📄 code"
            parts.append(
                f"- `{c.file_path}::{c.symbol_name}` "
                f"({source_label}, similarity: {c.similarity:.0%})"
            )
            parts.append(f"  ```{finding.unit.language}")
            parts.append(f"  {c.code_snippet}")
            parts.append("  ```")
        parts.append("")
        parts.append("</details>")
        parts.append("")

    # Suggestion block
    if finding.suggested_replacement:
        parts.append("**Suggested fix:**")
        parts.append(f"```suggestion")
        parts.append(finding.suggested_replacement)
        parts.append("```")
        parts.append("")

    return "\n".join(parts)


def render_comment(findings: List[Finding]) -> str:
    """
    Node 4 entry point.

    Renders a List[Finding] into one consolidated markdown comment,
    grouped by file path.

    Returns:
        Rendered markdown string ready to be posted as a PR comment.
        Returns an empty string if there are no findings.
    """
    if not findings:
        logger.info("Node 4: no findings to render")
        return ""

    # Group findings by file path
    by_file: Dict[str, List[Finding]] = defaultdict(list)
    for f in findings:
        by_file[f.unit.file_path].append(f)

    sections: List[str] = []

    # Header
    total = len(findings)
    dup_count = sum(1 for f in findings if f.is_duplicate)
    arch_count = sum(1 for f in findings if f.is_architectural_violation)

    sections.append("## 🔍 Tech Debt & Architecture Review")
    sections.append("")
    sections.append(
        f"Found **{total}** issue{'s' if total != 1 else ''}: "
        f"{dup_count} duplicated logic, {arch_count} architectural violation{'s' if arch_count != 1 else ''}."
    )
    sections.append("")
    sections.append("---")
    sections.append("")

    # File sections
    for file_path in sorted(by_file.keys()):
        file_findings = by_file[file_path]
        sections.append(f"### 📁 `{file_path}`")
        sections.append("")

        for finding in file_findings:
            sections.append(_render_finding(finding))
            sections.append("---")
            sections.append("")

    # Hidden HTML comment with finding IDs for idempotent updates
    finding_ids = [f.finding_id for f in findings]
    sections.append(BOT_COMMENT_MARKER)
    sections.append(f"<!-- finding_ids: {','.join(finding_ids)} -->")

    result = "\n".join(sections)
    logger.info(
        "Node 4: rendered comment with %d finding(s) across %d file(s), %d chars",
        total,
        len(by_file),
        len(result),
    )
    return result
