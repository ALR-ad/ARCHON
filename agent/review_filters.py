"""
agent/review_filters.py

Noise filters shared by Node 1 and Node 2. Each one removes a class of
false positive seen in real runs:
  - tests flagged as duplicates of other tests (they are meant to look alike)
  - a function flagged as duplicating its own caller or a thin wrapper of it
"""

import re
from typing import List, Optional

from indexing.ingest.repo_source import matches_any_pattern

# Paths excluded from review (both as PR units and as candidates) unless
# .agent-review.yml sets review_ignore_paths. Set it to [] to review tests too.
DEFAULT_REVIEW_IGNORE_PATHS: List[str] = [
    "tests/**",
    "test/**",
    "**/tests/**",
    "**/test/**",
    "**/__tests__/**",
    "test_*.py",
    "*_test.py",
    "*_test.go",
    "*.test.*",
    "*.spec.*",
    "conftest.py",
]


def is_review_ignored(file_path: str, patterns: Optional[List[str]] = None) -> bool:
    """True if file_path matches the review ignore patterns (defaults when None)."""
    if patterns is None:
        patterns = DEFAULT_REVIEW_IGNORE_PATHS
    return matches_any_pattern(file_path, patterns)


def _short_name(symbol_name: str) -> str:
    """'Class.method' -> 'method'."""
    return symbol_name.rsplit(".", 1)[-1]


def references_symbol(code: str, symbol_name: str) -> bool:
    """
    True if code calls, imports or assigns symbol_name, ignoring the
    symbol's own definition line. Plain mentions (e.g. in a docstring)
    don't count, so common words like "parse" don't over-match.
    """
    name = _short_name(symbol_name)
    if len(name) < 3 or not name.isidentifier():
        return False
    esc = re.escape(name)
    code = re.sub(rf"\b(def|function|func|fn|class)\s+{esc}\b", "", code)
    patterns = (
        rf"\b{esc}\s*\(",              # call: name(...)
        rf"\bimport\b[^\n]*\b{esc}\b",  # import / from x import name
        rf"=\s*{esc}\b",                # passed as a value: fn = name
    )
    return any(re.search(p, code) for p in patterns)


def is_caller_or_wrapper(unit_code: str, unit_symbol: str, cand_code: str, cand_symbol: str) -> bool:
    """
    True if the candidate uses the unit (caller / wrapper / delegating
    interface) or the unit uses the candidate. Either way the similarity
    comes from one depending on the other, not from duplicated logic.
    """
    if not cand_symbol:
        return False
    return references_symbol(cand_code, unit_symbol) or references_symbol(unit_code, cand_symbol)
