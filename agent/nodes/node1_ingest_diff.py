"""
PERSON B: Node 1.
Input:  GitHub webhook payload (PR number, repo, head_sha, base_sha)
Output: List[ChangedCodeUnit]  (see shared/schemas.py)

For now uses hardcoded fake diff data instead of calling the real GitHub API.
The real GitHub API integration lives in agent/github_client.py and will be
wired in once Task 5 is complete.
"""

import hashlib
import logging
from typing import Any, Dict, List

from shared.schemas import ChangedCodeUnit

logger = logging.getLogger("agent.nodes.node1")


def _make_unit_id(file_path: str, symbol_name: str, pr_sha: str) -> str:
    """Deterministic, stable ID for a changed code unit."""
    raw = f"{file_path}:{symbol_name}:{pr_sha}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _extract_pr_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pull the fields we care about out of a GitHub pull_request webhook payload.

    Expected top-level shape (we only read what we need):
        {
            "action": "opened" | "synchronize" | ...,
            "number": 42,
            "pull_request": {
                "number": 42,
                "head": {"sha": "abc123..."},
                "base": {"sha": "def456..."}
            },
            "repository": {
                "owner": {"login": "some-org"},
                "name": "some-repo",
                "full_name": "some-org/some-repo"
            }
        }
    """
    pr = payload["pull_request"]
    repo = payload["repository"]
    return {
        "action": payload.get("action", "unknown"),
        "pr_number": pr["number"],
        "head_sha": pr["head"]["sha"],
        "base_sha": pr["base"]["sha"],
        "repo_owner": repo["owner"]["login"],
        "repo_name": repo["name"],
        "repo_full_name": repo.get("full_name", f"{repo['owner']['login']}/{repo['name']}"),
    }


# ---------------------------------------------------------------------------
# Fake diff data for development -- will be replaced by real GitHub API call
# ---------------------------------------------------------------------------
_FAKE_DIFF_UNITS = [
    {
        "file_path": "src/orders/parseOrderDate.ts",
        "symbol_name": "parseOrderDate",
        "language": "typescript",
        "start_line": 12,
        "end_line": 28,
        "code": (
            "export function parseOrderDate(raw: string): Date {\n"
            "  const parts = raw.split('-');\n"
            "  const year = parseInt(parts[0], 10);\n"
            "  const month = parseInt(parts[1], 10) - 1;\n"
            "  const day = parseInt(parts[2], 10);\n"
            "  return new Date(year, month, day);\n"
            "}"
        ),
        "diff_type": "added",
    },
    {
        "file_path": "src/orders/validateOrder.ts",
        "symbol_name": "validateOrderPayload",
        "language": "typescript",
        "start_line": 5,
        "end_line": 22,
        "code": (
            "export function validateOrderPayload(order: Order): boolean {\n"
            "  if (!order.id || typeof order.id !== 'string') return false;\n"
            "  if (!order.items || order.items.length === 0) return false;\n"
            "  if (order.total <= 0) return false;\n"
            "  return true;\n"
            "}"
        ),
        "diff_type": "modified",
    },
]


def ingest_diff(payload: Dict[str, Any]) -> List[ChangedCodeUnit]:
    """
    Node 1 entry point.

    Takes a raw GitHub webhook payload (pull_request event), extracts
    metadata, fetches the diff (currently faked), and returns structured
    ChangedCodeUnit objects.
    """
    meta = _extract_pr_metadata(payload)
    logger.info(
        "Node 1: ingesting PR #%s on %s (head=%s, base=%s)",
        meta["pr_number"],
        meta["repo_full_name"],
        meta["head_sha"][:8],
        meta["base_sha"][:8],
    )

    # TODO: replace with real diff fetching via agent.github_client
    units: List[ChangedCodeUnit] = []
    for raw in _FAKE_DIFF_UNITS:
        unit = ChangedCodeUnit(
            unit_id=_make_unit_id(raw["file_path"], raw["symbol_name"], meta["head_sha"]),
            file_path=raw["file_path"],
            symbol_name=raw["symbol_name"],
            language=raw["language"],
            start_line=raw["start_line"],
            end_line=raw["end_line"],
            code=raw["code"],
            diff_type=raw["diff_type"],
        )
        units.append(unit)

    logger.info("Node 1: produced %d ChangedCodeUnit(s)", len(units))
    return units
