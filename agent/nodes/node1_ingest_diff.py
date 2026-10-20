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


import os
import re
from agent.github_client import get_pr_files, get_file_content
from indexing.chunking.code_chunker import chunk_code


def extract_modified_lines(patch: str) -> set[int]:
    """Parse unified diff to find added or modified line numbers."""
    modified_lines = set()
    if not patch:
        return modified_lines

    current_line = -1
    for line in patch.splitlines():
        if line.startswith("@@"):
            m = re.search(r"\+([0-9]+)", line)
            if m:
                current_line = int(m.group(1))
        elif line.startswith("+") and not line.startswith("+++"):
            if current_line != -1:
                modified_lines.add(current_line)
                current_line += 1
        elif line.startswith("-") and not line.startswith("---"):
            pass
        elif line.startswith(" ") or line == "":
            if current_line != -1:
                current_line += 1
    return modified_lines


def guess_language(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return ext[1:] if ext else "plaintext"


def ingest_diff(
    payload: Dict[str, Any],
    get_pr_files_fn=None,
    get_file_content_fn=None
) -> List[ChangedCodeUnit]:
    """
    Node 1 entry point.

    Takes a raw GitHub webhook payload (pull_request event), extracts
    metadata, fetches the real diff and file contents from GitHub,
    chunks the AST, and filters the chunks to those that were modified.
    """
    if get_pr_files_fn is None:
        from agent.github_client import get_pr_files
        get_pr_files_fn = get_pr_files
    if get_file_content_fn is None:
        from agent.github_client import get_file_content
        get_file_content_fn = get_file_content
    meta = _extract_pr_metadata(payload)
    logger.info(
        "Node 1: ingesting PR #%s on %s (head=%s, base=%s)",
        meta["pr_number"],
        meta["repo_full_name"],
        meta["head_sha"][:8],
        meta["base_sha"][:8],
    )

    units: List[ChangedCodeUnit] = []
    
    try:
        files = get_pr_files_fn(meta["repo_full_name"], meta["pr_number"])
    except Exception as e:
        logger.error("Failed to fetch PR files from GitHub: %s", e)
        return units

    for f in files:
        if f.status == "removed" or not f.patch:
            continue
            
        modified_lines = extract_modified_lines(f.patch)
        if not modified_lines:
            continue
            
        try:
            raw_content = get_file_content_fn(meta["repo_full_name"], f.filename, meta["head_sha"])
        except Exception as e:
            logger.warning("Failed to fetch content for %s: %s", f.filename, e)
            continue
            
        lang = guess_language(f.filename)
        
        try:
            chunks = chunk_code(
                file_path=f.filename,
                code=raw_content,
                language=lang,
                commit_sha=meta["head_sha"],
                diff_type="added" if f.status == "added" else "modified"
            )
        except Exception as e:
            logger.warning("Failed to chunk %s: %s", f.filename, e)
            continue
            
        # Keep chunks that overlap with modified lines
        for chunk in chunks:
            chunk_range = set(range(chunk.start_line, chunk.end_line + 1))
            if chunk_range.intersection(modified_lines):
                units.append(chunk)

    logger.info("Node 1: produced %d ChangedCodeUnit(s)", len(units))
    return units
