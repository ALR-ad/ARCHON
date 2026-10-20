"""
Tests for agent/nodes/node1_ingest_diff.py

Verifies:
1. ingest_diff() returns a list of ChangedCodeUnit objects
2. Each object has the correct shape/types from shared/schemas.py
3. unit_id is deterministic (same input -> same id)
4. Metadata extraction handles the expected webhook payload shape
5. Invalid payloads raise KeyError (fail-fast, no silent corruption)
"""

import pytest
from unittest.mock import MagicMock

from agent.nodes.node1_ingest_diff import (
    ingest_diff,
    _extract_pr_metadata,
    _make_unit_id,
)
from shared.schemas import ChangedCodeUnit


# -- Fixtures ----------------------------------------------------------------

FAKE_WEBHOOK_PAYLOAD = {
    "action": "opened",
    "number": 42,
    "pull_request": {
        "number": 42,
        "head": {"sha": "abc123def456abc123def456abc123def456abc1"},
        "base": {"sha": "000111222333444555666777888999aaabbbcccd"},
    },
    "repository": {
        "owner": {"login": "test-org"},
        "name": "test-repo",
        "full_name": "test-org/test-repo",
    },
}

def mock_get_pr_files(repo_full_name, pr_number):
    file1 = MagicMock()
    file1.status = "added"
    file1.filename = "src/parse.py"
    file1.patch = "@@ -1,0 +1,3 @@\n+def parse_date(raw: str):\n+    return None\n+"
    
    file2 = MagicMock()
    file2.status = "modified"
    file2.filename = "src/validate.py"
    file2.patch = "@@ -5,2 +5,2 @@\n def validate_order(order):\n-    return False\n+    return True\n"
    
    return [file1, file2]

def mock_get_file_content(repo_full_name, file_path, ref):
    if file_path == "src/parse.py":
        return "def parse_date(raw: str):\n    return None\n"
    elif file_path == "src/validate.py":
        return "\n\n\n\ndef validate_order(order):\n    return True\n"
    return ""

# -- Tests: metadata extraction ----------------------------------------------

def test_extract_pr_metadata_fields():
    meta = _extract_pr_metadata(FAKE_WEBHOOK_PAYLOAD)
    assert meta["action"] == "opened"
    assert meta["pr_number"] == 42
    assert meta["head_sha"] == "abc123def456abc123def456abc123def456abc1"
    assert meta["base_sha"] == "000111222333444555666777888999aaabbbcccd"
    assert meta["repo_owner"] == "test-org"
    assert meta["repo_name"] == "test-repo"
    assert meta["repo_full_name"] == "test-org/test-repo"


def test_extract_pr_metadata_missing_full_name():
    """full_name is optional — should be synthesized from owner+name."""
    payload = {
        "action": "opened",
        "pull_request": {
            "number": 1,
            "head": {"sha": "aaa"},
            "base": {"sha": "bbb"},
        },
        "repository": {
            "owner": {"login": "org"},
            "name": "repo",
        },
    }
    meta = _extract_pr_metadata(payload)
    assert meta["repo_full_name"] == "org/repo"


def test_extract_pr_metadata_missing_required_field():
    """Missing 'pull_request' key should raise KeyError, not silently fail."""
    with pytest.raises(KeyError):
        _extract_pr_metadata({"action": "opened", "repository": {}})


# -- Tests: unit_id determinism -----------------------------------------------

def test_unit_id_is_deterministic():
    id1 = _make_unit_id("src/foo.ts", "bar", "sha123")
    id2 = _make_unit_id("src/foo.ts", "bar", "sha123")
    assert id1 == id2


def test_unit_id_changes_with_different_sha():
    id1 = _make_unit_id("src/foo.ts", "bar", "sha_a")
    id2 = _make_unit_id("src/foo.ts", "bar", "sha_b")
    assert id1 != id2


# -- Tests: ingest_diff output shape ------------------------------------------

def test_ingest_diff_returns_list_of_changed_code_units():
    result = ingest_diff(
        FAKE_WEBHOOK_PAYLOAD,
        get_pr_files_fn=mock_get_pr_files,
        get_file_content_fn=mock_get_file_content
    )
    assert isinstance(result, list)
    assert len(result) > 0
    for unit in result:
        assert isinstance(unit, ChangedCodeUnit)


def test_ingest_diff_unit_fields_are_populated():
    result = ingest_diff(
        FAKE_WEBHOOK_PAYLOAD,
        get_pr_files_fn=mock_get_pr_files,
        get_file_content_fn=mock_get_file_content
    )
    unit = result[0]
    assert unit.unit_id  # non-empty
    assert unit.file_path  # non-empty
    assert unit.symbol_name  # non-empty
    assert unit.language  # non-empty
    assert unit.start_line >= 1
    assert unit.end_line >= unit.start_line
    assert unit.code  # non-empty
    assert unit.diff_type in ("added", "modified")


def test_ingest_diff_contains_both_diff_types():
    """The mock data includes both 'added' and 'modified' units."""
    result = ingest_diff(
        FAKE_WEBHOOK_PAYLOAD,
        get_pr_files_fn=mock_get_pr_files,
        get_file_content_fn=mock_get_file_content
    )
    diff_types = {u.diff_type for u in result}
    assert "added" in diff_types
    assert "modified" in diff_types


def test_ingest_diff_unit_ids_are_unique():
    result = ingest_diff(
        FAKE_WEBHOOK_PAYLOAD,
        get_pr_files_fn=mock_get_pr_files,
        get_file_content_fn=mock_get_file_content
    )
    ids = [u.unit_id for u in result]
    assert len(ids) == len(set(ids)), "unit_ids must be unique"


# -- Integration Test ---------------------------------------------------------

import os

@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION_TESTS"),
    reason="Skipping integration test by default. Set RUN_INTEGRATION_TESTS=1 to run."
)
def test_ingest_diff_real_github_api():
    """Hits the real GitHub API. Run only when requested."""
    payload = {
        "action": "opened",
        "number": 1,
        "pull_request": {
            "number": 1,
            "head": {"sha": "HEAD"},
            "base": {"sha": "BASE"},
        },
        "repository": {
            "owner": {"login": "ALR-ad"},
            "name": "ARCHON",
            "full_name": "ALR-ad/ARCHON",
        },
    }
    # This will hit real get_pr_files/get_file_content. We expect it to 
    # either return valid units or fail gracefully if PR #1 doesn't exist
    # or lacks modified lines, but it should not crash.
    # Note: depends on a valid GITHUB_TOKEN in env.
    result = ingest_diff(payload)
    assert isinstance(result, list)
