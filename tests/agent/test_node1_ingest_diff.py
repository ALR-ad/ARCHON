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
    result = ingest_diff(FAKE_WEBHOOK_PAYLOAD)
    assert isinstance(result, list)
    assert len(result) > 0
    for unit in result:
        assert isinstance(unit, ChangedCodeUnit)


def test_ingest_diff_unit_fields_are_populated():
    result = ingest_diff(FAKE_WEBHOOK_PAYLOAD)
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
    """The fake data includes both 'added' and 'modified' units."""
    result = ingest_diff(FAKE_WEBHOOK_PAYLOAD)
    diff_types = {u.diff_type for u in result}
    assert "added" in diff_types
    assert "modified" in diff_types


def test_ingest_diff_unit_ids_are_unique():
    result = ingest_diff(FAKE_WEBHOOK_PAYLOAD)
    ids = [u.unit_id for u in result]
    assert len(ids) == len(set(ids)), "unit_ids must be unique"
