"""
Tests for agent/github_client.py

All tests mock PyGithub — no real GitHub API calls are made.

Verifies:
1. fetch_pr_diff assembles diffs from PR file patches
2. find_existing_bot_comment detects the BOT_COMMENT_MARKER
3. post_or_update_comment creates new / edits existing comments
4. delete_bot_comment removes existing bot comments
5. Missing GITHUB_TOKEN raises EnvironmentError
"""

from unittest.mock import MagicMock, patch

import pytest

from agent.github_client import (
    _get_github_client,
    fetch_pr_diff,
    find_existing_bot_comment,
    post_or_update_comment,
    delete_bot_comment,
)
from agent.nodes.node4_comment import BOT_COMMENT_MARKER


# -- Helpers -----------------------------------------------------------------

def _mock_file(filename: str, patch_text: str):
    """Create a mock PR file object."""
    f = MagicMock()
    f.filename = filename
    f.patch = patch_text
    return f


def _mock_comment(comment_id: int, body: str):
    """Create a mock issue comment object."""
    c = MagicMock()
    c.id = comment_id
    c.body = body
    return c


# -- Tests: GITHUB_TOKEN validation ------------------------------------------

def test_get_github_client_raises_without_token():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(EnvironmentError, match="GITHUB_TOKEN"):
            _get_github_client()


# -- Tests: fetch_pr_diff ----------------------------------------------------

@patch("agent.github_client._get_github_client")
def test_fetch_pr_diff_assembles_patches(mock_get_client):
    mock_gh = MagicMock()
    mock_get_client.return_value = mock_gh

    mock_pr = MagicMock()
    mock_pr.base.ref = "main"
    mock_pr.head.ref = "feature"
    mock_pr.get_files.return_value = [
        _mock_file("src/a.ts", "@@ -1,3 +1,4 @@\n+added line"),
        _mock_file("src/b.ts", "@@ -5,2 +5,3 @@\n+another line"),
    ]
    mock_gh.get_repo.return_value.get_pull.return_value = mock_pr

    diff = fetch_pr_diff("org/repo", 42)
    assert "src/a.ts" in diff
    assert "src/b.ts" in diff
    assert "+added line" in diff
    assert "+another line" in diff


@patch("agent.github_client._get_github_client")
def test_fetch_pr_diff_skips_none_patches(mock_get_client):
    """Files without a patch (e.g. binary files) should be skipped."""
    mock_gh = MagicMock()
    mock_get_client.return_value = mock_gh

    mock_pr = MagicMock()
    mock_pr.base.ref = "main"
    mock_pr.head.ref = "feature"
    mock_pr.get_files.return_value = [
        _mock_file("image.png", None),
        _mock_file("src/a.ts", "@@ -1,1 +1,2 @@\n+code"),
    ]
    mock_gh.get_repo.return_value.get_pull.return_value = mock_pr

    diff = fetch_pr_diff("org/repo", 1)
    assert "image.png" not in diff
    assert "src/a.ts" in diff


# -- Tests: find_existing_bot_comment ----------------------------------------

@patch("agent.github_client._get_github_client")
def test_find_existing_bot_comment_found(mock_get_client):
    mock_gh = MagicMock()
    mock_get_client.return_value = mock_gh

    bot_comment = _mock_comment(999, f"Some review\n{BOT_COMMENT_MARKER}\n<!-- ids -->")
    human_comment = _mock_comment(100, "Looks good!")

    mock_gh.get_repo.return_value.get_issue.return_value.get_comments.return_value = [
        human_comment,
        bot_comment,
    ]

    result = find_existing_bot_comment("org/repo", 42)
    assert result == 999


@patch("agent.github_client._get_github_client")
def test_find_existing_bot_comment_not_found(mock_get_client):
    mock_gh = MagicMock()
    mock_get_client.return_value = mock_gh

    human_comment = _mock_comment(100, "Looks good!")
    mock_gh.get_repo.return_value.get_issue.return_value.get_comments.return_value = [
        human_comment,
    ]

    result = find_existing_bot_comment("org/repo", 42)
    assert result is None


# -- Tests: post_or_update_comment -------------------------------------------

@patch("agent.github_client.find_existing_bot_comment", return_value=None)
@patch("agent.github_client._get_github_client")
def test_post_creates_new_comment(mock_get_client, mock_find):
    mock_gh = MagicMock()
    mock_get_client.return_value = mock_gh

    new_comment = _mock_comment(555, "new comment body")
    mock_issue = mock_gh.get_repo.return_value.get_issue.return_value
    mock_issue.create_comment.return_value = new_comment

    result = post_or_update_comment("org/repo", 42, "review body")
    assert result == 555
    mock_issue.create_comment.assert_called_once_with("review body")


@patch("agent.github_client.find_existing_bot_comment", return_value=999)
@patch("agent.github_client._get_github_client")
def test_post_updates_existing_comment(mock_get_client, mock_find):
    mock_gh = MagicMock()
    mock_get_client.return_value = mock_gh

    existing = _mock_comment(999, "old body")
    mock_issue = mock_gh.get_repo.return_value.get_issue.return_value
    mock_issue.get_comment.return_value = existing

    result = post_or_update_comment("org/repo", 42, "updated body")
    assert result == 999
    existing.edit.assert_called_once_with("updated body")


# -- Tests: delete_bot_comment -----------------------------------------------

@patch("agent.github_client.find_existing_bot_comment", return_value=999)
@patch("agent.github_client._get_github_client")
def test_delete_removes_existing_comment(mock_get_client, mock_find):
    mock_gh = MagicMock()
    mock_get_client.return_value = mock_gh

    existing = _mock_comment(999, "bot comment")
    mock_issue = mock_gh.get_repo.return_value.get_issue.return_value
    mock_issue.get_comment.return_value = existing

    result = delete_bot_comment("org/repo", 42)
    assert result is True
    existing.delete.assert_called_once()


@patch("agent.github_client.find_existing_bot_comment", return_value=None)
@patch("agent.github_client._get_github_client")
def test_delete_returns_false_when_no_comment(mock_get_client, mock_find):
    mock_gh = MagicMock()
    mock_get_client.return_value = mock_gh

    result = delete_bot_comment("org/repo", 42)
    assert result is False
