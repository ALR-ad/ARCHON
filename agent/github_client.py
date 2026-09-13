"""
PERSON B: fetches PR diffs, posts/edits comments via GitHub API.

Uses PyGithub and reads GITHUB_TOKEN from the environment (.env file).
All functions are built but do NOT call the real API until explicitly
told to do so — the caller controls when real API calls happen.
"""

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from github import Github, GithubException

from agent.nodes.node4_comment import BOT_COMMENT_MARKER

logger = logging.getLogger("agent.github_client")

# Load .env into os.environ (no-op if already loaded)
load_dotenv()


def _get_github_client() -> Github:
    """Create an authenticated PyGithub client from GITHUB_TOKEN env var."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise EnvironmentError(
            "GITHUB_TOKEN not set. Add it to .env or export it."
        )
    return Github(token)


def fetch_pr_diff(
    repo_full_name: str,
    pr_number: int,
) -> str:
    """
    Fetch the unified diff for a pull request.

    Args:
        repo_full_name: e.g. "some-org/some-repo"
        pr_number: PR number

    Returns:
        The diff as a string (unified diff format).
    """
    gh = _get_github_client()
    repo = gh.get_repo(repo_full_name)
    pr = repo.get_pull(pr_number)

    logger.info(
        "Fetching diff for %s #%d (%s -> %s)",
        repo_full_name,
        pr_number,
        pr.base.ref,
        pr.head.ref,
    )

    # Get the list of files changed in the PR
    files = pr.get_files()
    diff_parts = []
    for f in files:
        if f.patch:
            diff_parts.append(
                f"--- a/{f.filename}\n"
                f"+++ b/{f.filename}\n"
                f"{f.patch}"
            )

    diff = "\n\n".join(diff_parts)
    logger.info("Fetched diff: %d file(s), %d chars", len(diff_parts), len(diff))
    return diff


def find_existing_bot_comment(
    repo_full_name: str,
    pr_number: int,
) -> Optional[int]:
    """
    Search for an existing bot comment on the PR (identified by
    BOT_COMMENT_MARKER in the comment body).

    Returns:
        The comment ID if found, None otherwise.
    """
    gh = _get_github_client()
    repo = gh.get_repo(repo_full_name)
    issue = repo.get_issue(pr_number)  # PRs are issues in the GitHub API

    for comment in issue.get_comments():
        if BOT_COMMENT_MARKER in (comment.body or ""):
            logger.info(
                "Found existing bot comment #%d on %s #%d",
                comment.id,
                repo_full_name,
                pr_number,
            )
            return comment.id

    logger.info("No existing bot comment on %s #%d", repo_full_name, pr_number)
    return None


def post_or_update_comment(
    repo_full_name: str,
    pr_number: int,
    comment_body: str,
) -> int:
    """
    Post a new comment or update the existing bot comment on a PR.

    If a comment with BOT_COMMENT_MARKER already exists, it is edited
    in place. Otherwise, a new comment is created.

    Args:
        repo_full_name: e.g. "some-org/some-repo"
        pr_number: PR number
        comment_body: The full markdown body to post

    Returns:
        The comment ID (new or existing).
    """
    gh = _get_github_client()
    repo = gh.get_repo(repo_full_name)
    issue = repo.get_issue(pr_number)

    # Check for existing bot comment
    existing_id = find_existing_bot_comment(repo_full_name, pr_number)

    if existing_id is not None:
        # Update existing comment
        comment = issue.get_comment(existing_id)
        comment.edit(comment_body)
        logger.info(
            "Updated existing comment #%d on %s #%d",
            existing_id,
            repo_full_name,
            pr_number,
        )
        return existing_id
    else:
        # Create new comment
        new_comment = issue.create_comment(comment_body)
        logger.info(
            "Created new comment #%d on %s #%d",
            new_comment.id,
            repo_full_name,
            pr_number,
        )
        return new_comment.id


def delete_bot_comment(
    repo_full_name: str,
    pr_number: int,
) -> bool:
    """
    Delete the existing bot comment on a PR, if it exists.

    Returns:
        True if a comment was deleted, False if none was found.
    """
    gh = _get_github_client()
    repo = gh.get_repo(repo_full_name)
    issue = repo.get_issue(pr_number)

    existing_id = find_existing_bot_comment(repo_full_name, pr_number)
    if existing_id is not None:
        comment = issue.get_comment(existing_id)
        comment.delete()
        logger.info(
            "Deleted bot comment #%d on %s #%d",
            existing_id,
            repo_full_name,
            pr_number,
        )
        return True

    logger.info("No bot comment to delete on %s #%d", repo_full_name, pr_number)
    return False
