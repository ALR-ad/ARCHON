"""
Tests for agent/webhook/server.py

Verifies:
1. Non-PR events are ignored
2. Valid PR events are accepted with 202 and the DAG runs in a background task
3. The payload is logged (not silently dropped) via the logging module
4. X-Hub-Signature-256 is enforced when GITHUB_WEBHOOK_SECRET is set
"""

import hashlib
import hmac
import json
import logging

from fastapi.testclient import TestClient

from agent.webhook.server import app


client = TestClient(app)


MOCK_PAYLOAD = {
    "action": "opened",
    "number": 42,
    "pull_request": {
        "number": 42,
        "head": {"sha": "abc1234"},
        "base": {"sha": "def5678"}
    },
    "repository": {
        "owner": {"login": "test-org"},
        "name": "test-repo",
        "full_name": "test-org/test-repo"
    }
}

import pytest
from unittest.mock import AsyncMock, patch

@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Tests opt in to the secret / posting explicitly; ignore any local .env values."""
    monkeypatch.delenv("GITHUB_WEBHOOK_SECRET", raising=False)
    monkeypatch.delenv("POST_TO_GITHUB", raising=False)


@pytest.fixture(autouse=True)
def mock_agent_graph():
    """Mock the global agent_graph in server.py so tests don't run the real DAG."""
    from strands.multiagent.graph import Graph
    
    class FakeGraph:
        async def stream_async(self, invocation_state, **kwargs):
            invocation_state["comment"] = "This is a fake comment"
            class FakeResult:
                state = invocation_state
            yield {"result": FakeResult()}
            
    with patch("agent.webhook.server.agent_graph", FakeGraph()):
        yield

def test_webhook_returns_status_received():
    """POST /webhook with any JSON body returns 200 + correct shape."""
    response = client.post("/webhook", json={"action": "unsupported", "number": 1})
    assert response.status_code == 200
    assert response.json() == {"status": "ignored", "reason": "Not a valid pull_request event or action"}

def test_webhook_processes_valid_payload(caplog):
    """POST /webhook with a valid pull_request event is accepted and the DAG runs in the background."""
    with caplog.at_level(logging.INFO, logger="agent.webhook"):
        response = client.post("/webhook", json=MOCK_PAYLOAD)
    assert response.status_code == 202
    assert response.json() == {"status": "accepted", "pr_number": 42}
    # TestClient runs background tasks before returning; POST_TO_GITHUB is off,
    # so the rendered comment is logged instead of posted.
    assert any("This is a fake comment" in record.message for record in caplog.records)


def test_webhook_posts_comment_when_enabled(monkeypatch):
    monkeypatch.setenv("POST_TO_GITHUB", "true")
    with patch("agent.webhook.server.post_or_update_comment") as mock_post:
        client.post("/webhook", json=MOCK_PAYLOAD)
    mock_post.assert_called_once_with(
        repo_full_name="test-org/test-repo",
        pr_number=42,
        comment_body="This is a fake comment",
    )


def _sign(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_webhook_rejects_missing_signature_when_secret_set(monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "s3cret")
    response = client.post("/webhook", json=MOCK_PAYLOAD)
    assert response.status_code == 401


def test_webhook_rejects_bad_signature(monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "s3cret")
    body = json.dumps(MOCK_PAYLOAD).encode()
    response = client.post(
        "/webhook",
        content=body,
        headers={"Content-Type": "application/json", "X-Hub-Signature-256": _sign(body, "wrong")},
    )
    assert response.status_code == 401


def test_webhook_accepts_valid_signature(monkeypatch):
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "s3cret")
    body = json.dumps(MOCK_PAYLOAD).encode()
    response = client.post(
        "/webhook",
        content=body,
        headers={"Content-Type": "application/json", "X-Hub-Signature-256": _sign(body, "s3cret")},
    )
    assert response.status_code == 202

def test_webhook_logs_payload(caplog):
    """The handler must log the received payload so it is visible in the terminal."""
    with caplog.at_level(logging.INFO, logger="agent.webhook"):
        client.post("/webhook", json=MOCK_PAYLOAD)
    # Verify our log message appeared with the payload contents
    assert any("Received webhook payload" in record.message for record in caplog.records)
    assert any("42" in record.message for record in caplog.records)


# -- Tests: push events re-index the default branch ---------------------------

PUSH_PAYLOAD = {
    "ref": "refs/heads/main",
    "after": "abc1234def",
    "repository": {"full_name": "test-org/test-repo", "default_branch": "main"},
}


@pytest.fixture
def reindex_calls(monkeypatch):
    """INDEX_REPO set, and reindex_repository replaced by a recorder."""
    import agent.webhook.server as server

    calls = []

    async def fake_reindex(repo_full_name):
        calls.append(repo_full_name)

    monkeypatch.setenv("INDEX_REPO", "test-org/test-repo")
    monkeypatch.setattr(server, "reindex_repository", fake_reindex)
    monkeypatch.setattr(server, "_reindex_queued", False)
    return calls


def _push(payload):
    return client.post("/webhook", json=payload, headers={"X-GitHub-Event": "push"})


def test_push_to_default_branch_queues_reindex(reindex_calls):
    response = _push(PUSH_PAYLOAD)
    assert response.status_code == 202
    assert response.json() == {"status": "accepted", "reindex": "queued"}
    assert reindex_calls == ["test-org/test-repo"]


def test_push_to_other_branch_is_ignored(reindex_calls):
    response = _push({**PUSH_PAYLOAD, "ref": "refs/heads/feature-x"})
    assert response.json()["status"] == "ignored"
    assert reindex_calls == []


def test_push_from_other_repo_is_ignored(reindex_calls):
    payload = {**PUSH_PAYLOAD, "repository": {"full_name": "someone/else", "default_branch": "main"}}
    assert _push(payload).json()["status"] == "ignored"
    assert reindex_calls == []


def test_push_ignored_when_index_repo_unset(reindex_calls, monkeypatch):
    monkeypatch.delenv("INDEX_REPO")
    assert _push(PUSH_PAYLOAD).json()["status"] == "ignored"
    assert reindex_calls == []


def test_second_push_while_queued_does_not_queue_again(reindex_calls):
    # The recorder never clears the queued flag, as if the first run were still waiting
    _push(PUSH_PAYLOAD)
    response = _push(PUSH_PAYLOAD)
    assert response.json() == {"status": "accepted", "reindex": "already queued"}
    assert reindex_calls == ["test-org/test-repo"]


def test_reindex_repository_runs_indexer_and_clears_queued_flag(monkeypatch):
    import asyncio
    import agent.webhook.server as server
    from indexing.incremental import IndexingStats

    seen = []
    monkeypatch.setattr(server, "_run_incremental_index", lambda repo: seen.append(repo) or IndexingStats())
    monkeypatch.setattr(server, "_reindex_queued", True)
    asyncio.run(server.reindex_repository("test-org/test-repo"))
    assert seen == ["test-org/test-repo"]
    assert server._reindex_queued is False


def test_reindex_failure_is_logged_not_raised(monkeypatch, caplog):
    import asyncio
    import agent.webhook.server as server

    def boom(repo):
        raise RuntimeError("clone failed")

    monkeypatch.setattr(server, "_run_incremental_index", boom)
    with caplog.at_level(logging.ERROR, logger="agent.webhook"):
        asyncio.run(server.reindex_repository("test-org/test-repo"))
    assert any("Re-index of test-org/test-repo failed" in r.message for r in caplog.records)
