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
