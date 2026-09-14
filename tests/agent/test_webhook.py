"""
Tests for agent/webhook/server.py

Verifies:
1. POST /webhook returns {"status": "received"} with 200
2. The payload is logged (not silently dropped) via the logging module
"""

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

def test_webhook_processes_valid_payload():
    """POST /webhook with a valid pull_request event executes DAG."""
    response = client.post("/webhook", json=MOCK_PAYLOAD)
    assert response.status_code == 200
    assert response.json() == {"status": "received", "findings_reported": True}

def test_webhook_logs_payload(caplog):
    """The handler must log the received payload so it is visible in the terminal."""
    with caplog.at_level(logging.INFO, logger="agent.webhook"):
        client.post("/webhook", json=MOCK_PAYLOAD)
    # Verify our log message appeared with the payload contents
    assert any("Received webhook payload" in record.message for record in caplog.records)
    assert any("42" in record.message for record in caplog.records)
