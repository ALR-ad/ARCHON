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


def test_webhook_returns_status_received():
    """POST /webhook with any JSON body returns 200 + correct shape."""
    response = client.post("/webhook", json={"action": "opened", "number": 1})
    assert response.status_code == 200
    assert response.json() == {"status": "received"}


def test_webhook_logs_payload(caplog):
    """The handler must log the received payload so it is visible in the terminal."""
    with caplog.at_level(logging.INFO, logger="agent.webhook"):
        client.post(
            "/webhook",
            json={"action": "opened", "pull_request": {"number": 42}},
        )
    # Verify our log message appeared with the payload contents
    assert any("Received webhook payload" in record.message for record in caplog.records)
    assert any("42" in record.message for record in caplog.records)
