import httpx
import json

payload = {
    "action": "opened",
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

print("Sending mock PR event to http://127.0.0.1:8000/webhook ...")
try:
    response = httpx.post("http://127.0.0.1:8000/webhook", json=payload, timeout=60.0)
    print(f"Response Status: {response.status_code}")
    print(f"Response Body: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error connecting to server: {e}")
