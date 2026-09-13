# Milestone 1: Integration Ready Checklist

This document summarizes the current state of the `agent/` track (Person B's work) and outlines exactly what needs to be changed when Person A's `indexing/` track is ready for integration.

## 1. What is Fully Real vs. Simulated

### ✅ Fully Real (Implemented & Verified)
- **Webhook Server**: `agent/webhook/server.py` receives payloads, logs them, and responds with a 200 OK.
- **DAG Orchestration**: `agent/dag.py` uses the Strands Agents SDK Graph to wire Nodes 1-4 together sequentially, passing an `invocation_state` context dictionary between them.
- **Node 1 (Ingest)**: Extracts pull request metadata from the webhook payload. (Currently uses fake diff output).
- **Node 2 (Semantic Search)**: Filters candidates by `SIMILARITY_FLOOR` and structures matches correctly.
- **Node 3 (LLM Evaluation)**: Fully functional integration with local Ollama (`qwen2.5-coder:7b`). Evaluates candidates, requests strict JSON formatting, and parses results into `Finding` objects.
- **Node 4 (Comment Formatting)**: Renders `Finding` objects into a clean, collapsible Markdown comment with issue badges and a hidden HTML tracking marker (`BOT_COMMENT_MARKER`).
- **GitHub Client**: `agent/github_client.py` is fully implemented using PyGithub to fetch PR unified diffs and idempotently post or edit comments based on the hidden HTML tracking marker. 
- **Tests**: Pytest suites exist and pass for all nodes and the DAG orchestration.

### 🚧 Currently Simulated (Requires Person A's Work)
- **Vector Store**: Node 2 currently uses `agent.llm.mock_vectorstore.MockVectorStore` which returns hardcoded semantic search candidates.
- **Text Embedding**: Node 2 uses a mock `embed_text()` stub because the real embedding pipeline is part of the `indexing/` track.
- **Diff Ingestion API**: Node 1 currently generates fake `ChangedCodeUnit` data instead of calling `github_client.fetch_pr_diff()`, as we are not calling the real GitHub API yet.

---

## 2. Integration Steps (What Needs to Change)

When Person A's track is complete, follow these exact steps to wire the two tracks together:

1. **Delete the Mock Vector Store**: 
   - Remove `agent/llm/mock_vectorstore.py` entirely.
2. **Import the Real Vector Store and Embedding Function**:
   - In `agent/nodes/node2_semantic_search.py`, remove the `MockVectorStore` import.
   - Import Person A's real `VectorStore` implementation (e.g., LanceDB) and the real `embed_text` function from the `indexing/` module.
3. **Connect Node 1 to Real GitHub Diffs**:
   - In `agent/nodes/node1_ingest_diff.py`, remove the fake `ChangedCodeUnit` hardcoded logic.
   - Import `fetch_pr_diff` from `agent/github_client`.
   - Implement the logic to fetch the unified diff, parse it (perhaps using Person A's tree-sitter chunking logic if applicable), and yield real `ChangedCodeUnit` objects.
4. **Wire the GitHub Client into the Webhook / DAG**:
   - Update the webhook endpoint or `dag.py` so that the output of Node 4 (`comment` string) is actually passed to `github_client.post_or_update_comment()`.

---

## 3. Known Limitations & Manual Testing Needed

- **Model Confidence**: The prompt in Node 3 uses a low temperature for strict JSON, but `qwen2.5-coder:7b`'s reasoning or formatting capabilities might vary. You should manually test Node 3 with a few diverse codebase examples before fully trusting the `is_duplicate` / `is_architectural_violation` flags.
- **PyGithub Credentials**: We have mocked PyGithub for all tests (`test_github_dryrun.py`). Before full deployment, you must test the actual PyGithub token against a real staging repository to ensure token scopes and permissions are correct.
- **Tree-Sitter Dependency**: Person A's track relies on `tree-sitter-language-pack`, which had installation issues on Python 3.13. Verify this is fully resolved in their environment before attempting to parse real diffs in Node 1.
