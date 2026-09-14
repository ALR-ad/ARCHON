# Milestone 2: Integration Verification Complete

This document summarizes the integration between Track A (`indexing/`) and Track B (`agent/`) of the Archon Tech Debt & Architectural PR Reviewer.

## 1. What is Fully Real vs. Simulated

### ✅ Fully Real (Implemented & Verified End-to-End)
- **Webhook Server**: `agent/webhook/server.py` receives payloads, logs them, and responds with a 200 OK.
- **DAG Orchestration**: `agent/dag.py` uses the Strands Agents SDK Graph to wire Nodes 1-4 together sequentially, passing an `invocation_state` context dictionary between them.
- **Node 1 (Ingest)**: Extracts pull request metadata from the webhook payload. (Currently uses fake diff output for mock-based tests; real diff ingestion via `github_client.fetch_pr_diff()` available but not yet wired into the DAG).
- **Node 2 (Semantic Search)**: Filters candidates by `SIMILARITY_FLOOR` and structures matches correctly. **NOW VERIFIED** with the real `QdrantVectorStore` and real `embed_text()` (Ollama `nomic-embed-text`) from Track A.
- **Node 3 (LLM Evaluation)**: Fully functional integration with local Ollama (`qwen2.5-coder:7b`). **NOW VERIFIED** end-to-end with real vector store results feeding into LLM evaluation.
- **Node 4 (Comment Formatting)**: Renders `Finding` objects into a clean, collapsible Markdown comment with issue badges and a hidden HTML tracking marker (`BOT_COMMENT_MARKER`). **NOW VERIFIED** with real findings from the full pipeline.
- **GitHub Client**: `agent/github_client.py` is fully implemented using PyGithub to fetch PR unified diffs and idempotently post or edit comments based on the hidden HTML tracking marker.
- **Vector Store (Track A)**: `indexing/vectorstore/store.py` implements `QdrantVectorStore` backed by Qdrant. **VERIFIED**: inherits from `shared.interfaces.VectorStore`, returns exact `Candidate` objects matching the schema, query method signature matches the contract perfectly.
- **Text Embedding (Track A)**: `indexing/embeddings/embedder.py` implements `embed_text()` wrapping Ollama `nomic-embed-text`. **VERIFIED**: produces 768-dimensional vectors matching `EMBEDDING_DIM`, callable through the `shared.interfaces.embed_text()` passthrough.
- **Indexing Pipeline (Track A)**: `indexing/run_index.py` + `IncrementalIndexer` handles repo ingestion, tree-sitter chunking, embedding, and Qdrant upsert. **VERIFIED** against a test fixture repo.
- **Mock-based Tests**: All 61 existing mock-based tests remain intact and passing.

### 🚧 Still Simulated (Not Yet Wired in Production Path)
- **Diff Ingestion in DAG**: `agent/dag.py`'s `_run_node1` still uses `ingest_diff()` with fake `ChangedCodeUnit` data, not the real GitHub API diff. The integration entry point (`tests/integration/run_real_dag.py`) bypasses this via `units_override`.
- **DAG Vector Store Injection**: `agent/dag.py`'s `_run_node2` still hardcodes `MockVectorStore()`. The real store is injected via the functional pipeline in `tests/integration/run_real_dag.py`, not through the Strands graph.
- **GitHub API Posting**: Node 4 output is not yet posted to a real GitHub PR. `github_client.py` is implemented and tested with mocks, but not wired into the live DAG.

---

## 2. Integration Test Results (Phase 4)

### Test Case A: True Positive (Duplicate Detection)
- **Input**: A `parse_order_date()` function that duplicates the indexed `parse_date()` from `date_utils.py`
- **Node 2 Results**: 3 candidates above `SIMILARITY_FLOOR` (0.78)
  - `parse_date` — similarity: **0.9362**
  - `parse_iso_datetime` — similarity: 0.8267
  - `format_date` — similarity: 0.7875
- **Node 3 Results**: 1 finding produced
  - `is_duplicate`: **True**
  - `is_architectural_violation`: False
  - `confidence`: **0.95**
  - `reasoning`: "The new code is nearly identical to the existing `parse_date` function in `src/utils/date_utils.py`, differing only in the function name and docstring."
- **Node 4 Results**: 1,537-char rendered markdown comment with duplicate badge, collapsible code snippets, and bot marker
- **Total Pipeline Time**: ~9.01 seconds

### Test Case B: True Negative (No False Positives)
- **Input**: A `compute_shipping_cost()` function (genuinely novel, no duplicates in index)
- **Node 2 Results**: 0 candidates above `SIMILARITY_FLOOR` — all 8 raw candidates scored below 0.78
- **Pipeline Result**: **Zero findings, zero false positives**
- **Total Pipeline Time**: ~2.50 seconds

### Summary
| Metric | Expected | Actual | Status |
|---|---|---|---|
| True positive detected | Yes | Yes | ✅ |
| Duplicate confidence | ≥ 0.75 | 0.95 | ✅ |
| Top candidate similarity | ≥ 0.78 | 0.9362 | ✅ |
| False positives | 0 | 0 | ✅ |
| Comment rendered | Yes | 1,537 chars | ✅ |
| Existing tests still pass | 61/61 | 61/61 | ✅ |

---

## 3. Contract Compliance Audit Summary

Zero breaking mismatches found between Track A's real implementations and `shared/interfaces.py`:

| Interface | Contract | Track A | Match? |
|---|---|---|---|
| `embed_text(text: str) -> List[float]` | 1 param | 1 required + 2 optional (host, model) | ✅ (defaults work) |
| Embedding dimension | 768 | 768 | ✅ |
| `VectorStore.query()` signature | `(embedding, top_k=8, filters=None)` | Identical | ✅ |
| `VectorStore` inheritance | ABC from `shared.interfaces` | `QdrantVectorStore(VectorStore)` | ✅ |
| `Candidate` field names/types | 6 fields | All match exactly | ✅ |
| `get_index_status()` return | `Dict[str, str]` with `last_indexed_sha`, `chunk_count` | Matches | ✅ |

Minor non-blocking observations:
- `embed_text()` has extra optional kwargs (`host`, `model`) not in the contract — harmless since they have defaults
- Dimension validation on mismatch is a warning, not an error — fine as long as `nomic-embed-text` continues producing 768-dim vectors

---

## 4. Integration Steps Remaining for Production

1. **Wire real diff ingestion in DAG**: Replace `_run_node1`'s fake data with `github_client.fetch_pr_diff()` + tree-sitter parsing to produce real `ChangedCodeUnit` objects from actual PR diffs.
2. **Make `build_dag()` injectable**: Parameterize `build_dag()` to accept a `VectorStore` and `embed_fn`, eliminating the hardcoded `MockVectorStore()` in `_run_node2`.
3. **Wire GitHub posting**: Connect Node 4 output to `github_client.post_or_update_comment()` in the webhook handler.
4. **Deploy Qdrant**: Set up a persistent Qdrant instance (Docker or embedded) and configure `QDRANT_URL` / `QDRANT_PATH` in `.env`.
5. **Set up indexing schedule**: Run `indexing/run_index.py` on push-to-main (via GitHub Action or cron) to keep the index fresh.

---

## 5. Known Limitations & Observations

- **Embedding latency**: Each `embed_text()` call to local Ollama takes ~2-3 seconds. For a PR with many changed functions, this adds up. Consider batching or caching.
- **LLM evaluation latency**: Ollama `qwen2.5-coder:7b` evaluation takes ~5-20 seconds per unit. Large PRs with many semantic matches will be slow.
- **Confidence scores are well-calibrated**: The 0.95 confidence for a near-exact duplicate is appropriate. The model correctly identified function-name-only differences.
- **Similarity floor works well**: The 0.78 floor cleanly separates relevant matches (date-related functions) from irrelevant ones (order validation, string helpers). No tuning needed.
- **Tree-sitter chunking**: Track A's code chunker extracted 9 chunks from 3 Python files (functions + classes + dataclasses), which is the right granularity.
- **Qdrant local storage**: Works well for testing. Production should use Docker or a server instance for reliability.
