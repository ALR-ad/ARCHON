# PR Reviewer Agent

Detects duplicated logic and architectural-convention violations on GitHub PRs.
See CONTRACTS.md for the interface both people build against.

## Tracks
- `indexing/` -- Person A: codebase/wiki -> vector store
- `agent/`    -- Person B: webhook -> DAG -> GitHub comment
- `shared/`   -- co-owned contracts (schemas.py, interfaces.py, config.py)

## Setup
1. Copy `.env.example` -> `.env` and fill in values
2. Copy `.agent-review.yml.example` -> `.agent-review.yml`
3. `pip install -r requirements.txt`
4. Install Ollama locally and pull models: `ollama pull nomic-embed-text` / `ollama pull qwen2.5-coder:32b`
