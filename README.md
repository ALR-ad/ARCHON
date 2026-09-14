# ARCHON

Detects duplicated logic and architectural-convention violations on GitHub PRs.
See CONTRACTS.md for the interface both people build against.

## Structure and ownership

| Path | Owner | What's there |
|---|---|---|
| `shared/` | both, needs review from both to change | data models (`schemas.py`), the interface between the two tracks (`interfaces.py`), shared constants (`config.py`) |
| `indexing/` | Person A | pulling in the repo/wiki, chunking, embeddings, vector store, incremental re-indexing |
| `agent/` | Person B | webhook receiver, the four DAG nodes, Ollama evaluation, posting comments, wiring the DAG |
| `tests/` | both | tests for each track |

The two tracks meet at exactly two functions, both defined in
`shared/interfaces.py`:

```
embed_text(text: str) -> List[float]
VectorStore.query(embedding, top_k, filters) -> List[Candidate]
```

Person B builds against `agent/llm/mock_vectorstore.py` until Person A's real
implementation is ready to swap in.

## Stack

- Python 3.11+
- Pydantic v2 for the shared schemas
- Strands Agents SDK for the DAG
- FastAPI for the webhook
- tree-sitter (`tree-sitter-language-pack`) for code chunking
- LanceDB for the vector store — embedded, no server to run
- Ollama for embeddings and evaluation, running locally
- PyGithub for the GitHub API
- pytest / pytest-asyncio for tests
- GitHub Actions for CI

## Status

- Webhook receiver — done
- Node 1, diff ingestion — done
- Node 2, semantic search — done, tested against the mock store
- Node 3, Ollama evaluation — done, tested against a real local Ollama call
- Node 4, comment rendering — done
- GitHub client — done, dry-run tested, not yet run against a live PR
- Full DAG wiring — done
- Real vector store and embeddings — in progress (Person A)
- Live GitHub App / real webhook — not started

See `agent/INTEGRATION_READY.md` for the exact list of what's mocked, what's
real, and what changes when the indexing side is ready.

## Setup

```bash
git clone <this-repo-url>
cd archon
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1
# Mac/Linux
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
cp .agent-review.yml.example .agent-review.yml
```

Install Ollama and pull the models listed in `shared/config.py`:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:7b
```

Run the webhook locally:

```bash
uvicorn agent.webhook.server:app --reload
```

Run tests:

```bash
pytest tests/ -v
```

## Working on this

`main` is protected. Everything goes through a PR.

Branch names: `agent-dag/<task>` for Person B, `indexing/<task>` for Person A.

Changes to `shared/schemas.py` or `shared/interfaces.py` need review from both
people — that file is the contract the rest of the project depends on, and one
person changing it without telling the other breaks things quietly.

CI runs the test suite on every PR and has to pass before merging.
