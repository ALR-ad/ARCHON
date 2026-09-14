```
 █████╗ ██████╗  ██████╗██╗  ██╗ ██████╗ ███╗   ██╗
██╔══██╗██╔══██╗██╔════╝██║  ██║██╔═══██╗████╗  ██║
███████║██████╔╝██║     ███████║██║   ██║██╔██╗ ██║
██╔══██║██╔══██╗██║     ██╔══██║██║   ██║██║╚██╗██║
██║  ██║██║  ██║╚██████╗██║  ██║╚██████╔╝██║ ╚████║
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
                                            v0.3.0
```

# Archon — Tech Debt & Architectural PR Reviewer

A GitHub bot that catches two specific things on pull requests: duplicated logic
that already exists elsewhere in the codebase, and code that breaks documented
architectural conventions. It reads a PR's diff, searches the codebase and
internal wiki for similar existing code, evaluates the matches with a local
LLM, and leaves one comment if it finds something worth flagging.

It doesn't try to be a linter, a security scanner, or a general code reviewer.
One job, done narrowly.

## How it works

```
GitHub PR opened/updated
        |
        v
+---------------+   +--------------+   +--------------+   +--------------+
| Node 1        |-->| Node 2       |-->| Node 3       |-->| Node 4       |
| Ingest diff   |   | Semantic     |   | Evaluate     |   | Draft and    |
|               |   | search       |   | (local LLM)  |   | post comment |
+---------------+   +--------------+   +--------------+   +--------------+
                            ^
                            | query
                     +--------------+
                     | Vector store |  <- indexed continuously from the
                     | (code+wiki)  |     codebase and wiki, separate
                     +--------------+     from the PR pipeline
```

The 4-node pipeline is a deterministic graph built with the Strands Agents SDK.
Only Node 3 calls an LLM. Everything else is plain Python you can unit test
without mocking a model.

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
