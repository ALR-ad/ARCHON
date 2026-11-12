```
 █████╗ ██████╗  ██████╗██╗  ██╗ ██████╗ ███╗   ██╗
██╔══██╗██╔══██╗██╔════╝██║  ██║██╔═══██╗████╗  ██║
███████║██████╔╝██║     ███████║██║   ██║██╔██╗ ██║
██╔══██║██╔══██╗██║     ██╔══██║██║   ██║██║╚██╗██║
██║  ██║██║  ██║╚██████╗██║  ██║╚██████╔╝██║ ╚████║
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
                                            v1.0.0
```

# Archon — Tech Debt & Architectural PR Reviewer

A GitHub bot that catches two specific things on pull requests: duplicated logic
that already exists elsewhere in the codebase, and code that breaks documented
architectural conventions. It reads a PR's diff, searches the codebase and
internal wiki for similar existing code, evaluates the matches with a local
LLM, and leaves one comment if it finds something worth flagging.

It doesn't try to be a linter, a security scanner, or a general code reviewer.
One job, done narrowly.

Everything runs on your own machine: the LLM and embeddings through Ollama, the
vector store in Qdrant. Your code only goes to GitHub's API.

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
                     | Vector store |  <- re-indexed on every push to the
                     | (code+wiki)  |     default branch
                     +--------------+
```

1. **Ingest diff.** Fetches the PR's changed files, splits them into whole
   functions and classes with tree-sitter, and keeps the ones that touch
   changed lines.
2. **Semantic search.** Embeds each changed unit and looks up the most similar
   existing code and wiki rules, then drops matches that aren't duplication
   (see [Keeping the review quiet](#keeping-the-review-quiet)).
3. **Evaluate.** A local LLM decides, for each unit and its matches, whether it
   is duplicated logic or an architectural violation, and how confident it is.
4. **Comment.** Renders one comment grouped by file and edits it in place on
   later pushes, so a PR never collects a pile of bot comments.

The pipeline is a deterministic graph built with the Strands Agents SDK. Only
Node 3 calls an LLM. Everything else is plain Python you can unit test without
mocking a model.

Code is chunked by function, class and method for Python, TypeScript/TSX,
JavaScript, Go, Java, Rust, C, C++, C#, Ruby and PHP. Source files in other
languages aren't indexed; markdown docs are indexed as wiki rules.

## Quick start with Docker

Runs Archon, Qdrant and Ollama together:

```bash
cp .env.example .env    # set GITHUB_TOKEN, GITHUB_WEBHOOK_SECRET and INDEX_REPO=owner/name
docker compose up -d
```

The first start downloads the two Ollama models (about 5 GB) and indexes
`INDEX_REPO`. Then add a webhook on that repository:

- **Payload URL:** `http://<host>:8000/webhook`
- **Content type:** `application/json`
- **Secret:** the same value as `GITHUB_WEBHOOK_SECRET`
- **Events:** Pull requests and Pushes

Archon starts in dry-run mode: it logs the comment it would post. Set
`POST_TO_GITHUB=true` once you're happy with what it says.

To use an NVIDIA GPU, uncomment the `deploy` block under `ollama` in
`docker-compose.yml`.

## Manual setup

Needs Python 3.11+ and [Ollama](https://ollama.com).

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

Pull the models:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:7b
```

Build the index, then run the webhook:

```bash
python -m indexing.run_index --repo-path /path/to/your/repo
uvicorn agent.webhook.server:app
```

If you set `INDEX_REPO` in `.env`, skip the first command: the server clones
and indexes that repository itself.

## Configuration

### `.env`

| Variable | Default | What it does |
|---|---|---|
| `GITHUB_TOKEN` | — | Reads PRs and posts comments. Needs Contents (read) and Pull requests (read and write) on the repository. |
| `GITHUB_WEBHOOK_SECRET` | — | When set, requests without a valid `X-Hub-Signature-256` are rejected with 401. When empty, unsigned requests are accepted, which is for local development only. |
| `POST_TO_GITHUB` | `false` | `true` posts the comment. Anything else only logs it. |
| `INDEX_REPO` | — | The one repository this server indexes (`owner/name`). Turns on automatic indexing. |
| `OLLAMA_HOST` | `http://localhost:11434` | Where Ollama is running. |
| `QDRANT_PATH` | `./data/qdrant` | On-disk vector store. With `QDRANT_URL` set, it only holds index metadata. |
| `QDRANT_URL` | — | Use a Qdrant server instead of the on-disk store. |
| `AGENT_REVIEW_CONFIG_ROOT` | `.` | Folder containing the `.agent-review.yml` the server uses. |

### `.agent-review.yml`

| Key | Default | What it does |
|---|---|---|
| `similarity_floor` | `0.82` | Minimum embedding similarity for a match to reach the LLM. Lower finds more and is noisier. |
| `confidence_threshold` | `0.75` | Minimum LLM confidence for a finding to be reported. |
| `enabled_checks` | both `true` | Turn `duplication` or `architectural_violation` off. |
| `review_ignore_paths` | test files | Files the review skips, both as changed code and as matches. Set `[]` to review tests too. |
| `wiki_top_k` | `4` | How many wiki rules each changed function is checked against. `0` turns the rule lookup off. |
| `ignore_paths` | lock files, `node_modules`, `dist` | Files the indexer skips. |

Restart the server after editing it.

## Keeping the index fresh

With `INDEX_REPO` set, the server keeps its own clone of that repository
(under `data/repos/`) and:

- builds the index on startup if it is empty
- re-indexes incrementally on every push to the default branch, re-embedding
  only files that were added, changed or renamed

Reviews and re-indexing run one at a time, so a review never sees a
half-updated index.

Without `INDEX_REPO`, run the indexer yourself:

```bash
python -m indexing.run_index --repo-path .            # incremental
python -m indexing.run_index --repo-path . --full     # rebuild from scratch
python -m indexing.run_index --status                 # chunk count and last indexed commit
```

Add `--wiki-path <dir>` for a folder of markdown docs, or
`--confluence-path <dir-or-zip>` for a Confluence export.

## Architectural rules

Rules are plain markdown. The indexer picks them up from `docs/`, `wiki/` and
files such as `ARCHITECTURE.md`, `CONTRIBUTING.md` or `STYLEGUIDE.md` in the
indexed repository, and splits them into one rule per bullet or paragraph:

```markdown
## HTTP calls

- All outbound HTTP requests must use the shared `http_client` wrapper.
  Do not call `requests` or `httpx` directly.
```

Prose and code never score high against each other, so rules aren't filtered
by similarity. For each changed function Archon takes the `wiki_top_k` closest
rules and asks the LLM whether one is broken. A violation is only reported if:

1. the LLM names the rule it breaks, and
2. a second, narrower check of that one rule against that one function agrees
   and quotes the offending line, and the quote really is in the code.

Write rules as clear requirements or prohibitions ("must", "never", "do not").
Descriptive text is ignored. With no rule documents in the repository, this
check costs nothing.

After upgrading, the next index run re-embeds existing wiki chunks once.

## Keeping the review quiet

A duplicate detector is only useful if it stays quiet when nothing is wrong.
Before anything reaches the LLM, Node 2 drops matches that are:

- **In test files.** Tests are meant to look alike.
- **Callers or wrappers.** Code that calls, imports or wraps the changed
  function, or that the changed function calls.
- **Already there before the PR.** For a modified function, any match its
  base-branch version already had, so a small edit to an old function doesn't
  resurface old similarities.
- **The function itself, or a rename of it.**

The comment itself is capped: the 20 most confident findings, 3 matches each,
25 lines per snippet, and under GitHub's 65,536-character limit.

## Current limits

- **One repository per server.** The index and `.agent-review.yml` are not
  per-repository yet.
- **Rule checks are new and lightly tested.** They were tuned on seven sample
  rules. A touched function is checked as a whole, so a rule it already broke
  before the PR is reported too.
- **Confidence isn't calibrated.** The model rates most findings 90–95%, so
  `confidence_threshold` filters little. `similarity_floor` does the real work.
- **One comment per PR.** Findings aren't posted as inline review comments, so
  suggested fixes can't be applied with one click.
- **Personal access token only.** No GitHub App yet, so the token has to be
  renewed by hand when it expires.
- **No job queue.** A review in progress is lost if the server restarts.
- **The Docker setup is new** and hasn't been run end to end yet.

## Project layout

| Path | What's there |
|---|---|
| `shared/` | Data models (`schemas.py`), the interface between the two halves (`interfaces.py`), shared constants (`config.py`) |
| `indexing/` | Pulling in the repo and wiki, chunking, embeddings, vector store, incremental re-indexing |
| `agent/` | Webhook server, the four pipeline nodes, noise filters, LLM evaluation, GitHub client |
| `tests/` | Unit tests, plus integration tests that need Ollama running |

The indexing and agent halves meet at two functions, both defined in
`shared/interfaces.py`:

```
embed_text(text: str) -> List[float]
VectorStore.query(embedding, top_k, filters) -> List[Candidate]
```

## Stack

- Python 3.11+
- Pydantic v2 for the shared schemas
- Strands Agents SDK for the pipeline graph
- FastAPI for the webhook
- tree-sitter (`tree-sitter-language-pack`) for code chunking
- Qdrant for the vector store, on disk or as a server
- Ollama for embeddings (`nomic-embed-text`) and evaluation (`qwen2.5-coder:7b`)
- PyGithub for the GitHub API
- pytest / pytest-asyncio for tests
- GitHub Actions for CI

## Development

```bash
pytest tests/ -m "not integration"    # unit tests, no services needed
pytest tests/ -m integration          # needs Ollama running with both models
```

`main` is protected. Everything goes through a PR, and CI has to pass before
merging.

Changes to `shared/schemas.py` or `shared/interfaces.py` need review from both
maintainers. Those files are the contract the rest of the project depends on,
and changing them quietly breaks the other half.

## License

Apache 2.0. See [LICENSE](LICENSE).
