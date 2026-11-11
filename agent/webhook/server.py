import asyncio
import hashlib
import hmac
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse

# Rendered comments contain emoji; a Windows console (cp1252) can't encode them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Configure root logger so application-level loggers have a handler.
# Use stderr (unbuffered on Windows) to avoid the stdout buffering issue
# that occurs when uvicorn --reload spawns a child subprocess.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(name)s - %(message)s",
    stream=sys.stdout,
    force=True,
)

from agent.dag import build_dag
from agent.github_client import post_or_update_comment
from agent.nodes.node1_ingest_diff import _extract_pr_metadata
from indexing.vectorstore.store import QdrantVectorStore
from indexing.embeddings.embedder import embed_text
from indexing.incremental import IncrementalIndexer
from agent.llm.ollama_eval import confirm_violation_with_ollama, evaluate_with_ollama
from indexing.config import QDRANT_PATH, QDRANT_URL, load_agent_review_config
from shared.config import CONFIDENCE_THRESHOLD, SIMILARITY_FLOOR

logger = logging.getLogger("agent.webhook")

# Initialize global dependencies for the DAG
vector_store = None
agent_graph = None

# Reviews and re-indexing run one at a time: they share one Ollama instance
# and one vector store, and a review should see a fully updated index.
_work_lock = asyncio.Lock()
# True while a re-index is waiting for the lock. Each run pulls the latest
# commit, so pushes that arrive meanwhile don't need a run of their own.
_reindex_queued = False


def _index_repo() -> Optional[str]:
    """The one repository this server indexes ("owner/name"), if configured."""
    return os.getenv("INDEX_REPO") or None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, agent_graph
    # Per-repo review settings from .agent-review.yml (falls back to shared/config.py)
    review_cfg = load_agent_review_config(os.getenv("AGENT_REVIEW_CONFIG_ROOT", "."))

    # Instantiate the real vector store and bind it to the DAG
    logger.info("Initializing QdrantVectorStore at %s", QDRANT_URL or QDRANT_PATH)
    vector_store = QdrantVectorStore(path=QDRANT_PATH)
    agent_graph = build_dag(
        store=vector_store,
        embed_fn=embed_text,
        eval_fn=evaluate_with_ollama,
        confirm_fn=confirm_violation_with_ollama,
        similarity_floor=float(review_cfg.get("similarity_floor", SIMILARITY_FLOOR)),
        confidence_threshold=float(review_cfg.get("confidence_threshold", CONFIDENCE_THRESHOLD)),
        enabled_checks=review_cfg.get("enabled_checks"),
        review_ignore_paths=review_cfg.get("review_ignore_paths"),
        wiki_top_k=review_cfg.get("wiki_top_k"),
    )
    logger.info("DAG built successfully.")

    # First run: build the index so reviews have something to search
    initial_index = None
    if _index_repo() and vector_store.count() == 0:
        logger.info("Index is empty; indexing %s in the background", _index_repo())
        initial_index = asyncio.create_task(reindex_repository(_index_repo()))

    yield
    if initial_index and not initial_index.done():
        initial_index.cancel()
    if vector_store:
        vector_store.close()


app = FastAPI(lifespan=lifespan)


def verify_signature(body: bytes, signature_header: Optional[str], secret: str) -> bool:
    """Check GitHub's X-Hub-Signature-256 header (HMAC-SHA256 of the raw body)."""
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header[len("sha256="):])


@app.get("/health")
def health_check():
    return {"status": "ok"}


def _run_incremental_index(repo_full_name: str):
    """Pull the repo's default branch and index what changed (blocking)."""
    indexer = IncrementalIndexer(
        repo_root=None,
        github_repo=repo_full_name,
        vector_store=vector_store,
        embed_fn=embed_text,
    )
    # No indexer.close(): the vector store is shared with the DAG
    return indexer.run_indexing()


async def reindex_repository(repo_full_name: str) -> None:
    """Bring the index up to date with the repo's default branch."""
    global _reindex_queued
    async with _work_lock:
        _reindex_queued = False
        logger.info("Re-indexing %s", repo_full_name)
        try:
            stats = await asyncio.to_thread(_run_incremental_index, repo_full_name)
        except Exception as e:
            logger.error("Re-index of %s failed: %s", repo_full_name, e, exc_info=True)
            return
        logger.info(
            "Re-index of %s done at %s: %d added, %d modified, %d deleted, %d renamed, %d chunks upserted",
            repo_full_name,
            (stats.last_indexed_sha or "unknown")[:8],
            stats.added_files,
            stats.modified_files,
            stats.deleted_files,
            stats.renamed_files,
            stats.chunks_upserted,
        )


async def process_pull_request(payload: Dict[str, Any]) -> None:
    """Run the DAG for one PR and post (or log) the resulting comment."""
    meta = _extract_pr_metadata(payload)

    final_state = {"payload": payload}
    async with _work_lock:
        logger.info("Processing PR #%s for %s", meta["pr_number"], meta["repo_full_name"])
        try:
            async for chunk in agent_graph.stream_async(task="evaluate", invocation_state=final_state):
                pass
        except Exception as e:
            logger.error("DAG execution failed: %s", e, exc_info=True)
            return

    # Extract the comment
    comment = final_state.get("comment", "")
    if not comment:
        logger.info("No comment generated for PR #%s.", meta["pr_number"])
        return

    # Check if we are allowed to post to GitHub (safety toggle)
    post_enabled = os.getenv("POST_TO_GITHUB", "false").lower() == "true"
    if not post_enabled:
        logger.info("POST_TO_GITHUB is not 'true'. Skipping real API call. Comment would be:\n%s", comment)
        return

    try:
        logger.info("Posting comment to GitHub...")
        post_or_update_comment(
            repo_full_name=meta["repo_full_name"],
            pr_number=meta["pr_number"],
            comment_body=comment,
        )
    except Exception as e:
        logger.error("Failed to post comment: %s", e, exc_info=True)


def _handle_push(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    """A push to the default branch makes the index stale; queue a re-index."""
    global _reindex_queued
    repo = payload.get("repository") or {}
    repo_full_name = repo.get("full_name", "")
    default_ref = f"refs/heads/{repo.get('default_branch', 'main')}"

    if payload.get("ref") != default_ref or payload.get("deleted"):
        return {"status": "ignored", "reason": "Push is not to the default branch"}
    if not _index_repo():
        return {"status": "ignored", "reason": "INDEX_REPO is not set; automatic re-indexing is off"}
    if repo_full_name.lower() != _index_repo().lower():
        return {"status": "ignored", "reason": f"This server indexes {_index_repo()}, not {repo_full_name}"}

    after = payload.get("after", "")
    if _reindex_queued:
        logger.info("Push %s to %s: re-index already queued", after[:8], repo_full_name)
        return JSONResponse(status_code=202, content={"status": "accepted", "reindex": "already queued"})

    _reindex_queued = True
    logger.info("Push %s to %s: queued re-index", after[:8], repo_full_name)
    background_tasks.add_task(reindex_repository, repo_full_name)
    return JSONResponse(status_code=202, content={"status": "accepted", "reindex": "queued"})


@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()

    secret = os.getenv("GITHUB_WEBHOOK_SECRET")
    if secret:
        if not verify_signature(body, request.headers.get("X-Hub-Signature-256"), secret):
            logger.warning("Rejected webhook with missing or invalid signature")
            return JSONResponse(status_code=401, content={"status": "error", "message": "Invalid signature"})
    else:
        logger.warning("GITHUB_WEBHOOK_SECRET is not set; accepting unsigned webhook (dev mode only)")

    try:
        payload = json.loads(body)
    except ValueError:
        return JSONResponse(status_code=400, content={"status": "error", "message": "Invalid JSON"})
    logger.info("Received webhook payload")

    if request.headers.get("X-GitHub-Event") == "push":
        return _handle_push(payload, background_tasks)

    # Only process pull_request events
    action = payload.get("action")
    if "pull_request" not in payload or action not in ("opened", "synchronize", "reopened"):
        return {"status": "ignored", "reason": "Not a valid pull_request event or action"}

    meta = _extract_pr_metadata(payload)
    logger.info("Queued PR #%s for %s", meta["pr_number"], meta["repo_full_name"])

    # GitHub times out webhook deliveries after 10s, so the slow DAG
    # (embedding + LLM calls) runs after the response is sent.
    background_tasks.add_task(process_pull_request, payload)
    return JSONResponse(status_code=202, content={"status": "accepted", "pr_number": meta["pr_number"]})
