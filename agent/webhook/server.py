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
from agent.llm.ollama_eval import evaluate_with_ollama
from indexing.config import QDRANT_PATH, load_agent_review_config
from shared.config import CONFIDENCE_THRESHOLD, SIMILARITY_FLOOR

logger = logging.getLogger("agent.webhook")

# Initialize global dependencies for the DAG
vector_store = None
agent_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, agent_graph
    # Per-repo review settings from .agent-review.yml (falls back to shared/config.py)
    review_cfg = load_agent_review_config(os.getenv("AGENT_REVIEW_CONFIG_ROOT", "."))

    # Instantiate the real vector store and bind it to the DAG
    logger.info("Initializing QdrantVectorStore at %s", QDRANT_PATH)
    vector_store = QdrantVectorStore(path=QDRANT_PATH)
    agent_graph = build_dag(
        store=vector_store,
        embed_fn=embed_text,
        eval_fn=evaluate_with_ollama,
        similarity_floor=float(review_cfg.get("similarity_floor", SIMILARITY_FLOOR)),
        confidence_threshold=float(review_cfg.get("confidence_threshold", CONFIDENCE_THRESHOLD)),
        enabled_checks=review_cfg.get("enabled_checks"),
    )
    logger.info("DAG built successfully.")
    yield
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


async def process_pull_request(payload: Dict[str, Any]) -> None:
    """Run the DAG for one PR and post (or log) the resulting comment."""
    meta = _extract_pr_metadata(payload)
    logger.info("Processing PR #%s for %s", meta["pr_number"], meta["repo_full_name"])

    final_state = {"payload": payload}
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
