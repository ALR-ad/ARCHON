import logging
import sys

from fastapi import FastAPI, Request

# Configure root logger so application-level loggers have a handler.
# Use stderr (unbuffered on Windows) to avoid the stdout buffering issue
# that occurs when uvicorn --reload spawns a child subprocess.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:     %(name)s - %(message)s",
    stream=sys.stdout,
    force=True,
)

import os

from agent.dag import build_dag
from agent.github_client import post_or_update_comment
from agent.nodes.node1_ingest_diff import _extract_pr_metadata
from indexing.vectorstore.store import QdrantVectorStore
from indexing.embeddings.embedder import embed_text
from agent.llm.ollama_eval import evaluate_with_ollama
from indexing.config import QDRANT_PATH

logger = logging.getLogger("agent.webhook")

# Initialize global dependencies for the DAG
vector_store = None
agent_graph = None

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.on_event("startup")
async def startup_event():
    global vector_store, agent_graph
    # Instantiate the real vector store and bind it to the DAG
    logger.info("Initializing QdrantVectorStore at %s", QDRANT_PATH)
    vector_store = QdrantVectorStore(path=QDRANT_PATH)
    agent_graph = build_dag(store=vector_store, embed_fn=embed_text, eval_fn=evaluate_with_ollama)
    logger.info("DAG built successfully.")

@app.on_event("shutdown")
async def shutdown_event():
    if vector_store:
        vector_store.close()

@app.post("/webhook")
async def receive_webhook(request: Request):
    payload = await request.json()
    logger.info("Received webhook payload")

    # Only process pull_request events
    action = payload.get("action")
    if "pull_request" not in payload or action not in ("opened", "synchronize", "reopened"):
        return {"status": "ignored", "reason": "Not a valid pull_request event or action"}

    meta = _extract_pr_metadata(payload)
    logger.info("Processing PR #%s for %s", meta["pr_number"], meta["repo_full_name"])

    # Execute the DAG asynchronously
    final_state = {"payload": payload}
    try:
        async for chunk in agent_graph.stream_async(task="evaluate", invocation_state=final_state):
            pass
    except Exception as e:
        logger.error("DAG execution failed: %s", e, exc_info=True)
        return {"status": "error", "message": "Pipeline failed"}

    # Extract the comment
    comment = final_state.get("comment", "")
    if not comment:
        logger.info("No comment generated.")
        return {"status": "success", "message": "No findings to report"}

    # Check if we are allowed to post to GitHub (safety toggle)
    post_enabled = os.getenv("POST_TO_GITHUB", "false").lower() == "true"
    if post_enabled:
        try:
            logger.info("Posting comment to GitHub...")
            post_or_update_comment(
                repo_full_name=meta["repo_full_name"],
                pr_number=meta["pr_number"],
                comment_body=comment,
            )
        except Exception as e:
            logger.error("Failed to post comment: %s", e, exc_info=True)
            return {"status": "error", "message": "Failed to post to GitHub"}
    else:
        logger.info("POST_TO_GITHUB is not 'true'. Skipping real API call. Comment would be:\n%s", comment)

    return {"status": "received", "findings_reported": True}