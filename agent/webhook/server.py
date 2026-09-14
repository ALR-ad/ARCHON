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

logger = logging.getLogger("agent.webhook")

app = FastAPI()


@app.post("/webhook")
async def receive_webhook(request: Request):
    payload = await request.json()
    logger.info("Received webhook payload: %s", payload)
    return {"status": "received"}