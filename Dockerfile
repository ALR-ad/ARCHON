FROM python:3.11-slim

# git: the indexer clones and updates the repository it indexes
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ shared/
COPY indexing/ indexing/
COPY agent/ agent/
COPY .agent-review.yml.example .agent-review.yml

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "agent.webhook.server:app", "--host", "0.0.0.0", "--port", "8000"]
