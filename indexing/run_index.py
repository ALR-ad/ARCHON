"""
indexing/run_index.py

PERSON A: CLI entrypoint -- run manually or via cron/GitHub Action on push to main.
Coordinates repository ingestion, chunking, embedding, LanceDB upsert, and
incremental state tracking.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add repo root to sys.path if running as standalone script
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from shared.config import EMBEDDING_MODEL
from indexing.config import OLLAMA_HOST, QDRANT_PATH, QDRANT_URL
from indexing.embeddings.embedder import embed_text
from indexing.incremental import IncrementalIndexer
from indexing.vectorstore.store import QdrantVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("indexing.run_index")


def main():
    parser = argparse.ArgumentParser(
        description="ARCHON Track A: Continuous & Incremental Codebase/Wiki Vector Indexer (Qdrant)"
    )
    parser.add_argument(
        "--repo-path",
        default=".",
        help="Path to repository root to index (default: current directory)",
    )
    parser.add_argument(
        "--github-repo",
        default=None,
        help="GitHub repository to clone/pull and index (e.g. 'owner/repo')",
    )
    parser.add_argument(
        "--wiki-path",
        default=None,
        help="Optional path to external/in-repo wiki or docs directory",
    )
    parser.add_argument(
        "--confluence-path",
        default=None,
        help="Optional path to Confluence export directory or zip file",
    )
    parser.add_argument(
        "--db-path",
        default=QDRANT_PATH,
        help=f"Path to local Qdrant database directory (default: {QDRANT_PATH})",
    )
    parser.add_argument(
        "--qdrant-url",
        default=QDRANT_URL,
        help="Optional Qdrant server URL (e.g. http://localhost:6333 for Docker)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force a full re-index of all repository files regardless of incremental state",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Display current vector store index status and exit",
    )
    parser.add_argument(
        "--host",
        default=OLLAMA_HOST,
        help=f"Ollama server URL (default: {OLLAMA_HOST})",
    )
    parser.add_argument(
        "--model",
        default=EMBEDDING_MODEL,
        help=f"Ollama embedding model name (default: {EMBEDDING_MODEL})",
    )

    args = parser.parse_args()

    vs = QdrantVectorStore(url=args.qdrant_url, path=args.db_path)

    if args.status:
        status = vs.get_index_status()
        print("--- Index Status (Qdrant) ---")
        print(f"Database Path/URL: {vs.db_path or args.qdrant_url}")
        print(f"Collection:        {vs.collection_name}")
        print(f"Total Chunks:      {status.get('chunk_count', '0')}")
        print(f"Last Indexed SHA:  {status.get('last_indexed_sha', 'None')}")
        return

    logger.info(f"Starting Track A Indexing (Backend: Qdrant)")
    if args.github_repo:
        logger.info(f"Repository source: GitHub repo '{args.github_repo}'")
    else:
        logger.info(f"Repository source: Local directory '{args.repo_path}'")
    logger.info(f"Embedding model: '{args.model}' via '{args.host}'")
    if args.full:
        logger.info("Mode: FULL re-indexing forced")
    else:
        logger.info("Mode: INCREMENTAL indexing")

    # Embedding wrapper with custom host and model if specified
    def custom_embed(text: str):
        return embed_text(text, host=args.host, model=args.model)

    indexer = IncrementalIndexer(
        repo_root=args.repo_path if not args.github_repo else None,
        github_repo=args.github_repo,
        wiki_path=args.wiki_path,
        confluence_export_path=args.confluence_path,
        vector_store=vs,
        embed_fn=custom_embed,
    )

    try:
        stats = indexer.run_indexing(force_full=args.full)
        print("\n================ Indexing Complete ================")
        print(f"Added Files:         {stats.added_files}")
        print(f"Modified Files:      {stats.modified_files}")
        print(f"Deleted Files:       {stats.deleted_files}")
        print(f"Renamed Files:       {stats.renamed_files}")
        print(f"Unchanged Files:     {stats.unchanged_files}")
        print(f"Chunks Upserted:     {stats.chunks_upserted}")
        print(f"Last Indexed SHA:    {stats.last_indexed_sha or 'N/A'}")
        print(f"Total Chunks Stored: {vs.count()}")
        print("====================================================\n")
    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

