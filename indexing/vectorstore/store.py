"""
indexing/vectorstore/store.py

PERSON A: implements the VectorStore contract using Qdrant.
Matches the team decision (pgvector OR Qdrant; local Docker or local client).
Supports vector similarity search (cosine), payload filtering, upserts,
and deletions by file path / chunk ID.
Returns exact shared.schemas.Candidate objects sorted by similarity.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from qdrant_client import QdrantClient, models
from shared.config import EMBEDDING_DIM
from shared.interfaces import VectorStore
from shared.schemas import Candidate

from indexing.config import QDRANT_COLLECTION, QDRANT_PATH, QDRANT_URL

logger = logging.getLogger(__name__)

METADATA_FILENAME = "index_metadata.json"


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Generate a deterministic UUID string for a chunk_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class QdrantVectorStore(VectorStore):
    """
    Concrete VectorStore implementation backed by Qdrant.
    Supports running against:
    1. Local Docker / remote Qdrant server (via url, e.g. http://localhost:6333)
    2. Local embedded directory (via path, e.g. ./data/qdrant)
    3. In-memory mode (via location=":memory:")
    """

    def __init__(
        self,
        url: Optional[str] = None,
        path: Optional[str] = None,
        location: Optional[str] = None,
        collection_name: str = QDRANT_COLLECTION,
        embedding_dim: int = EMBEDDING_DIM,
    ):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim

        # Priority: explicit location -> explicit url -> explicit path -> env QDRANT_URL -> env QDRANT_PATH
        if location:
            self.client = QdrantClient(location=location)
            self.db_path = None
        elif url or QDRANT_URL:
            target_url = url or QDRANT_URL
            self.client = QdrantClient(url=target_url)
            self.db_path = None
        else:
            storage_path = Path(path or QDRANT_PATH).resolve()
            storage_path.mkdir(parents=True, exist_ok=True)
            self.db_path = storage_path
            self.client = QdrantClient(path=str(storage_path))

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create Qdrant collection if it does not already exist."""
        try:
            collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.embedding_dim,
                        distance=models.Distance.COSINE,
                    ),
                )
        except Exception as e:
            logger.warning(f"Error ensuring Qdrant collection '{self.collection_name}': {e}")

    def close(self) -> None:
        """Close underlying Qdrant client connection/storage."""
        if hasattr(self, "client") and self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


    def _get_metadata_path(self) -> Optional[Path]:
        if self.db_path:
            return self.db_path / METADATA_FILENAME
        return None

    def _read_metadata(self) -> Dict[str, str]:
        p = self._get_metadata_path()
        if p and p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"last_indexed_sha": "", "chunk_count": "0"}

    def _write_metadata(self, data: Dict[str, str]) -> None:
        p = self._get_metadata_path()
        if p:
            try:
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write metadata: {e}")

    def set_last_indexed_sha(self, sha: str) -> None:
        """Persist the last indexed repository commit SHA."""
        meta = self._read_metadata()
        meta["last_indexed_sha"] = sha
        self._write_metadata(meta)

    def upsert(self, records: List[Dict[str, Any]]) -> int:
        """
        Upsert chunk records into Qdrant collection.
        Each point has deterministic UUID derived from chunk_id.
        """
        if not records:
            return 0

        points: List[models.PointStruct] = []
        for rec in records:
            vec = rec.get("vector")
            if not vec or len(vec) != self.embedding_dim:
                raise ValueError(
                    f"Vector dimension mismatch: expected {self.embedding_dim}, got {len(vec) if vec else 0}"
                )

            chunk_id = str(rec["chunk_id"])
            point_id = chunk_id_to_uuid(chunk_id)
            payload = {
                "chunk_id": chunk_id,
                "file_path": str(rec.get("file_path", "")).replace("\\", "/"),
                "symbol_name": str(rec.get("symbol_name", "")),
                "code_snippet": str(rec.get("code_snippet", "")),
                "source_type": str(rec.get("source_type", "code")),
                "language": str(rec.get("language", "")),
                "start_line": int(rec.get("start_line", 0)),
                "end_line": int(rec.get("end_line", 0)),
                "commit_sha": str(rec.get("commit_sha", "")),
                "content_hash": str(rec.get("content_hash", "")),
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=[float(x) for x in vec],
                    payload=payload,
                )
            )

        # Batch upsert in chunks of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )

        self._update_chunk_count_meta()
        return len(points)

    def delete_by_file_paths(self, file_paths: List[str]) -> None:
        """Delete all chunk points originating from specified file paths."""
        if not file_paths:
            return

        normalized_paths = [fp.replace("\\", "/") for fp in file_paths]
        conditions = [
            models.FieldCondition(key="file_path", match=models.MatchValue(value=p))
            for p in normalized_paths
        ]

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(should=conditions)
                ),
                wait=True,
            )
        except Exception as e:
            logger.debug(f"Delete by file_paths error: {e}")

        self._update_chunk_count_meta()

    def delete_by_chunk_ids(self, chunk_ids: List[str]) -> None:
        """Delete points by chunk ID."""
        if not chunk_ids:
            return

        point_ids = [chunk_id_to_uuid(cid) for cid in chunk_ids]
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=point_ids),
                wait=True,
            )
        except Exception as e:
            logger.debug(f"Delete by chunk_ids error: {e}")

        self._update_chunk_count_meta()

    def _update_chunk_count_meta(self) -> None:
        count = self.count()
        meta = self._read_metadata()
        meta["chunk_count"] = str(count)
        self._write_metadata(meta)

    def count(self) -> int:
        """Return total number of chunks currently stored in Qdrant collection."""
        try:
            res = self.client.count(collection_name=self.collection_name, exact=True)
            return res.count
        except Exception:
            return 0

    def query(
        self,
        embedding: List[float],
        top_k: int = 8,
        filters: Optional[Dict[str, str]] = None,
    ) -> List[Candidate]:
        """
        Input:
            embedding: List[float] -- output of embed_text()
            top_k: int -- max results to return, default 8
            filters: Optional[Dict[str, str]] -- e.g. {"source_type": "code"}

        Output:
            List[Candidate] -- sorted descending by similarity.
        """
        if not embedding or len(embedding) != self.embedding_dim:
            logger.warning(
                f"Query embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {len(embedding) if embedding else 0}"
            )
            return []

        query_filter = None
        if filters:
            conditions = []
            for k, v in filters.items():
                conditions.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
            if conditions:
                query_filter = models.Filter(must=conditions)

        try:
            search_res = self.client.query_points(
                collection_name=self.collection_name,
                query=embedding,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            points = search_res.points
        except Exception as e:
            logger.error(f"Qdrant query error: {e}")
            return []

        candidates: List[Candidate] = []
        for p in points:
            payload = p.payload or {}
            score = float(p.score)
            # Normalize cosine similarity score to [0.0, 1.0]
            similarity = max(0.0, min(1.0, score))

            candidates.append(
                Candidate(
                    chunk_id=str(payload.get("chunk_id", str(p.id))),
                    file_path=str(payload.get("file_path", "")),
                    symbol_name=str(payload.get("symbol_name", "")),
                    code_snippet=str(payload.get("code_snippet", "")),
                    source_type="code" if payload.get("source_type") == "code" else "wiki",
                    similarity=round(similarity, 4),
                )
            )

        candidates.sort(key=lambda c: c.similarity, reverse=True)
        return candidates

    def get_index_status(self) -> Dict[str, str]:
        """
        Output: Dict with at least {"last_indexed_sha": str, "chunk_count": str}
        """
        meta = self._read_metadata()
        meta["chunk_count"] = str(self.count())
        return meta


# Alias for backward compatibility / generic import
VectorStoreImpl = QdrantVectorStore


