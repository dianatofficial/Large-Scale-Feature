import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)

class BaseVectorSink(ABC):
    """Abstract base interface for bulk vector database ingestion sinks."""

    @abstractmethod
    def create_collection_if_not_exists(self, collection_name: str, vector_dim: int) -> bool:
        pass

    @abstractmethod
    def upsert_batch(self, collection_name: str, ids: List[str], vectors: np.ndarray, payloads: List[Dict[str, Any]]) -> int:
        pass

class MockVectorSink(BaseVectorSink):
    """In-memory validation sink for testing and standalone simulation."""

    def __init__(self):
        self.collections: Dict[str, Dict[str, Any]] = {}

    def create_collection_if_not_exists(self, collection_name: str, vector_dim: int) -> bool:
        if collection_name not in self.collections:
            self.collections[collection_name] = {
                "vector_dim": vector_dim,
                "points": {}
            }
            logger.info("Created mock collection: %s (dim=%d)", collection_name, vector_dim)
        return True

    def upsert_batch(self, collection_name: str, ids: List[str], vectors: np.ndarray, payloads: List[Dict[str, Any]]) -> int:
        if collection_name not in self.collections:
            self.create_collection_if_not_exists(collection_name, vectors.shape[1] if vectors.ndim > 1 else len(vectors))

        coll = self.collections[collection_name]["points"]
        for idx, vec, payload in zip(ids, vectors, payloads):
            coll[idx] = {"vector": vec, "payload": payload}

        logger.info("Upserted %d points to mock collection %s (Total=%d)", len(ids), collection_name, len(coll))
        return len(ids)

class QdrantVectorSink(BaseVectorSink):
    """Direct bulk ingestion sink for Qdrant Vector Database."""

    def __init__(self, url: str = "http://localhost:6333", api_key: Optional[str] = None):
        self.url = url
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                self._client = QdrantClient(url=self.url, api_key=self.api_key)
            except ImportError:
                logger.warning("qdrant-client package not installed. Qdrant sink unavailable.")
                return None
        return self._client

    def create_collection_if_not_exists(self, collection_name: str, vector_dim: int) -> bool:
        client = self._get_client()
        if client is None:
            return False
        try:
            from qdrant_client.models import VectorParams, Distance
            collections = [c.name for c in client.get_collections().collections]
            if collection_name not in collections:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE)
                )
                logger.info("Created Qdrant collection: %s (dim=%d)", collection_name, vector_dim)
            return True
        except Exception as exc:
            logger.error("Failed to create Qdrant collection: %s", exc)
            return False

    def upsert_batch(self, collection_name: str, ids: List[str], vectors: np.ndarray, payloads: List[Dict[str, Any]]) -> int:
        client = self._get_client()
        if client is None:
            return 0
        try:
            from qdrant_client.models import PointStruct
            points = [
                PointStruct(id=i, vector=v.tolist() if isinstance(v, np.ndarray) else v, payload=p)
                for i, v, p in zip(ids, vectors, payloads)
            ]
            client.upsert(collection_name=collection_name, points=points)
            logger.info("Upserted %d points to Qdrant collection %s", len(points), collection_name)
            return len(points)
        except Exception as exc:
            logger.error("Failed to upsert points to Qdrant: %s", exc)
            return 0

class FaissIndexSink:
    """Local dense vector index constructor using FAISS or pure NumPy cosine search fallback."""

    def __init__(self, vector_dim: int = 384):
        self.vector_dim = vector_dim
        self.ids: List[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.payloads: List[Dict[str, Any]] = []

    def add_vectors(self, ids: List[str], vectors: np.ndarray, payloads: Optional[List[Dict[str, Any]]] = None) -> None:
        self.ids.extend(ids)
        if self.vectors is None:
            self.vectors = vectors.astype(np.float32)
        else:
            self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
        if payloads:
            self.payloads.extend(payloads)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Executes Cosine similarity search over staged vectors."""
        if self.vectors is None or len(self.ids) == 0:
            return []

        q = query_vector.flatten().astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-12:
            q = q / q_norm

        v_norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        v_norms = np.maximum(v_norms, 1e-12)
        normalized_vectors = self.vectors / v_norms

        scores = np.dot(normalized_vectors, q)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "id": self.ids[idx],
                "score": float(scores[idx]),
                "payload": self.payloads[idx] if idx < len(self.payloads) else {}
            })
        return results
