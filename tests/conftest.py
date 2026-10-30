"""Shared pytest configuration and autouse fixtures for AI Interview Platform tests.

Provides mocked services for external APIs (Ollama embedding, Pinecone vector storage)
so the entire test suite runs offline, deterministically, and without network dependencies.
"""

from typing import List, Optional
import pytest

from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.services.chunking_service import TextChunk
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.pinecone_service import PineconeService, get_pinecone_service
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

MOCK_EMBEDDING_DIM = 768

_test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
Base.metadata.create_all(bind=_test_engine)
_TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_test_engine)


def _get_test_db():
    db = _TestSessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


class DefaultMockEmbeddingService(EmbeddingService):
    """Default mock embedding service returning deterministic 768-dim float vectors."""

    async def embed_chunks(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[List[float]]:
        if not texts:
            return []
        return [[0.01 * (i % 10) for i in range(MOCK_EMBEDDING_DIM)] for _ in texts]

    async def embed_text(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> List[float]:
        if not text or not text.strip():
            from app.services.embedding_service import EmbeddingResponseError
            raise EmbeddingResponseError("Cannot generate embedding for empty text.")
        return [0.01 * (i % 10) for i in range(MOCK_EMBEDDING_DIM)]


class DefaultMockPineconeService(PineconeService):
    """Default mock Pinecone service returning successful upserts and isolated queries."""

    def __init__(self):
        super().__init__(api_key="mock-test-key", index_name="ai-interview-prep-v2")
        self.upserted_records = []

    async def upsert_document_chunks(
        self,
        document_id: str,
        chunks: List[TextChunk],
        vectors: List[List[float]],
        filename: str,
        namespace: Optional[str] = None,
        batch_size: int = 100,
    ) -> int:
        for chunk, vec in zip(chunks, vectors):
            self.upserted_records.append({
                "id": f"{document_id}#chunk_{chunk.index}",
                "document_id": document_id,
                "source_id": document_id,
                "chunk_index": chunk.index,
                "filename": filename,
                "text": chunk.text,
            })
        return len(chunks)

    async def verify_index(self):
        return {
            "index_name": "ai-interview-prep-v2",
            "dimension": 768,
            "metric": "cosine",
            "status": "ready",
        }

    async def delete_document(self, document_id: str, namespace: Optional[str] = None) -> bool:
        self.upserted_records = [r for r in self.upserted_records if r["document_id"] != document_id]
        return True

    async def query_similar_chunks(
        self,
        query_vector: List[float],
        document_id: Optional[str] = None,
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
        filter_dict: Optional[dict] = None,
    ) -> List[dict]:
        records = [r for r in self.upserted_records if not document_id or r.get("document_id") == document_id or r.get("source_id") == document_id]
        if not records and document_id:
            return [
                {
                    "id": f"{document_id}#chunk_0",
                    "score": 0.89,
                    "document_id": document_id,
                    "chunk_index": 0,
                    "filename": "candidate_resume.pdf",
                    "text": "Senior Software Engineer with 5+ years building FastAPI microservices and distributed backend systems.",
                    "metadata": {
                        "document_id": document_id,
                        "chunk_index": 0,
                        "filename": "candidate_resume.pdf",
                        "text": "Senior Software Engineer with 5+ years building FastAPI microservices and distributed backend systems.",
                    },
                }
            ]
        k = top_k or 4
        return [
            {
                "id": r["id"],
                "score": 0.92 - (i * 0.04),
                "document_id": r["document_id"],
                "chunk_index": r.get("chunk_index", 0),
                "filename": r.get("filename", "resume.pdf"),
                "text": r.get("text", ""),
                "metadata": r,
            }
            for i, r in enumerate(records[:k])
        ]


@pytest.fixture(autouse=True)
def setup_default_service_mocks():
    """Autouse fixture ensuring that tests calling external endpoints have working default mocks."""
    # Only set if not already overridden by a specific test
    original_overrides = dict(app.dependency_overrides)
    if get_embedding_service not in app.dependency_overrides:
        app.dependency_overrides[get_embedding_service] = lambda: DefaultMockEmbeddingService()
    if get_pinecone_service not in app.dependency_overrides:
        app.dependency_overrides[get_pinecone_service] = lambda: DefaultMockPineconeService()
    if get_db not in app.dependency_overrides:
        app.dependency_overrides[get_db] = _get_test_db

    yield

    app.dependency_overrides = original_overrides
