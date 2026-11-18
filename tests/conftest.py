"""Shared pytest configuration and autouse fixtures for AI Interview Platform tests.

Provides mocked services for external APIs (Ollama embedding, Pinecone vector storage)
so the entire test suite runs offline, deterministically, and without network dependencies.
"""

from typing import List, Optional
import pytest

from app.api.dependencies.auth import get_current_user, require_active_user
from app.core.rate_limit import reset_auth_rate_limiter
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.models.auth import User
from app.services.chunking_service import TextChunk
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.pinecone_service import PineconeService, get_pinecone_service
from sqlalchemy import create_engine, select
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

DEFAULT_TEST_USER_ID = "test-user-0001"


def _ensure_default_test_user() -> User:
    db = _TestSessionLocal()
    try:
        user = db.execute(select(User).where(User.id == DEFAULT_TEST_USER_ID)).scalars().first()
        if user is None:
            user = User(
                id=DEFAULT_TEST_USER_ID,
                name="Default Test User",
                email="default-test@example.com",
                password_hash=None,
                is_active=True,
                email_verified=True,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        return user
    finally:
        db.close()


_DEFAULT_TEST_USER = _ensure_default_test_user()


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "real_auth: run test without default auth dependency overrides",
    )


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
        user_id: Optional[str] = None,
        extra_metadata_list: Optional[List[dict]] = None,
    ) -> int:
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
            record = {
                "id": f"{document_id}#chunk_{chunk.index}",
                "document_id": document_id,
                "source_id": document_id,
                "chunk_index": chunk.index,
                "filename": filename,
                "text": chunk.text,
            }
            if extra_metadata_list and idx < len(extra_metadata_list) and extra_metadata_list[idx]:
                for k, v in extra_metadata_list[idx].items():
                    if k not in {"document_id", "source_id", "user_id"} and v is not None:
                        record[k] = v
            if user_id is not None:
                record["user_id"] = user_id
            self.upserted_records.append(record)
        return len(chunks)

    async def verify_index(self):
        return {
            "index_name": "ai-interview-prep-v2",
            "dimension": 768,
            "metric": "cosine",
            "status": "ready",
        }

    async def delete_document(
        self,
        document_id: str,
        namespace: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        self.upserted_records = [
            r
            for r in self.upserted_records
            if not (r["document_id"] == document_id and (user_id is None or r.get("user_id") == user_id))
        ]
        return True

    async def query_similar_chunks(
        self,
        query_vector: List[float],
        document_id: Optional[str] = None,
        top_k: Optional[int] = None,
        namespace: Optional[str] = None,
        filter_dict: Optional[dict] = None,
        user_id: Optional[str] = None,
    ) -> List[dict]:
        doc_records = [
            r
            for r in self.upserted_records
            if not document_id or r.get("document_id") == document_id or r.get("source_id") == document_id
        ]
        records = [
            r
            for r in doc_records
            if (user_id is None or r.get("user_id") == user_id)
            and (
                not filter_dict
                or all(
                    r.get(fk) == (fv.get("$eq") if isinstance(fv, dict) else fv)
                    for fk, fv in filter_dict.items()
                )
            )
        ]
        if not doc_records and document_id:
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
def setup_default_service_mocks(request):
    """Autouse fixture ensuring that tests calling external endpoints have working default mocks."""
    reset_auth_rate_limiter()
    original_overrides = dict(app.dependency_overrides)
    if get_embedding_service not in app.dependency_overrides:
        app.dependency_overrides[get_embedding_service] = lambda: DefaultMockEmbeddingService()
    if get_pinecone_service not in app.dependency_overrides:
        app.dependency_overrides[get_pinecone_service] = lambda: DefaultMockPineconeService()
    if get_db not in app.dependency_overrides:
        app.dependency_overrides[get_db] = _get_test_db

    module_name = getattr(request.module, "__name__", "")
    use_real_auth = (
        request.node.get_closest_marker("real_auth") is not None
        or "test_auth" in module_name
        or "test_authorization" in module_name
    )
    if not use_real_auth:
        if get_current_user not in app.dependency_overrides:
            app.dependency_overrides[get_current_user] = lambda: _DEFAULT_TEST_USER
        if require_active_user not in app.dependency_overrides:
            app.dependency_overrides[require_active_user] = lambda: _DEFAULT_TEST_USER

    yield

    app.dependency_overrides = original_overrides
    reset_auth_rate_limiter()
