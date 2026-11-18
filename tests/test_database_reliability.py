"""Phase 2 Production Database Reliability tests.

Covers:
1. Missing DATABASE_URL in production (Settings, lifespan startup, get_db_optional).
2. SQLAlchemy connection pool configuration & engine disposal on shutdown.
3. Database connection failures (session init failure, OperationalError on endpoints).
4. Commit failures in get_db and get_db_optional (rollback + propagation).
5. Repository persistence failures in InterviewRepository and SourceRepository.
6. Production in-memory fallback prevention in SessionService, SourceService, and RAGQuestionService.
7. Failed source status persistence via independent transactions (URL, PDF, GitHub URL).
8. Production-mode end-to-end interview session, PDF, URL, Pinecone, and GitHub MCP workflows.
"""

import io
import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import httpx
import pypdf
from pydantic import ValidationError
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings, settings
from app.db.base import Base
from app.db.session import (
    DatabaseConfigurationError,
    DatabaseConnectionError,
    dispose_engine,
    get_db,
    get_db_optional,
    get_engine,
    reset_engine,
    validate_database_configuration,
)
from app.main import app
from app.models.interview import DocumentSource, InterviewSession
from app.repositories.interview_repository import InterviewRepository
from app.repositories.source_repository import SourceRepository
from app.schemas.document import SourceProcessingStatus, SourceType
from app.schemas.github import RepoReadmeResponse
from app.schemas.interview import (
    DifficultyLevel,
    EvaluationRating,
    ExperienceLevel,
    InterviewEvaluationResponse,
    InterviewQuestionResponse,
    QuestionType,
)
from app.schemas.session import CreateSessionRequest
from app.services.embedding_service import EmbeddingConnectionError, get_embedding_service
from app.services.github_mcp_service import GitHubMCPService
from app.services.ollama_service import OllamaService, get_ollama_service
from app.services.pinecone_service import PineconeConnectionError, get_pinecone_service
from app.services.rag_question_service import get_rag_question_service
from app.services.session_service import SessionService, get_session_service
from app.services.source_service import SourceNotFoundError, SourceService, get_source_service
from app.services.url_service import URLService


def _create_sample_pdf(text: str = "Senior Backend Engineer with PostgreSQL expertise") -> bytes:
    """Create a valid 1-page PDF for upload tests."""
    writer = pypdf.PdfWriter()
    page_stream = f"BT\n/F1 12 Tf\n100 700 Td\n({text}) Tj\nET".encode("latin-1")
    raw_pdf = (
        f"%PDF-1.4\n"
        f"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
        f"2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n"
        f"3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources <</Font <</F1 5 0 R>>>>>> endobj\n"
        f"4 0 obj <</Length {len(page_stream)}>> stream\n"
        f"{page_stream.decode('latin-1')}\n"
        f"endstream endobj\n"
        f"5 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj\n"
        f"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000056 00000 n \n0000000111 00000 n \n0000000216 00000 n \n0000000300 00000 n \n"
        f"trailer <</Size 6 /Root 1 0 R>>\nstartxref\n380\n%%EOF"
    ).encode("latin-1")
    reader = pypdf.PdfReader(io.BytesIO(raw_pdf))
    writer.add_page(reader.pages[0])
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


@pytest.fixture
def isolated_db_engine():
    """Provide an isolated SQLite StaticPool engine and clean tables per test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def isolated_db(isolated_db_engine):
    """Provide an isolated SQLAlchemy Session bound to isolated_db_engine."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=isolated_db_engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================================================================
# 1. Production Database Configuration & Lifespan Tests
# ============================================================================


def test_settings_require_database_url_in_production():
    """Settings validation fails immediately when APP_ENV=production and DATABASE_URL is missing."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(APP_ENV="production", DATABASE_URL=None, _env_file=None)
    assert "DATABASE_URL is required when APP_ENV=production" in str(exc_info.value)

    with pytest.raises(ValidationError):
        Settings(APP_ENV="production", DATABASE_URL="   ", _env_file=None)

    # Local development continues to allow missing DATABASE_URL
    dev_settings = Settings(APP_ENV="development", DATABASE_URL=None, _env_file=None)
    assert dev_settings.DATABASE_URL is None
    assert dev_settings.is_production is False


def test_startup_fails_in_production_when_database_url_missing(monkeypatch):
    """FastAPI lifespan startup must fail if APP_ENV=production and DATABASE_URL is missing."""
    monkeypatch.setattr(settings, "APP_ENV", "production")
    monkeypatch.setattr(settings, "DATABASE_URL", None)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    reset_engine()

    with pytest.raises(DatabaseConfigurationError) as exc_info:
        with TestClient(app):
            pass

    assert "DATABASE_URL is required when APP_ENV=production" in str(exc_info.value)


def test_startup_validates_and_shutdown_disposes_engine(monkeypatch):
    """FastAPI lifespan initializes engine on startup and disposes it on shutdown."""
    reset_engine()
    monkeypatch.setattr(settings, "APP_ENV", "production")
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql://user:secret@localhost:5432/testdb")
    monkeypatch.setattr(settings, "JWT_SECRET_KEY", "test-production-jwt-secret-key-minimum-32-chars")

    dispose_called = False
    original_dispose = dispose_engine

    def tracking_dispose():
        nonlocal dispose_called
        dispose_called = True
        original_dispose()

    monkeypatch.setattr("app.main.dispose_engine", tracking_dispose)

    with TestClient(app):
        # Engine should be initialized during startup when DB is configured
        assert get_engine() is not None

    assert dispose_called is True
    reset_engine()


def test_engine_uses_configured_pool_settings(monkeypatch):
    """SQLAlchemy engine applies DB_POOL_SIZE, DB_MAX_OVERFLOW, DB_POOL_TIMEOUT, DB_POOL_RECYCLE."""
    reset_engine()
    monkeypatch.setattr(settings, "DB_POOL_SIZE", 12)
    monkeypatch.setattr(settings, "DB_MAX_OVERFLOW", 20)
    monkeypatch.setattr(settings, "DB_POOL_TIMEOUT", 45)
    monkeypatch.setattr(settings, "DB_POOL_RECYCLE", 600)
    monkeypatch.setattr(settings, "DB_POOL_PRE_PING", True)

    engine = get_engine("postgresql://user:pass@localhost:5432/appdb")
    try:
        assert engine.pool.size() == 12
        assert engine.pool._max_overflow == 20
        assert engine.pool._timeout == 45
        assert engine.pool._recycle == 600
        assert engine.pool._pre_ping is True
    finally:
        engine.dispose()
        reset_engine()


def test_validate_database_configuration_allows_dev_without_url(monkeypatch):
    """validate_database_configuration succeeds in development without DATABASE_URL."""
    monkeypatch.setattr(settings, "APP_ENV", "development")
    monkeypatch.setattr(settings, "DATABASE_URL", None)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    validate_database_configuration()


# ============================================================================
# 2. Database Connection Failure & Commit Failure Tests
# ============================================================================


def test_get_db_optional_missing_url_raises_in_production(monkeypatch):
    """get_db_optional must raise DatabaseConfigurationError in production when DATABASE_URL is unset."""
    monkeypatch.setattr(settings, "APP_ENV", "production")
    monkeypatch.setattr(settings, "DATABASE_URL", None)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(DatabaseConfigurationError):
        next(get_db_optional())


def test_get_db_optional_connection_failure_propagates_and_fails_api(monkeypatch):
    """Database connection failures in get_db_optional must NOT silently return 200 OK."""
    monkeypatch.setattr(settings, "APP_ENV", "production")
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql://user:pass@localhost:5432/db")

    def failing_session_factory():
        raise OperationalError("SELECT 1", {}, Exception("Connection refused"))

    monkeypatch.setattr("app.db.session.get_session_factory", lambda: failing_session_factory)

    with pytest.raises(DatabaseConnectionError):
        next(get_db_optional())

    # Remove conftest's get_db override so real get_db_optional runs during the HTTP request
    saved_get_db = app.dependency_overrides.pop(get_db, None)
    try:
        client = TestClient(app)
        resp = client.get("/api/v1/interview/sessions")
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()
    finally:
        if saved_get_db is not None:
            app.dependency_overrides[get_db] = saved_get_db


def test_get_db_optional_commit_failure_rolls_back_and_propagates(isolated_db_engine, monkeypatch):
    """Commit failure in get_db_optional must trigger rollback and propagate error to API caller."""
    rolled_back = False

    class CommitFailingSession(Session):
        def commit(self):
            raise OperationalError("COMMIT", {}, Exception("Disk full or connection dropped"))

        def rollback(self):
            nonlocal rolled_back
            rolled_back = True
            super().rollback()

    FailingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=isolated_db_engine,
        class_=CommitFailingSession,
    )

    monkeypatch.setattr("app.db.session.is_database_configured", lambda: True)
    monkeypatch.setattr("app.db.session.get_session_factory", lambda: FailingSessionLocal)

    gen = get_db_optional()
    _db = next(gen)
    with pytest.raises(OperationalError):
        next(gen)
    assert rolled_back is True

    # Verify HTTP endpoint returns 503/500 instead of silent 201 Created when commit fails
    rolled_back = False
    saved_get_db = app.dependency_overrides.pop(get_db, None)
    try:
        client = TestClient(app)
        resp = client.post(
            "/api/v1/interview/sessions",
            json={
                "role": "Backend Engineer",
                "experience_level": "Mid-Level",
                "difficulty": "Medium",
                "topic": "PostgreSQL",
                "question_type": "Technical",
                "total_questions": 2,
            },
        )
        assert resp.status_code in (500, 503)
        assert rolled_back is True
    finally:
        if saved_get_db is not None:
            app.dependency_overrides[get_db] = saved_get_db


def test_get_db_commit_failure_rolls_back_and_propagates(isolated_db_engine, monkeypatch):
    """Commit failure in get_db must roll back and propagate the exception."""
    rolled_back = False

    class CommitFailingSession(Session):
        def commit(self):
            raise SQLAlchemyError("Simulated commit error")

        def rollback(self):
            nonlocal rolled_back
            rolled_back = True
            super().rollback()

    FailingSessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=isolated_db_engine,
        class_=CommitFailingSession,
    )

    monkeypatch.setattr("app.db.session.is_database_configured", lambda: True)
    monkeypatch.setattr("app.db.session.get_session_factory", lambda: FailingSessionLocal)

    gen = get_db()
    _db = next(gen)
    with pytest.raises(SQLAlchemyError):
        next(gen)
    assert rolled_back is True


# ============================================================================
# 3. Repository Persistence Failure Tests
# ============================================================================


def test_interview_repository_propagates_errors_and_rolls_back():
    """InterviewRepository must roll back and re-raise on DB failures instead of swallowing them."""
    repo = InterviewRepository()
    mock_db = MagicMock(spec=Session)
    mock_db.flush.side_effect = OperationalError("INSERT", {}, Exception("DB down"))
    mock_db.execute.side_effect = OperationalError("SELECT", {}, Exception("DB down"))

    req = CreateSessionRequest(
        role="Backend Engineer",
        experience_level=ExperienceLevel.MID_LEVEL,
        difficulty=DifficultyLevel.MEDIUM,
        topic="Databases",
        question_type=QuestionType.TECHNICAL,
        total_questions=3,
    )

    with pytest.raises(OperationalError):
        repo.create_session(
            mock_db,
            session_id="sess_fail_1",
            payload=req,
            total_questions=3,
            current_question_number=0,
            questions_list=[],
        )
    mock_db.rollback.assert_called_once()

    mock_db.reset_mock()
    with pytest.raises(OperationalError):
        repo.get_session_by_id(mock_db, "sess_fail_1")

    with pytest.raises(OperationalError):
        repo.list_sessions(mock_db)

    mock_db.reset_mock()
    q = InterviewQuestionResponse(
        question="Explain MVCC.",
        expected_answer_points=["Row versioning"],
        follow_up_question="What is vacuum?",
        difficulty=DifficultyLevel.MEDIUM,
        topic="PostgreSQL",
    )
    with pytest.raises(OperationalError):
        repo.add_question(
            mock_db,
            session_id="sess_fail_1",
            question_number=1,
            question_type="Technical",
            question_data=q,
        )
    mock_db.rollback.assert_called_once()

    mock_db.reset_mock()
    eval_resp = InterviewEvaluationResponse(
        overall_score=80,
        rating=EvaluationRating.GOOD,
        correct_points=["Point"],
        missing_points=[],
        strengths=["Good"],
        improvements=[],
        sample_answer="Sample",
    )
    with pytest.raises(OperationalError):
        repo.record_answer_evaluation(
            mock_db,
            session_id="sess_fail_1",
            question_number=1,
            candidate_answer="My answer",
            evaluation=eval_resp,
            completed_questions=1,
        )
    mock_db.rollback.assert_called_once()

    mock_db.reset_mock()
    mock_db.get.side_effect = OperationalError("SELECT", {}, Exception("DB down"))
    with pytest.raises(OperationalError):
        repo.update_status(mock_db, session_id="sess_fail_1", status_value="completed")
    mock_db.rollback.assert_called_once()


def test_source_repository_propagates_errors_and_rolls_back():
    """SourceRepository must roll back and re-raise on DB failures instead of swallowing them."""
    repo = SourceRepository()
    mock_db = MagicMock(spec=Session)
    mock_db.flush.side_effect = OperationalError("INSERT", {}, Exception("DB down"))
    mock_db.get.side_effect = OperationalError("SELECT", {}, Exception("DB down"))
    mock_db.execute.side_effect = OperationalError("SELECT", {}, Exception("DB down"))

    with pytest.raises(OperationalError):
        repo.create(
            mock_db,
            source_id="src_fail_1",
            source_type="url",
            title="Test",
            url="https://example.com",
            filename=None,
            status="processing",
            character_count=100,
            chunk_count=0,
            vector_count=0,
            error_message=None,
        )
    mock_db.rollback.assert_called_once()

    mock_db.reset_mock()
    with pytest.raises(OperationalError):
        repo.update(mock_db, source_id="src_fail_1", status="completed")
    mock_db.rollback.assert_called_once()

    with pytest.raises(OperationalError):
        repo.get_by_id(mock_db, "src_fail_1")

    with pytest.raises(OperationalError):
        repo.list_recent(mock_db)


# ============================================================================
# 4. Production In-Memory Fallback Prevention Tests
# ============================================================================


def test_session_service_disables_in_memory_fallback_in_production(isolated_db, monkeypatch):
    """In production, SessionService requires db and never falls back to process-local dicts."""
    monkeypatch.setattr(settings, "APP_ENV", "production")
    service = SessionService()

    req = CreateSessionRequest(
        role="Backend Engineer",
        experience_level=ExperienceLevel.SENIOR,
        difficulty=DifficultyLevel.HARD,
        topic="Distributed Systems",
        question_type=QuestionType.TECHNICAL,
        total_questions=2,
    )

    # 1. Calling without db in production fails fast
    with pytest.raises(DatabaseConfigurationError):
        service.create_session(req, db=None)
    with pytest.raises(DatabaseConfigurationError):
        service.get_session("any-id", db=None)
    with pytest.raises(DatabaseConfigurationError):
        service.list_sessions(db=None)

    # 2. Creating with db does NOT store persistent state in _dev_sessions
    created = service.create_session(req, db=isolated_db)
    isolated_db.commit()
    assert len(service._dev_sessions) == 0

    # 3. Even if _dev_sessions somehow had an entry, production get_session ignores it
    service._dev_sessions["fake-inmem-id"] = created
    with pytest.raises(Exception) as exc_info:
        service.get_session("fake-inmem-id", db=isolated_db)
    assert "not found" in str(exc_info.value).lower()

    # 4. DB failure on get_session / list_sessions propagates instead of falling back to memory
    broken_db = MagicMock(spec=Session)
    broken_db.execute.side_effect = OperationalError("SELECT", {}, Exception("DB outage"))
    with pytest.raises(OperationalError):
        service.get_session(created.session_id, db=broken_db)
    with pytest.raises(OperationalError):
        service.list_sessions(db=broken_db)


def test_source_service_disables_in_memory_fallback_in_production(isolated_db, monkeypatch):
    """In production, SourceService requires db and never falls back to process-local dicts."""
    monkeypatch.setattr(settings, "APP_ENV", "production")
    service = SourceService()

    # 1. Calling without db in production fails fast
    with pytest.raises(DatabaseConfigurationError):
        service.register_source(
            source_id="src_prod_none",
            source_type=SourceType.URL,
            title="Test",
            db=None,
        )
    with pytest.raises(DatabaseConfigurationError):
        service.get_source("src_prod_none", db=None)
    with pytest.raises(DatabaseConfigurationError):
        service.list_sources(db=None)

    # 2. Registering with db does NOT populate _dev_sources
    registered = service.register_source(
        source_id="src_prod_db_1",
        source_type=SourceType.DOCUMENT,
        title="resume.pdf",
        filename="resume.pdf",
        status=SourceProcessingStatus.COMPLETED,
        db=isolated_db,
    )
    isolated_db.commit()
    assert len(service._dev_sources) == 0

    # 3. Even if _dev_sources has an item, production get_source does not fall back to it
    service._dev_sources["src_inmem_only"] = registered
    with pytest.raises(SourceNotFoundError):
        service.get_source("src_inmem_only", db=isolated_db)

    # 4. DB failure propagates rather than returning _dev_sources
    broken_db = MagicMock(spec=Session)
    broken_db.get.side_effect = OperationalError("SELECT", {}, Exception("DB outage"))
    broken_db.execute.side_effect = OperationalError("SELECT", {}, Exception("DB outage"))
    with pytest.raises(OperationalError):
        service.get_source("src_prod_db_1", db=broken_db)
    with pytest.raises(OperationalError):
        service.list_sources(db=broken_db)


@pytest.mark.anyio
async def test_rag_question_service_validates_db_source_before_temporary_cache(isolated_db, monkeypatch):
    """Even if temporary question cache has an entry, a failed/missing DB source is rejected."""
    monkeypatch.setattr(settings, "APP_ENV", "production")
    source_service = SourceService()
    source_service.register_source(
        source_id="src_failed_cached",
        source_type=SourceType.DOCUMENT,
        title="resume.pdf",
        status=SourceProcessingStatus.FAILED,
        error_message="Pinecone upsert failed",
        db=isolated_db,
    )
    isolated_db.commit()

    rag_service = get_rag_question_service()
    rag_service.source_service = source_service
    rag_service.reset_store()

    # Pollute temporary cache with stale entry
    from app.schemas.document import SourceQuestionItem, SourceQuestionsRequest, SourceQuestionsResponse
    from app.services.source_service import SourceNotReadyError

    rag_service._cached_questions["src_failed_cached"] = SourceQuestionsResponse(
        source_id="src_failed_cached",
        context_grounded=True,
        total_questions=1,
        questions=[
            SourceQuestionItem(
                id="q_1",
                order=1,
                question="Stale question",
                expected_answer_points=["Point"],
                difficulty=DifficultyLevel.MEDIUM,
                topic="Python",
            )
        ],
        retrieved_chunks=[],
    )

    with pytest.raises(SourceNotReadyError):
        await rag_service.generate_source_questions(
            SourceQuestionsRequest(source_id="src_failed_cached"),
            db=isolated_db,
        )
    rag_service.reset_store()


# ============================================================================
# 5. Failed Source Processing Persistence Tests (Independent Transaction)
# ============================================================================


def test_url_ingestion_failure_persists_failed_status_despite_outer_rollback(isolated_db_engine, monkeypatch):
    """When URL processing fails (e.g. embedding outage), FAILED status is persisted independently."""
    monkeypatch.setattr(settings, "APP_ENV", "production")
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql://user:pass@localhost:5432/testdb")
    get_source_service().reset_store()

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=isolated_db_engine)

    def override_get_db():
        db = SessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    class FailingEmbedding:
        async def embed_chunks(self, *args, **kwargs):
            raise EmbeddingConnectionError("Ollama embedding service unreachable")

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_embedding_service] = lambda: FailingEmbedding()

    mock_html = (
        "<html><head><title>Resilient DB Guide</title></head>"
        "<body><p>PostgreSQL transactions and independent error status persistence test content.</p></body></html>"
    )
    mock_resp = httpx.Response(
        status_code=200,
        content=mock_html.encode("utf-8"),
        headers={"Content-Type": "text/html"},
        request=httpx.Request("GET", "https://example.com/db-guide"),
    )

    try:
        client = TestClient(app)
        with patch.object(URLService, "validate_ssrf_url", return_value="https://example.com/db-guide"):
            with patch("httpx.AsyncClient.get", return_value=mock_resp):
                resp = client.post("/api/v1/sources/url", json={"url": "https://example.com/db-guide"})
        assert resp.status_code == 503

        # Outer request transaction rolled back, yet FAILED status must be persisted in DB!
        list_resp = client.get("/api/v1/sources")
        assert list_resp.status_code == 200
        sources = list_resp.json()
        assert len(sources) == 1
        src = sources[0]
        assert src["status"] == "failed"
        assert "unreachable" in (src["error_message"] or "").lower()

        # Status endpoint returns the persisted failed status
        status_resp = client.get(f"/api/v1/sources/{src['source_id']}/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "failed"
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_embedding_service, None)


def test_pdf_ingestion_failure_persists_failed_status_despite_outer_rollback(isolated_db_engine, monkeypatch):
    """When PDF processing fails at Pinecone upsert, FAILED status is persisted independently."""
    monkeypatch.setattr(settings, "APP_ENV", "production")
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql://user:pass@localhost:5432/testdb")
    get_source_service().reset_store()

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=isolated_db_engine)

    def override_get_db():
        db = SessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    class FailingPinecone:
        async def upsert_document_chunks(self, *args, **kwargs):
            raise PineconeConnectionError("Pinecone index unavailable")

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_pinecone_service] = lambda: FailingPinecone()

    try:
        client = TestClient(app)
        pdf_bytes = _create_sample_pdf("Distributed Systems and PostgreSQL Reliability")
        resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("arch_resume.pdf", pdf_bytes, "application/pdf")},
        )
        assert resp.status_code == 503

        # Verify FAILED status is persisted in DB
        with SessionLocal() as check_db:
            rows = check_db.execute(select(DocumentSource)).scalars().all()
            assert len(rows) == 1
            assert rows[0].status == "failed"
            assert rows[0].filename == "arch_resume.pdf"
            assert "pinecone" in (rows[0].error_message or "").lower()

        status_resp = client.get(f"/api/v1/sources/{rows[0].id}/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "failed"
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_pinecone_service, None)


# ============================================================================
# 6. Production-Mode End-to-End Workflows (PDF, URL, GitHub MCP, Sessions)
# ============================================================================


def test_production_mode_end_to_end_workflows_with_db(isolated_db_engine, monkeypatch):
    """Verify PDF, URL, GitHub MCP, and interview session workflows operate cleanly in production mode."""
    monkeypatch.setattr(settings, "APP_ENV", "production")
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql://user:pass@localhost:5432/testdb")
    get_source_service().reset_store()
    get_session_service().reset_store()
    get_rag_question_service().reset_store()

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=isolated_db_engine)

    def override_get_db():
        db = SessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    class MockOllama(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            return json.dumps(
                {
                    "question": "How does SQLAlchemy pool_pre_ping prevent stale connection errors?",
                    "expected_answer_points": [
                        "Issues lightweight ping before checkout",
                        "Recycles broken connections transparently",
                    ],
                    "follow_up_question": "How does pool_recycle complement pool_pre_ping?",
                    "difficulty": "Medium",
                    "topic": "PostgreSQL",
                }
            )

        async def evaluate_interview_answer(self, question, expected_answer_points, candidate_answer):
            return json.dumps(
                {
                    "overall_score": 92,
                    "rating": "Excellent",
                    "correct_points": ["Explained checkout validation"],
                    "missing_points": [],
                    "strengths": ["Clear explanation"],
                    "improvements": [],
                    "sample_answer": "pool_pre_ping checks connection liveness before use.",
                }
            )

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_ollama_service] = lambda: MockOllama()

    try:
        client = TestClient(app)

        # 1. PDF upload workflow in production mode
        pdf_bytes = _create_sample_pdf("Senior Python & PostgreSQL Architect")
        pdf_resp = client.post(
            "/api/v1/documents/upload",
            files={"file": ("prod_resume.pdf", pdf_bytes, "application/pdf")},
        )
        assert pdf_resp.status_code == 200
        doc_id = pdf_resp.json()["source_id"]
        assert client.get(f"/api/v1/sources/{doc_id}/status").json()["status"] == "completed"

        # 2. GitHub MCP URL workflow in production mode
        readme_text = "# FastAPI Backend\n\nProduction-ready FastAPI + PostgreSQL service architecture."
        mock_readme = RepoReadmeResponse(
            owner="acme",
            repo="backend",
            filename="README.md",
            size_bytes=len(readme_text),
            content=readme_text,
            html_url="https://github.com/acme/backend/blob/main/README.md",
        )
        with patch.object(URLService, "validate_ssrf_url", return_value="https://github.com/acme/backend"):
            with patch.object(GitHubMCPService, "get_readme", return_value=mock_readme):
                gh_resp = client.post(
                    "/api/v1/sources/url",
                    json={"url": "https://github.com/acme/backend"},
                )
        assert gh_resp.status_code == 200
        gh_source_id = gh_resp.json()["source_id"]
        assert gh_resp.json()["source_type"] == "github"

        # 3. Session creation, duplicate prevention, question, answer, completion in DB
        sess_payload = {
            "source_id": gh_source_id,
            "role": "Backend Engineer",
            "experience_level": "Senior",
            "difficulty": "Medium",
            "topic": "PostgreSQL",
            "question_type": "Technical",
            "total_questions": 2,
        }
        s1 = client.post("/api/v1/interview/sessions", json=sess_payload)
        assert s1.status_code == 201
        session_id = s1.json()["session_id"]

        # Duplicate check hits DB even in production
        s2 = client.post("/api/v1/interview/sessions", json=sess_payload)
        assert s2.status_code == 201
        assert s2.json()["session_id"] == session_id

        nq = client.post(f"/api/v1/interview/sessions/{session_id}/next-question")
        assert nq.status_code == 200

        ans = client.post(
            f"/api/v1/interview/sessions/{session_id}/answers",
            json={"candidate_answer": "pool_pre_ping verifies the DB connection before executing queries."},
        )
        assert ans.status_code == 200
        assert ans.json()["completed_questions"] == 1

        comp = client.post(f"/api/v1/interview/sessions/{session_id}/complete")
        assert comp.status_code == 200
        assert comp.json()["session_status"] == "completed"

        # Verify zero process-local persistent fallback state was stored
        assert len(get_session_service()._dev_sessions) == 0
        assert len(get_source_service()._dev_sources) == 0

        # Verify records are persisted in DB
        with SessionLocal() as check_db:
            db_sess = check_db.get(InterviewSession, session_id)
            assert db_sess is not None
            assert db_sess.status == "completed"
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_ollama_service, None)

