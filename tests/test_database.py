"""
Unit and integration tests for database models, configuration, dependencies, and persistence.
Does not require a live Neon PostgreSQL database to run.
"""

from datetime import datetime, timezone
import json
import os
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db import (
    Base,
    DatabaseConfigurationError,
    get_db,
    is_database_configured,
    mask_database_url,
    normalize_database_url,
)
from app.main import app
from app.models.interview import AnswerEvaluation, InterviewQuestion, InterviewSession
from app.schemas.interview import (
    DifficultyLevel,
    EvaluationRating,
    ExperienceLevel,
    InterviewEvaluationResponse,
    InterviewQuestionResponse,
    QuestionType,
)
from app.schemas.session import CreateSessionRequest, SessionStatus
from app.services.session_service import SessionService

client = TestClient(app)


# ============================================================================
# SQLite In-Memory Database Fixture for Isolated DB Tests
# ============================================================================


@pytest.fixture
def in_memory_db():
    """Create an isolated, in-memory SQLite database with all tables for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


# ============================================================================
# 1. Database Configuration & URL Normalization Tests
# ============================================================================


def test_normalize_database_url_postgresql_prefix():
    """Verify postgresql:// is normalized to postgresql+psycopg:// with sslmode=require."""
    raw = "postgresql://neondb_owner:npg_secret123@ep-cool-fog-12345.us-east-2.aws.neon.tech/neondb"
    normalized = normalize_database_url(raw)
    assert normalized.startswith("postgresql+psycopg://")
    assert "sslmode=require" in normalized
    assert "npg_secret123" in normalized


def test_normalize_database_url_postgres_short_prefix():
    """Verify legacy postgres:// is normalized to postgresql+psycopg:// with sslmode=require."""
    raw = "postgres://user:pass@ep-host.neon.tech/neondb"
    normalized = normalize_database_url(raw)
    assert normalized.startswith("postgresql+psycopg://")
    assert "sslmode=require" in normalized


def test_normalize_database_url_already_has_psycopg():
    """Verify URL already containing postgresql+psycopg:// retains the driver."""
    raw = "postgresql+psycopg://user:pass@ep-host.neon.tech/neondb?sslmode=require"
    normalized = normalize_database_url(raw)
    assert normalized.startswith("postgresql+psycopg://")
    assert "sslmode=require" in normalized


def test_normalize_database_url_preserves_existing_sslmode():
    """Verify sslmode=require is not duplicated if already present."""
    raw = "postgresql://user:pass@ep-host.neon.tech/neondb?sslmode=require"
    normalized = normalize_database_url(raw)
    assert normalized.count("sslmode=require") == 1


def test_normalize_database_url_missing_raises_clear_error():
    """Verify missing or blank DATABASE_URL raises DatabaseConfigurationError."""
    with pytest.raises(DatabaseConfigurationError) as exc_info:
        normalize_database_url("")
    assert "DATABASE_URL is not configured" in str(exc_info.value)

    with pytest.raises(DatabaseConfigurationError) as exc_info:
        normalize_database_url(None)
    assert "DATABASE_URL is not configured" in str(exc_info.value)


def test_mask_database_url_hides_password():
    """Verify passwords in connection strings are strictly masked."""
    secret_url = "postgresql+psycopg://my_user:SUPER_SECRET_PASSWORD@ep-123.neon.tech:5432/neondb?sslmode=require"
    masked = mask_database_url(secret_url)
    assert "SUPER_SECRET_PASSWORD" not in masked
    assert "*****" in masked
    assert "my_user" in masked
    assert "ep-123.neon.tech" in masked


def test_mask_database_url_empty_or_none():
    """Verify mask_database_url handles empty or None inputs safely."""
    assert mask_database_url(None) == "<DATABASE_URL not set>"
    assert mask_database_url("") == "<DATABASE_URL not set>"


# ============================================================================
# 2. Database Model Definition Tests
# ============================================================================


def test_interview_session_model_attributes():
    """Verify InterviewSession table name, columns, and relationships."""
    assert InterviewSession.__tablename__ == "interview_sessions"
    cols = {c.name for c in InterviewSession.__table__.columns}
    expected_cols = {
        "id",
        "role",
        "experience_level",
        "difficulty",
        "topic",
        "question_type",
        "total_questions",
        "completed_questions",
        "current_question_number",
        "status",
        "created_at",
        "updated_at",
    }
    assert expected_cols.issubset(cols)


def test_interview_question_model_attributes():
    """Verify InterviewQuestion table name, columns, and foreign keys."""
    assert InterviewQuestion.__tablename__ == "interview_questions"
    cols = {c.name for c in InterviewQuestion.__table__.columns}
    expected_cols = {
        "id",
        "session_id",
        "order",
        "question",
        "topic",
        "difficulty",
        "question_type",
        "expected_answer_points",
        "follow_up_question",
        "candidate_answer",
        "created_at",
        "updated_at",
    }
    assert expected_cols.issubset(cols)

    # Check foreign key to interview_sessions.id
    fks = list(InterviewQuestion.__table__.foreign_keys)
    assert any(fk.target_fullname == "interview_sessions.id" for fk in fks)


def test_answer_evaluation_model_attributes():
    """Verify AnswerEvaluation table name, columns, and foreign keys."""
    assert AnswerEvaluation.__tablename__ == "answer_evaluations"
    cols = {c.name for c in AnswerEvaluation.__table__.columns}
    expected_cols = {
        "id",
        "question_id",
        "session_id",
        "candidate_answer",
        "score",
        "rating",
        "correct_points",
        "missing_points",
        "strengths",
        "improvements",
        "sample_answer",
        "created_at",
    }
    assert expected_cols.issubset(cols)

    # Check foreign keys to interview_questions.id and interview_sessions.id
    fks = [fk.target_fullname for fk in AnswerEvaluation.__table__.foreign_keys]
    assert "interview_questions.id" in fks
    assert "interview_sessions.id" in fks


# ============================================================================
# 3. Database Dependency (`get_db`) Lifecycle Tests
# ============================================================================


def test_get_db_unconfigured_raises_error(monkeypatch):
    """Verify get_db raises DatabaseConfigurationError when DATABASE_URL is unset."""
    monkeypatch.setattr(settings, "DATABASE_URL", None)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(DatabaseConfigurationError) as exc_info:
        next(get_db())
    assert "DATABASE_URL is not configured" in str(exc_info.value)


def test_get_db_lifecycle_commit_and_close(in_memory_db, monkeypatch):
    """Verify get_db dependency commits on normal completion and closes session."""
    closed = False

    class MockSession(Session):
        def close(self):
            nonlocal closed
            closed = True
            super().close()

    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=in_memory_db.get_bind(), class_=MockSession)
    monkeypatch.setattr("app.db.session.is_database_configured", lambda: True)
    monkeypatch.setattr("app.db.session.get_session_factory", lambda: TestingSessionLocal)

    gen = get_db()
    db_session = next(gen)
    assert isinstance(db_session, Session)
    assert not closed

    with pytest.raises(StopIteration):
        next(gen)
    assert closed


def test_get_db_lifecycle_rollback_on_exception(in_memory_db, monkeypatch):
    """Verify get_db dependency rolls back on unhandled error and closes session."""
    rolled_back = False
    closed = False

    class MockSession(Session):
        def rollback(self):
            nonlocal rolled_back
            rolled_back = True
            super().rollback()

        def close(self):
            nonlocal closed
            closed = True
            super().close()

    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=in_memory_db.get_bind(), class_=MockSession)
    monkeypatch.setattr("app.db.session.is_database_configured", lambda: True)
    monkeypatch.setattr("app.db.session.get_session_factory", lambda: TestingSessionLocal)

    gen = get_db()
    db_session = next(gen)
    assert not rolled_back

    with pytest.raises(RuntimeError):
        gen.throw(RuntimeError("Simulated application error"))

    assert rolled_back
    assert closed


# ============================================================================
# 4. Database Persistence Flow Tests (Session -> Question -> Evaluation -> Complete)
# ============================================================================


def test_database_full_session_persistence_lifecycle(in_memory_db):
    """
    Test full end-to-end interview lifecycle with database persistence:
    1. Create session and verify record in DB
    2. Add question and verify record in DB
    3. Record evaluation and verify record in DB
    4. Complete session and verify status updated in DB
    """
    service = SessionService()

    # 1. Create Session
    create_req = CreateSessionRequest(
        role="Full Stack Engineer",
        experience_level=ExperienceLevel.MID_LEVEL,
        difficulty=DifficultyLevel.MEDIUM,
        topic="PostgreSQL Architecture",
        question_type=QuestionType.TECHNICAL,
        total_questions=3,
    )
    session_resp = service.create_session(create_req, db=in_memory_db)
    session_id = session_resp.session_id

    # Verify session persisted in DB
    db_session = in_memory_db.get(InterviewSession, session_id)
    assert db_session is not None
    assert db_session.id == session_id
    assert db_session.role == "Full Stack Engineer"
    assert db_session.status == "in_progress"
    assert db_session.total_questions == 3
    assert db_session.completed_questions == 0

    # 2. Add Question
    q_data = InterviewQuestionResponse(
        question="How does PostgreSQL handle MVCC and vacuuming?",
        expected_answer_points=[
            "Tuple versions with xmin and xmax",
            "Dead tuples created on UPDATE and DELETE",
            "VACUUM cleans dead tuples and updates visibility map",
        ],
        follow_up_question="What is autovacuum wraparound?",
        difficulty=DifficultyLevel.MEDIUM,
        topic="PostgreSQL Architecture",
    )
    next_q = service.add_question_to_session(session_id, q_data, db=in_memory_db)
    assert next_q.question_number == 1

    # Verify question persisted in DB
    stmt = select(InterviewQuestion).where(InterviewQuestion.session_id == session_id)
    db_questions = in_memory_db.execute(stmt).scalars().all()
    assert len(db_questions) == 1
    assert db_questions[0].order == 1
    assert "MVCC" in db_questions[0].question
    assert len(db_questions[0].expected_answer_points) == 3

    # 3. Record Answer and Evaluation
    eval_resp = InterviewEvaluationResponse(
        overall_score=88,
        rating=EvaluationRating.EXCELLENT,
        correct_points=["Explained xmin/xmax transaction visibility"],
        missing_points=[],
        strengths=["Solid grasp of MVCC storage"],
        improvements=["Mention freeze transaction IDs"],
        sample_answer="PostgreSQL implements MVCC by creating new tuple versions...",
    )
    submit_resp = service.record_answer_and_evaluation(
        session_id=session_id,
        question_number=1,
        candidate_answer="PostgreSQL stores rows with xmin and xmax transaction IDs.",
        evaluation=eval_resp,
        db=in_memory_db,
    )
    assert submit_resp.completed_questions == 1

    # Verify evaluation persisted in DB
    stmt = select(AnswerEvaluation).where(AnswerEvaluation.session_id == session_id)
    db_evals = in_memory_db.execute(stmt).scalars().all()
    assert len(db_evals) == 1
    assert db_evals[0].score == 88
    assert db_evals[0].rating == "Excellent"
    assert "xmin" in db_evals[0].candidate_answer

    # 4. Complete Session
    summary = service.complete_session(session_id, db=in_memory_db)
    assert summary.session_status == SessionStatus.COMPLETED
    assert summary.questions_answered == 1
    assert summary.average_score == 88.0

    # Verify session completed in DB
    db_session_updated = in_memory_db.get(InterviewSession, session_id)
    assert db_session_updated.status == "completed"

    # 5. Retrieve Session from DB via get_session
    fresh_service = SessionService()  # Clean in-memory store
    fetched = fresh_service.get_session(session_id, db=in_memory_db)
    assert fetched.session_id == session_id
    assert fetched.status == SessionStatus.COMPLETED
    assert len(fetched.questions) == 1
    assert fetched.questions[0].evaluation.overall_score == 88


# ============================================================================
# 5. Database Health Check Endpoint Tests (/api/v1/health/db)
# ============================================================================


def test_health_db_unconfigured(monkeypatch):
    """Verify /health/db returns 503 when DATABASE_URL is not set."""
    monkeypatch.setattr("app.db.session.is_database_configured", lambda: False)
    response = client.get("/api/v1/health/db")
    assert response.status_code == 503
    data = response.json()
    assert "DATABASE_URL environment variable is missing" in data["detail"]


def test_health_db_connected(in_memory_db, monkeypatch):
    """Verify /health/db returns 200 when database connection succeeds."""
    monkeypatch.setattr("app.db.session.is_database_configured", lambda: True)
    monkeypatch.setattr("app.db.session.get_engine", lambda: in_memory_db.get_bind())

    response = client.get("/api/v1/health/db")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "postgresql"
    assert data["database_connected"] is True
    assert data["database_configured"] is True


def test_health_db_connection_failure(monkeypatch):
    """Verify /health/db returns 503 without exposing credentials if connection fails."""
    monkeypatch.setattr("app.db.session.is_database_configured", lambda: True)

    class FailingEngine:
        def connect(self):
            raise ConnectionRefusedError("Simulated network timeout to Neon")

    monkeypatch.setattr("app.db.session.get_engine", lambda: FailingEngine())


    response = client.get("/api/v1/health/db")
    assert response.status_code == 503
    data = response.json()
    assert "Could not connect to PostgreSQL" in data["detail"]
    # Verify no secrets or credentials leaked
    assert "password" not in response.text.lower()


def test_root_endpoint_includes_database_health():
    """Verify the root endpoint includes the database health check link."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "database_health" in data
    assert data["database_health"] == "/api/v1/health/db"


def test_api_database_dependency_override_persistence(in_memory_db):
    """
    Verify that API routes persist sessions, questions, evaluations, and completions
    to the database when get_db is overridden with a test DB session.
    """
    from app.services.ollama_service import OllamaService, get_ollama_service

    # Provide DB dependency override
    def override_get_db():
        yield in_memory_db

    class MockOllama(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            return json.dumps({
                "question": "What is an index in PostgreSQL?",
                "expected_answer_points": ["B-Tree structure", "Improves query lookup speed"],
                "follow_up_question": "What is a composite index?",
                "difficulty": "Medium",
                "topic": "PostgreSQL",
            })

        async def evaluate_interview_answer(self, question, expected_answer_points, candidate_answer):
            return json.dumps({
                "overall_score": 90,
                "rating": "Excellent",
                "correct_points": ["Accurate index definition"],
                "missing_points": [],
                "strengths": ["Clear explanation"],
                "improvements": [],
                "sample_answer": "An index is a redundant data structure...",
            })

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_ollama_service] = lambda: MockOllama()

    try:
        # 1. Create Session via API
        create_resp = client.post(
            "/api/v1/interview/sessions",
            json={
                "role": "Database Administrator",
                "experience_level": "Senior",
                "difficulty": "Hard",
                "topic": "PostgreSQL Indexing",
                "question_type": "Technical",
                "total_questions": 2,
            },
        )
        assert create_resp.status_code == 201
        session_id = create_resp.json()["session_id"]

        # Verify InterviewSession record in database
        db_session = in_memory_db.get(InterviewSession, session_id)
        assert db_session is not None
        assert db_session.role == "Database Administrator"

        # 2. Generate Next Question via API
        next_q_resp = client.post(f"/api/v1/interview/sessions/{session_id}/next-question")
        assert next_q_resp.status_code == 200

        # Verify InterviewQuestion record in database
        stmt = select(InterviewQuestion).where(InterviewQuestion.session_id == session_id)
        db_questions = in_memory_db.execute(stmt).scalars().all()
        assert len(db_questions) == 1
        assert "PostgreSQL" in db_questions[0].question

        # 3. Submit Answer via API
        submit_resp = client.post(
            f"/api/v1/interview/sessions/{session_id}/answers",
            json={"candidate_answer": "An index speeds up search using a B-Tree structure."},
        )
        assert submit_resp.status_code == 200

        # Verify AnswerEvaluation record in database
        stmt_eval = select(AnswerEvaluation).where(AnswerEvaluation.session_id == session_id)
        db_evals = in_memory_db.execute(stmt_eval).scalars().all()
        assert len(db_evals) == 1
        assert db_evals[0].score == 90

        # 4. Complete Session via API
        complete_resp = client.post(f"/api/v1/interview/sessions/{session_id}/complete")
        assert complete_resp.status_code == 200

        # Verify InterviewSession status in database
        in_memory_db.expire_all()
        db_session_final = in_memory_db.get(InterviewSession, session_id)
        assert db_session_final.status == "completed"

        # 5. List Sessions via API with DB persistence
        list_resp = client.get("/api/v1/interview/sessions")
        assert list_resp.status_code == 200
        listed_sessions = list_resp.json()
        assert len(listed_sessions) >= 1
        assert any(s["session_id"] == session_id for s in listed_sessions)

    finally:
        app.dependency_overrides.clear()




# ============================================================================
# 6. Optional Live Neon Database Integration Test
# ============================================================================


@pytest.mark.skipif(
    not os.getenv("NEON_DATABASE_URL"),
    reason="Requires live NEON_DATABASE_URL environment variable to run integration test.",
)
def test_live_neon_database_connectivity():
    """
    Optional live integration test against Neon PostgreSQL.
    To execute, export NEON_DATABASE_URL='postgresql+psycopg://USER:PASSWORD@HOST/DATABASE?sslmode=require'.
    """
    neon_url = os.environ["NEON_DATABASE_URL"]
    norm_url = normalize_database_url(neon_url)
    engine = create_engine(norm_url)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();")).scalar()
        assert "PostgreSQL" in result
    engine.dispose()
