"""Regression tests covering backend audit fixes, error mappings, security isolation, and schema compatibility."""

from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
import httpx
import pytest
from pydantic import BaseModel, ValidationError
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.exceptions import handle_github_errors
from app.core.security import hash_token
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.models.auth import RefreshToken, User
from app.schemas.document import SourceProcessingStatus, SourceQuestionsRequest, SourceType
from app.schemas.github import RepoQuestionRequest
from app.schemas.interview import DifficultyLevel, ExperienceLevel, InterviewQuestionRequest, InterviewQuestionResponse, QuestionType
from app.schemas.session import CreateSessionRequest
from app.services.document_service import DocumentService
from app.services.embedding_service import EmbeddingConnectionError, EmbeddingService
from app.services.github_mcp_service import (
    GitHubConnectionError,
    GitHubMCPService,
    GitHubRateLimitError,
    GitHubRepoNotFoundError,
    GitHubTimeoutError,
)
from app.services.ollama_service import OllamaResponseError, OllamaService
from app.services.pdf_service import PDFNoTextError
from app.services.question_generation_service import QuestionGenerationService, QuestionOutputFormatError
from app.services.rag_question_service import RAGQuestionService
from app.services.repo_question_service import RepoQuestionService
from app.services.session_service import SessionService, SessionStateError
from app.services.url_service import EmptyContentError, URLService, get_url_service


@pytest.fixture
def isolated_auth_db():
    """Isolated SQLite database for real-auth refresh cookie tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def _override_db():
        db = TestingSessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_db
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        app.dependency_overrides.pop(get_db, None)
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.mark.real_auth
def test_refresh_401_clears_auth_cookies(isolated_auth_db):
    """Invalid refresh token returns 401 AND includes Set-Cookie deletion headers."""
    client = TestClient(app)
    client.cookies.set("refresh_token", "invalid-token-value")
    client.cookies.set("csrf_token", "csrf-value")

    resp = client.post("/api/v1/auth/refresh", headers={"X-CSRF-Token": "csrf-value"})
    assert resp.status_code == 401
    set_cookies = "\n".join(resp.headers.get_list("set-cookie"))
    assert "refresh_token=" in set_cookies
    assert "csrf_token=" in set_cookies


@pytest.mark.real_auth
def test_refresh_403_inactive_user_clears_auth_cookies(isolated_auth_db):
    """Inactive user attempting token refresh receives 403 AND cleared auth cookies."""
    user = User(
        id="user-inactive-1",
        name="Inactive",
        email="inactive@example.com",
        password_hash=None,
        is_active=False,
        email_verified=True,
    )
    isolated_auth_db.add(user)
    raw_refresh = "valid-shape-but-inactive-user-token"
    isolated_auth_db.add(
        RefreshToken(
            user_id=user.id,
            token_hash=hash_token(raw_refresh),
            family_id="fam-inactive-1",
            expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        )
    )
    isolated_auth_db.commit()

    client = TestClient(app)
    client.cookies.set("refresh_token", raw_refresh)
    client.cookies.set("csrf_token", "csrf-value")

    resp = client.post("/api/v1/auth/refresh", headers={"X-CSRF-Token": "csrf-value"})
    assert resp.status_code == 403
    set_cookies = "\n".join(resp.headers.get_list("set-cookie"))
    assert "refresh_token=" in set_cookies
    assert "csrf_token=" in set_cookies


@pytest.mark.parametrize(
    ("exc", "expected_status"),
    [
        (GitHubRepoNotFoundError("Repository not found"), 404),
        (GitHubConnectionError("Cannot connect to GitHub MCP"), 503),
        (GitHubRateLimitError("Rate limited"), 429),
        (GitHubTimeoutError("MCP timed out"), 504),
    ],
)
def test_github_url_ingestion_maps_errors_instead_of_500(exc, expected_status):
    """GitHub exceptions raised during /sources/url ingestion map cleanly to 4xx/5xx instead of 500."""
    mock_url_service = MagicMock()
    mock_url_service.ingest_url = AsyncMock(side_effect=exc)
    app.dependency_overrides[get_url_service] = lambda: mock_url_service

    client = TestClient(app)
    resp = client.post("/api/v1/sources/url", json={"url": "https://github.com/octocat/Hello-World"})
    assert resp.status_code == expected_status
    assert "detail" in resp.json()


def test_handle_github_errors_maps_pydantic_validation_error_to_502():
    """LLM response Pydantic ValidationError inside handle_github_errors yields 502 Bad Gateway."""
    class _MiniSchema(BaseModel):
        value: int

    with pytest.raises(Exception) as exc_info:
        with handle_github_errors():
            _MiniSchema(value="not-an-int")  # type: ignore[arg-type]

    assert getattr(exc_info.value, "status_code", None) == 502


def test_schema_contract_long_source_title_and_question_limit():
    """Source-derived long titles (>100 chars) are safely truncated, and >20 questions return 422."""
    long_title = "Comprehensive Distributed Microservices Architecture & Event Sourcing Guide " * 3
    assert len(long_title) > 100

    src_req = SourceQuestionsRequest(
        source_id="src-123",
        topic=long_title,
        difficulty=DifficultyLevel.MEDIUM,
        question_type=QuestionType.TECHNICAL,
    )
    assert src_req.topic is not None and len(src_req.topic) <= 100

    short_src_req = SourceQuestionsRequest(source_id="src-123", topic="A")
    assert short_src_req.topic is None

    sess_req = CreateSessionRequest(
        role="Backend Engineer",
        experience_level=ExperienceLevel.MID_LEVEL,
        difficulty=DifficultyLevel.MEDIUM,
        topic=long_title,
        question_type=QuestionType.TECHNICAL,
        total_questions=3,
        source_id="src-123",
    )
    assert len(sess_req.topic) <= 100

    with pytest.raises(ValidationError):
        CreateSessionRequest(
            role="Backend Engineer",
            experience_level=ExperienceLevel.MID_LEVEL,
            difficulty=DifficultyLevel.MEDIUM,
            topic="FastAPI",
            question_type=QuestionType.TECHNICAL,
            total_questions=5,
            questions=[{"question": f"Question {i}"} for i in range(21)],
        )


@pytest.mark.anyio
async def test_rag_source_questions_low_score_and_malformed_llm_normalization():
    """Source question generation uses min_score=0.0, separates cache keys, and normalizes imperfect LLM fields."""
    fake_source = SimpleNamespace(
        source_id="src-low-score",
        status=SourceProcessingStatus.COMPLETED,
        title="System Design Notes",
        error_message=None,
    )
    mock_source_service = MagicMock()
    mock_source_service.get_source.return_value = fake_source

    low_score_chunk = SimpleNamespace(
        chunk_id="src-low-score_chunk_0",
        document_id="src-low-score",
        chunk_index=0,
        text="Consistent hashing distributes partition keys across ring nodes with minimal rebalancing.",
        safe_excerpt="Consistent hashing distributes partition keys across ring nodes with minimal rebalancing.",
        score=0.42,
        filename="notes.pdf",
    )
    mock_retrieval = MagicMock()
    mock_retrieval.retrieve_chunks = AsyncMock(return_value=[low_score_chunk])

    mock_ollama = MagicMock()
    mock_ollama.generate_chat_completion = AsyncMock(
        return_value=json.dumps({
            "questions": [
                {
                    "question": "How does consistent hashing minimize key movement during node scaling?",
                    "expected_answer_points": "Hash ring partitioning reduces key remapping to O(K/N).",
                    "follow_up_question": None,
                    "difficulty": "Expert",
                    "topic": "Consistent Hashing",
                }
            ]
        })
    )

    service = RAGQuestionService(
        retrieval_service=mock_retrieval,
        ollama_service=mock_ollama,
        source_service=mock_source_service,
    )
    service.clear_cache()

    req_medium = SourceQuestionsRequest(
        source_id="src-low-score",
        role="Backend Engineer",
        difficulty=DifficultyLevel.MEDIUM,
        question_type=QuestionType.TECHNICAL,
        count=1,
    )
    res_medium = await service.generate_source_questions(req_medium, user_id="u1")

    assert res_medium.total_questions == 1
    assert res_medium.questions[0].difficulty == DifficultyLevel.MEDIUM
    assert res_medium.questions[0].expected_answer_points == ["Hash ring partitioning reduces key remapping to O(K/N)."]
    assert res_medium.questions[0].follow_up_question is None
    # Assert min_score=0.0 was passed so score=0.42 wasn't filtered out
    assert mock_retrieval.retrieve_chunks.call_args.kwargs["min_score"] == 0.0

    # Changing difficulty must NOT return the Medium cached result
    req_hard = SourceQuestionsRequest(
        source_id="src-low-score",
        role="Backend Engineer",
        difficulty=DifficultyLevel.HARD,
        question_type=QuestionType.TECHNICAL,
        count=1,
    )
    res_hard = await service.generate_source_questions(req_hard, user_id="u1")
    assert res_hard.questions[0].difficulty == DifficultyLevel.HARD
    assert mock_ollama.generate_chat_completion.await_count == 2


@pytest.mark.anyio
async def test_rag_source_questions_raises_502_error_on_empty_question_output():
    """Malformed/too-short LLM questions raise OllamaResponseError (mapped to 502), not 500 ValidationError."""
    fake_source = SimpleNamespace(
        source_id="src-bad-llm",
        status=SourceProcessingStatus.COMPLETED,
        title="Backend Docs",
        error_message=None,
    )
    mock_source_service = MagicMock()
    mock_source_service.get_source.return_value = fake_source

    mock_retrieval = MagicMock()
    mock_retrieval.retrieve_chunks = AsyncMock(
        return_value=[
            SimpleNamespace(
                chunk_id="c1",
                document_id="src-bad-llm",
                chunk_index=0,
                text="FastAPI dependency injection manages per-request database sessions.",
                score=0.81,
                filename="docs.pdf",
            )
        ]
    )
    mock_ollama = MagicMock()
    mock_ollama.generate_chat_completion = AsyncMock(
        return_value=json.dumps({"questions": [{"question": "Hi?", "expected_answer_points": []}]})
    )

    service = RAGQuestionService(
        retrieval_service=mock_retrieval,
        ollama_service=mock_ollama,
        source_service=mock_source_service,
    )
    service.clear_cache()
    with pytest.raises(OllamaResponseError):
        await service.generate_source_questions(
            SourceQuestionsRequest(source_id="src-bad-llm", count=1),
            user_id="u1",
        )


@pytest.mark.anyio
async def test_repo_question_service_normalizes_imperfect_llm_output():
    """RepoQuestionService safely normalizes unrecognized difficulty, null follow-up, and string answer points."""
    mock_github = MagicMock()
    mock_github.fetch_repository_context = AsyncMock(
        return_value={
            "tree_summary": "app/main.py",
            "readme_content": "FastAPI platform.",
            "target_files": [{"path": "app/main.py", "content": "app = FastAPI()"}],
            "citations": ["app/main.py"],
        }
    )
    mock_ollama = MagicMock()
    mock_ollama.generate_chat_completion = AsyncMock(
        return_value=json.dumps({
            "question": "How is middleware registered in FastAPI?",
            "expected_answer_points": "Using app.add_middleware cleanly",
            "follow_up_question": None,
            "difficulty": "Legendary",
            "topic": "",
        })
    )

    svc = RepoQuestionService(github_service=mock_github, ollama_service=mock_ollama)
    resp = await svc.generate_repo_question(
        RepoQuestionRequest(repo_url_or_name="octocat/Hello-World", difficulty=DifficultyLevel.MEDIUM)
    )
    assert resp.difficulty == DifficultyLevel.MEDIUM
    assert resp.expected_answer_points == ["Using app.add_middleware cleanly"]
    assert len(resp.follow_up_question) >= 5
    assert resp.topic == "System Architecture & Code Quality"


@pytest.mark.anyio
async def test_question_generation_service_handles_fences_and_normalization():
    """QuestionGenerationService parses fenced JSON and normalizes unrecognized difficulty / null follow-up."""
    mock_ollama = MagicMock()
    mock_ollama.generate_chat_completion = AsyncMock(
        return_value='```json\n{"question": "Explain Python asyncio event loop scheduling.", "expected_answer_points": "Cooperative multitasking with coroutines", "follow_up_question": null, "difficulty": "Ultra", "topic": ""}\n```'
    )
    svc = QuestionGenerationService(ollama_service=mock_ollama)
    res = await svc.generate_question(
        InterviewQuestionRequest(
            role="Python Engineer",
            experience_level=ExperienceLevel.SENIOR,
            difficulty=DifficultyLevel.HARD,
            topic="AsyncIO",
            question_type=QuestionType.TECHNICAL,
        )
    )
    assert res.difficulty == DifficultyLevel.HARD
    assert res.topic == "AsyncIO"
    assert res.expected_answer_points == ["Cooperative multitasking with coroutines"]
    assert len(res.follow_up_question) >= 5

    mock_ollama.generate_chat_completion = AsyncMock(return_value="not valid json")
    with pytest.raises(QuestionOutputFormatError):
        await svc.generate_question(
            InterviewQuestionRequest(
                role="Python Engineer",
                experience_level=ExperienceLevel.SENIOR,
                difficulty=DifficultyLevel.HARD,
                topic="AsyncIO",
                question_type=QuestionType.TECHNICAL,
            )
        )


@pytest.mark.anyio
async def test_document_and_url_services_clean_up_pinecone_and_reject_empty_chunks():
    """Failed ingestion after Pinecone upsert triggers best-effort delete_document, and 0 chunks fail cleanly."""
    mock_pdf = MagicMock()
    mock_pdf.extract_text_and_metadata = AsyncMock(
        return_value=("resume.pdf", "Sample extracted PDF text", 1, 25, "Sample")
    )
    mock_chunking = MagicMock()
    mock_chunking.create_chunks.return_value = [SimpleNamespace(text="chunk 0", chunk_index=0)]
    mock_embedding = MagicMock()
    mock_embedding.embed_chunks = AsyncMock(return_value=[[0.1] * 768])
    mock_pinecone = MagicMock()
    mock_pinecone.upsert_document_chunks = AsyncMock(return_value=1)
    mock_pinecone.delete_document = AsyncMock(return_value=True)
    mock_source = MagicMock()
    mock_source.update_source.side_effect = RuntimeError("Simulated DB commit failure")

    doc_svc = DocumentService(
        pdf_service=mock_pdf,
        chunking_service=mock_chunking,
        embedding_service=mock_embedding,
        pinecone_service=mock_pinecone,
        source_service=mock_source,
    )
    fake_file = SimpleNamespace(filename="resume.pdf")
    with pytest.raises(RuntimeError):
        await doc_svc.ingest_pdf(fake_file, user_id="user-1")

    mock_pinecone.delete_document.assert_awaited_once()
    mock_source.mark_source_failed.assert_called_once()

    # Empty chunks guard on DocumentService
    mock_chunking.create_chunks.return_value = []
    with pytest.raises(PDFNoTextError):
        await doc_svc.ingest_pdf(fake_file, user_id="user-1")

    # URLService cleanup + empty chunks guard
    url_svc = URLService(
        chunking_service=MagicMock(),
        embedding_service=mock_embedding,
        pinecone_service=MagicMock(),
        source_service=MagicMock(),
        github_service=MagicMock(),
    )
    url_svc.validate_ssrf_url = MagicMock(return_value="https://example.com")
    url_svc._fetch_webpage_content = AsyncMock(
        return_value=(
            "https://example.com",
            "Web Page Title",
            "Sample extracted web content with enough length to exceed fifty characters cleanly.",
        )
    )
    url_svc.chunking_service.create_chunks.return_value = [SimpleNamespace(text="chunk 0", chunk_index=0)]
    url_svc.pinecone_service.upsert_document_chunks = AsyncMock(return_value=1)
    url_svc.pinecone_service.delete_document = AsyncMock(return_value=True)
    url_svc.source_service.update_source.side_effect = RuntimeError("Simulated DB write failure")

    with pytest.raises(RuntimeError):
        await url_svc.ingest_url("https://example.com", user_id="user-1")
    url_svc.pinecone_service.delete_document.assert_awaited_once()

    url_svc.chunking_service.create_chunks.return_value = []
    with pytest.raises(EmptyContentError):
        await url_svc.ingest_url("https://example.com", user_id="user-1")


def test_session_pagination_bounds_and_concurrent_integrity_error():
    """Negative/excessive pagination params return 422, and IntegrityError maps cleanly to SessionStateError (400)."""
    client = TestClient(app)
    assert client.get("/api/v1/interview/sessions?limit=-1").status_code == 422
    assert client.get("/api/v1/interview/sessions?limit=500").status_code == 422
    assert client.get("/api/v1/interview/sessions?skip=-1").status_code == 422

    fake_db = MagicMock()
    mock_repo = MagicMock()
    mock_session_obj = SimpleNamespace(
        session_id="sess-race-1",
        role="Backend",
        experience_level=ExperienceLevel.SENIOR,
        difficulty=DifficultyLevel.MEDIUM,
        topic="Databases",
        question_type=QuestionType.TECHNICAL,
        total_questions=3,
        current_question_number=0,
        completed_questions=0,
        status="in_progress",
        source_id=None,
        created_at="2026-11-10T00:00:00Z",
        updated_at="2026-11-10T00:00:00Z",
        questions=[],
    )
    svc = SessionService(repository=mock_repo)
    svc.get_session = MagicMock(return_value=mock_session_obj)
    mock_repo.add_question.side_effect = IntegrityError("stmt", "params", Exception("unique violation"))

    q_data = InterviewQuestionResponse(
        question="Explain MVCC in PostgreSQL.",
        expected_answer_points=["Snapshot isolation", "Tuple versioning"],
        follow_up_question="What causes vacuum bloat?",
        difficulty=DifficultyLevel.MEDIUM,
        topic="Databases",
    )
    with pytest.raises(SessionStateError):
        svc.add_question_to_session("sess-race-1", q_data, db=fake_db, user_id="user-1")
    fake_db.rollback.assert_called_once()


def test_github_mcp_env_allowlist_and_token_sanitization(monkeypatch):
    """MCP subprocess env never receives unrelated DB/JWT/Pinecone secrets, and _sanitize masks PAT patterns."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://secret-user:secret-pass@db.example/app")
    monkeypatch.setenv("JWT_SECRET_KEY", "super-secret-jwt-key")
    monkeypatch.setenv("PINECONE_API_KEY", "pcsk-super-secret-pinecone-key")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    svc = GitHubMCPService(token="ghp_testtoken1234567890")
    params = svc._resolve_server_params()
    assert "DATABASE_URL" not in params.env
    assert "JWT_SECRET_KEY" not in params.env
    assert "PINECONE_API_KEY" not in params.env
    assert params.env["GITHUB_PERSONAL_ACCESS_TOKEN"] == "ghp_testtoken1234567890"
    assert params.env["PATH"] == "/usr/bin:/bin"

    leaked = "Error with ghp_testtoken1234567890 and github_pat_abcdef1234567890"
    sanitized = svc._sanitize(leaked)
    assert "ghp_testtoken1234567890" not in sanitized
    assert "github_pat_abcdef1234567890" not in sanitized


@pytest.mark.anyio
async def test_ollama_and_embedding_transient_retry_recovers():
    """OllamaService and EmbeddingService retry once on transient ConnectError and recover on 200 OK."""
    ollama_ok = MagicMock()
    ollama_ok.status_code = 200
    ollama_ok.raise_for_status = MagicMock()
    ollama_ok.json.return_value = {"message": {"content": "Recovered response"}}

    ollama_svc = OllamaService(base_url="http://localhost:11434", max_retries=1)
    with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=[httpx.ConnectError("transient"), ollama_ok])):
        out = await ollama_svc.generate_chat_completion([{"role": "user", "content": "hi"}])
        assert out == "Recovered response"

    embed_ok = MagicMock()
    embed_ok.status_code = 200
    embed_ok.is_error = False
    embed_ok.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

    embed_svc = EmbeddingService(base_url="http://localhost:11434", max_retries=1)
    with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=[httpx.ConnectError("transient"), embed_ok])):
        vectors = await embed_svc.embed_chunks(["hello world"])
        assert vectors == [[0.1, 0.2, 0.3]]

    # Persistent ConnectError exhausts retries and raises EmbeddingConnectionError
    with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=httpx.ConnectError("down"))):
        with pytest.raises(EmbeddingConnectionError):
            await embed_svc.embed_chunks(["hello world"])
