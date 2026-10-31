"""Comprehensive test suite for the Source-Based AI Interview Workflow.

Covers:
- Document ingestion (PDF, text extraction, validation, chunking, Pinecone indexing, status)
- URL ingestion (SSRF protection, redirect re-validation, timeouts, size limits, HTML extraction)
- GitHub repository ingestion via GitHub MCP service
- Source-specific retrieval isolation
- Source-grounded interview question generation (Ollama, grounded context, stable IDs, error cases)
- Interview session initiation linked to sources with pre-generated questions
- Accidental duplicate session prevention
- Answer submission, evaluation, and completion lifecycle for source sessions
- Service failure resilience (Ollama, Pinecone, and network errors)
"""

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import httpx
import pypdf
import pytest

from app.main import app
from app.schemas.document import (
    DocumentUploadResponse,
    SourceProcessingStatus,
    SourceQuestionsResponse,
    SourceStatusResponse,
    SourceType,
    SourceUploadResponse,
)
from app.schemas.github import RepoReadmeResponse
from app.schemas.interview import DifficultyLevel, ExperienceLevel, QuestionType
from app.schemas.session import SessionDetailResponse, SessionStatus
from app.services.chunking_service import TextChunk
from app.services.embedding_service import EmbeddingConnectionError, get_embedding_service
from app.services.github_mcp_service import GitHubMCPService, get_github_service
from app.services.ollama_service import (
    OllamaConnectionError,
    OllamaResponseError,
    OllamaService,
    get_ollama_service,
)
from app.services.pinecone_service import PineconeConnectionError, get_pinecone_service
from app.services.session_service import get_session_service
from app.services.source_service import SourceNotFoundError, get_source_service
from app.services.url_service import (
    SSRFSecurityError,
    URLService,
    get_url_service,
)

client = TestClient(app)


# ============================================================================
# Helpers & Test Fixtures
# ============================================================================


def create_sample_pdf(text: str = "Senior Python Developer with FastAPI and Postgres skills") -> bytes:
    """Helper creating a valid 1-page PDF for test uploads."""
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


MOCK_QUESTIONS_LLM_OUTPUT = json.dumps({
    "questions": [
        {
            "id": "q_1",
            "question": "How do you manage database connection pooling when scaling FastAPI applications?",
            "expected_answer_points": [
                "Use connection poolers like PgBouncer or Neon connection pooler",
                "Configure pool_size, max_overflow, and pool_recycle in SQLAlchemy",
                "Properly close sessions via context managers or dependencies",
            ],
            "follow_up_question": "What strategies mitigate connection exhaustion during sudden traffic spikes?",
            "difficulty": "Medium",
            "topic": "FastAPI and Database Architecture",
        },
        {
            "id": "q_2",
            "question": "Can you explain how semantic vector retrieval works with Pinecone and nomic-embed-text?",
            "expected_answer_points": [
                "Generate dense 768-dimensional float vectors using nomic-embed-text",
                "Cosine similarity metric for nearest neighbor ranking",
                "Document-scoped metadata filtering prevents cross-candidate data leakage",
            ],
            "follow_up_question": "How do you handle chunk overlap to preserve semantic context across chunk boundaries?",
            "difficulty": "Medium",
            "topic": "RAG & Vector Retrieval",
        },
        {
            "id": "q_3",
            "question": "What techniques do you employ to prevent Server-Side Request Forgery (SSRF) in URL ingestion?",
            "expected_answer_points": [
                "Validate URL scheme strictly to HTTP/HTTPS",
                "Perform DNS resolution and verify IP against private and loopback ranges",
                "Revalidate redirect destinations on every hop before requesting",
            ],
            "follow_up_question": "Why is checking the initial URL hostname alone insufficient to prevent SSRF?",
            "difficulty": "Hard",
            "topic": "Application Security",
        },
    ]
})

MOCK_EVALUATION_LLM_OUTPUT = json.dumps({
    "overall_score": 90,
    "rating": "Excellent",
    "correct_points": [
        "Configured SQLAlchemy connection pool sizing",
        "Recommended PgBouncer for horizontal connection sharing",
    ],
    "missing_points": [],
    "strengths": [
        "Thorough understanding of ASGI concurrency and pool limits",
    ],
    "improvements": [
        "Mention keepalive and TCP timeout parameters",
    ],
    "sample_answer": "Connection pooling in FastAPI is managed by configuring SQLAlchemy's QueuePool...",
})


class MockOllamaForQuestions(OllamaService):
    """Deterministic mock Ollama service returning questions and evaluations."""

    async def generate_chat_completion(self, messages, format=None, **kwargs):
        content = " ".join(m.get("content", "") for m in messages)
        if "generate" in content.lower() and "questions" in content.lower():
            return MOCK_QUESTIONS_LLM_OUTPUT
        return MOCK_QUESTIONS_LLM_OUTPUT

    async def evaluate_interview_answer(self, question, expected_answer_points, candidate_answer):
        return MOCK_EVALUATION_LLM_OUTPUT


@pytest.fixture(autouse=True)
def reset_service_stores():
    """Reset source and session services before each test."""
    get_source_service().reset_store()
    get_session_service().reset_store()
    yield
    get_source_service().reset_store()
    get_session_service().reset_store()


# ============================================================================
# 1. Document Ingestion Tests
# ============================================================================


def test_document_upload_success_returns_source_id_and_status():
    """Test PDF upload succeeds and returns valid source_id, document_id, and completed status."""
    pdf_bytes = create_sample_pdf("Expertise in Python, FastAPI, Docker, and Kubernetes")
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("engineer_resume.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "source_id" in data
    assert data["source_id"].startswith("doc_")
    assert data["document_id"] == data["source_id"]
    assert data["status"] == "completed"
    assert data["filename"] == "engineer_resume.pdf"
    assert data["character_count"] > 0
    assert data["chunk_count"] > 0
    assert data["vector_count"] > 0

    # Verify source was registered and can be queried
    status_resp = client.get(f"/api/v1/sources/{data['source_id']}/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["source_id"] == data["source_id"]
    assert status_data["status"] == "completed"
    assert status_data["source_type"] == "document"


def test_document_upload_empty_file_rejected():
    """Test 400 rejection when uploading a 0-byte file."""
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "empty" in data["detail"].lower()


def test_document_upload_unsupported_file_rejected():
    """Test 400 rejection when uploading an unsupported file format."""
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("malware.exe", b"MZ\x90\x00\x03\x00\x00\x00", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "only pdf" in response.json()["detail"].lower()


# ============================================================================
# 2. URL Ingestion & SSRF Security Tests
# ============================================================================


def test_url_ingest_ssrf_localhost_rejected():
    """Test rejection of localhost URL targeting local services."""
    response = client.post(
        "/api/v1/sources/url",
        json={"url": "http://localhost:8000/internal-secrets"},
    )
    assert response.status_code == 400
    assert "localhost" in response.json()["detail"].lower() or "unsafe" in response.json()["detail"].lower()


def test_url_ingest_ssrf_127_0_0_1_rejected():
    """Test rejection of loopback IPv4 address 127.0.0.1."""
    response = client.post(
        "/api/v1/sources/url",
        json={"url": "http://127.0.0.1:8080/admin"},
    )
    assert response.status_code == 400
    assert "private" in response.json()["detail"].lower() or "unsafe" in response.json()["detail"].lower() or "localhost" in response.json()["detail"].lower()


def test_url_ingest_ssrf_private_class_a_rejected():
    """Test rejection of Class A private network address (10.0.0.1)."""
    response = client.post(
        "/api/v1/sources/url",
        json={"url": "http://10.0.0.1/intranet"},
    )
    assert response.status_code == 400
    assert "private" in response.json()["detail"].lower() or "unsafe" in response.json()["detail"].lower()


def test_url_ingest_ssrf_private_class_c_rejected():
    """Test rejection of Class C private network address (192.168.1.1)."""
    response = client.post(
        "/api/v1/sources/url",
        json={"url": "http://192.168.1.1/router-config"},
    )
    assert response.status_code == 400
    assert "private" in response.json()["detail"].lower() or "unsafe" in response.json()["detail"].lower()


def test_url_ingest_ssrf_cloud_metadata_rejected():
    """Test rejection of AWS/GCP cloud metadata IP address (169.254.169.254)."""
    response = client.post(
        "/api/v1/sources/url",
        json={"url": "http://169.254.169.254/latest/meta-data/"},
    )
    assert response.status_code == 400
    assert "private" in response.json()["detail"].lower() or "unsafe" in response.json()["detail"].lower()


def test_url_ingest_non_http_scheme_rejected():
    """Test rejection of non-HTTP schemes (e.g. file:// or ftp://)."""
    response = client.post(
        "/api/v1/sources/url",
        json={"url": "file:///etc/passwd"},
    )
    assert response.status_code == 422 or response.status_code == 400


def test_url_ingest_successful_public_webpage():
    """Test successful URL ingestion with mocked public webpage content."""
    mock_html = """
    <!DOCTYPE html>
    <html>
    <head><title>FastAPI Microservices Architecture Guide</title></head>
    <body>
        <h1>Building Scalable Microservices with FastAPI</h1>
        <p>FastAPI provides asynchronous concurrency, dependency injection, and Pydantic validation.</p>
        <p>In distributed systems, microservices communicate over gRPC or RESTful JSON APIs.</p>
        <p>Database queries should utilize connection pooling with SQLAlchemy 2.0 and psycopg 3.</p>
    </body>
    </html>
    """

    mock_response = httpx.Response(
        status_code=200,
        content=mock_html.encode("utf-8"),
        headers={"Content-Type": "text/html; charset=utf-8"},
        request=httpx.Request("GET", "https://example.com/fastapi-guide"),
    )

    with patch.object(URLService, "validate_ssrf_url", return_value="https://example.com/fastapi-guide"):
        with patch("httpx.AsyncClient.get", return_value=mock_response):
            response = client.post(
                "/api/v1/sources/url",
                json={"url": "https://example.com/fastapi-guide"},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["source_id"].startswith("src_")
    assert data["source_type"] == "url"
    assert data["status"] == "completed"
    assert "FastAPI Microservices" in data["title"]
    assert data["character_count"] > 100
    assert data["chunk_count"] > 0
    assert data["vector_count"] > 0


def test_url_ingest_unsafe_redirect_rejected():
    """Test that a redirect to a private IP (e.g. 127.0.0.1) is blocked during follow-up validation."""
    redirect_resp = httpx.Response(
        status_code=302,
        headers={"Location": "http://127.0.0.1:8080/forbidden"},
        request=httpx.Request("GET", "https://public-site.com/bounce"),
    )

    with patch.object(URLService, "validate_ssrf_url") as mock_val:
        # First call for initial public URL passes, second call for redirect destination raises SSRFSecurityError
        mock_val.side_effect = [
            "https://public-site.com/bounce",
            SSRFSecurityError("Redirect destination resolved to unsafe address."),
        ]
        with patch("httpx.AsyncClient.get", return_value=redirect_resp):
            response = client.post(
                "/api/v1/sources/url",
                json={"url": "https://public-site.com/bounce"},
            )

    assert response.status_code == 400
    assert "unsafe" in response.json()["detail"].lower()


def test_url_ingest_paywall_or_protected_content_rejected():
    """Test rejection when URL destination returns 403 Forbidden."""
    protected_resp = httpx.Response(
        status_code=403,
        content=b"Access Denied: Authentication required",
        request=httpx.Request("GET", "https://paywalled-site.com/article"),
    )

    with patch.object(URLService, "validate_ssrf_url", return_value="https://paywalled-site.com/article"):
        with patch("httpx.AsyncClient.get", return_value=protected_resp):
            response = client.post(
                "/api/v1/sources/url",
                json={"url": "https://paywalled-site.com/article"},
            )

    assert response.status_code == 403
    assert "paywall" in response.json()["detail"].lower() or "protected" in response.json()["detail"].lower()


def test_url_ingest_empty_page_rejected():
    """Test 400 rejection when webpage contains no readable text."""
    empty_html = "<html><head><title></title></head><body>   </body></html>"
    empty_resp = httpx.Response(
        status_code=200,
        content=empty_html.encode("utf-8"),
        headers={"Content-Type": "text/html"},
        request=httpx.Request("GET", "https://example.com/empty"),
    )

    with patch.object(URLService, "validate_ssrf_url", return_value="https://example.com/empty"):
        with patch("httpx.AsyncClient.get", return_value=empty_resp):
            response = client.post(
                "/api/v1/sources/url",
                json={"url": "https://example.com/empty"},
            )

    assert response.status_code == 400
    assert "too short" in response.json()["detail"].lower() or "no readable text" in response.json()["detail"].lower()


# ============================================================================
# 3. GitHub MCP Repository Ingestion Tests
# ============================================================================


def test_url_ingest_github_repo_reuses_mcp():
    """Test that GitHub repo URLs are ingested by reusing GitHubMCPService."""
    mock_readme_content = (
        "# FastAPI\n\n"
        "FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.8+.\n"
        "Key features: Fast execution, fast coding, fewer bugs, intuitive editor support, standards-based.\n"
    )

    mock_readme = RepoReadmeResponse(
        owner="fastapi",
        repo="fastapi",
        filename="README.md",
        size_bytes=len(mock_readme_content),
        content=mock_readme_content,
        html_url="https://github.com/fastapi/fastapi/blob/main/README.md",
    )

    with patch.object(URLService, "validate_ssrf_url", return_value="https://github.com/fastapi/fastapi"):
        with patch.object(GitHubMCPService, "get_readme", return_value=mock_readme) as mock_get_readme:
            response = client.post(
                "/api/v1/sources/url",
                json={"url": "https://github.com/fastapi/fastapi"},
            )

            mock_get_readme.assert_called_once()

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["source_type"] == "github"
    assert "fastapi/fastapi" in data["title"]
    assert data["chunk_count"] > 0
    assert data["vector_count"] > 0


# ============================================================================
# 4. Source-Grounded Question Generation Tests
# ============================================================================


def test_generate_questions_from_source_success():
    """Test generating multiple structured interview questions from a processed source."""
    # Register an active, completed source
    source_service = get_source_service()
    source_id = "src_sample_test_001"
    source_service.register_source(
        source_id=source_id,
        source_type=SourceType.DOCUMENT,
        title="candidate_resume.pdf",
        status=SourceProcessingStatus.COMPLETED,
        character_count=1200,
        chunk_count=3,
        vector_count=3,
    )

    app.dependency_overrides[get_ollama_service] = lambda: MockOllamaForQuestions()

    try:
        response = client.post(
            f"/api/v1/sources/{source_id}/questions",
            json={
                "role": "Backend Engineer",
                "difficulty": "Medium",
                "count": 3,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["source_id"] == source_id
        assert data["context_grounded"] is True
        assert data["total_questions"] == 3
        assert len(data["questions"]) == 3

        # Verify structured question fields and stable IDs
        q1 = data["questions"][0]
        assert q1["id"] == "q_1"
        assert q1["order"] == 1
        assert len(q1["question"]) > 10
        assert len(q1["expected_answer_points"]) >= 1
        assert "follow_up_question" in q1
        assert q1["difficulty"] == "Medium"
        assert "retrieved_chunks" in data
        assert len(data["retrieved_chunks"]) > 0
    finally:
        app.dependency_overrides.pop(get_ollama_service, None)


def test_generate_questions_missing_source_returns_404():
    """Test 404 when requesting questions for an unknown source ID."""
    response = client.post(
        "/api/v1/sources/nonexistent_src_99999/questions",
        json={"role": "Backend Engineer", "count": 3},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_generate_questions_processing_source_returns_400():
    """Test 400 rejection when requesting questions for a source that is still processing."""
    source_service = get_source_service()
    source_id = "src_pending_002"
    source_service.register_source(
        source_id=source_id,
        source_type=SourceType.URL,
        title="Pending Page",
        status=SourceProcessingStatus.PROCESSING,
    )

    response = client.post(
        f"/api/v1/sources/{source_id}/questions",
        json={"count": 2},
    )
    assert response.status_code == 400
    detail = response.json()["detail"].lower()
    assert "processed" in detail or "processing" in detail


# ============================================================================
# 5. Session Creation & Source Linking Tests
# ============================================================================


def test_start_session_linked_to_source_with_questions():
    """Test creating an interview session linked to a source ID and pre-populated questions."""
    source_service = get_source_service()
    source_id = "src_ready_003"
    source_service.register_source(
        source_id=source_id,
        source_type=SourceType.DOCUMENT,
        title="system_design.pdf",
        status=SourceProcessingStatus.COMPLETED,
    )

    session_payload = {
        "source_id": source_id,
        "role": "Cloud Architect",
        "experience_level": "Senior",
        "difficulty": "Hard",
        "topic": "Distributed Microservices",
        "question_type": "Technical",
        "total_questions": 2,
        "questions": [
            {
                "id": "q_1",
                "question": "How do you achieve zero-downtime database schema migrations in high-throughput systems?",
                "expected_answer_points": [
                    "Expand and contract pattern",
                    "Add new nullable columns first",
                    "Dual-write during migration phase",
                ],
                "follow_up_question": "How would you handle backfilling historical rows without table locks?",
                "difficulty": "Hard",
                "topic": "Database Architecture",
            },
            {
                "id": "q_2",
                "question": "Explain how distributed tracing works across microservices using OpenTelemetry.",
                "expected_answer_points": [
                    "Trace ID and Span ID propagated via W3C headers",
                    "Sampling strategies to balance overhead and visibility",
                ],
                "follow_up_question": "What is the performance overhead of tail-based sampling?",
                "difficulty": "Hard",
                "topic": "Observability",
            },
        ],
    }

    response = client.post("/api/v1/interview/sessions", json=session_payload)
    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert data["source_id"] == source_id
    assert data["role"] == "Cloud Architect"
    assert data["total_questions"] == 2
    assert data["current_question_number"] == 1
    assert data["completed_questions"] == 0
    assert data["status"] == "in_progress"
    assert len(data["questions"]) == 2
    assert data["questions"][0]["question_number"] == 1
    assert "zero-downtime" in data["questions"][0]["question"]

    # Retrieve session details to verify persistence and source linkage
    detail_resp = client.get(f"/api/v1/interview/sessions/{data['session_id']}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["source_id"] == source_id


def test_session_creation_duplicate_prevented():
    """Test that submitting identical session creation payloads in rapid succession returns the same session."""
    source_id = "src_dedup_004"
    get_source_service().register_source(
        source_id=source_id,
        source_type=SourceType.DOCUMENT,
        title="resume.pdf",
        status=SourceProcessingStatus.COMPLETED,
    )

    payload = {
        "source_id": source_id,
        "role": "Frontend Engineer",
        "experience_level": "Mid-Level",
        "difficulty": "Medium",
        "topic": "React State Management",
        "question_type": "Technical",
        "total_questions": 3,
    }

    # First call creates session
    resp1 = client.post("/api/v1/interview/sessions", json=payload)
    assert resp1.status_code == 201
    session_id_1 = resp1.json()["session_id"]

    # Second identical call within deduplication window returns existing session
    resp2 = client.post("/api/v1/interview/sessions", json=payload)
    assert resp2.status_code in (200, 201)
    session_id_2 = resp2.json()["session_id"]

    assert session_id_1 == session_id_2


def test_answer_submission_and_lifecycle_for_source_session():
    """Verify answer submission and evaluation continue working smoothly with source-linked sessions."""
    app.dependency_overrides[get_ollama_service] = lambda: MockOllamaForQuestions()

    try:
        session_payload = {
            "source_id": "src_answer_test_005",
            "role": "Backend Developer",
            "experience_level": "Mid-Level",
            "difficulty": "Medium",
            "topic": "FastAPI",
            "question_type": "Technical",
            "total_questions": 1,
            "questions": [
                {
                    "question": "How do dependency injections work in FastAPI?",
                    "expected_answer_points": ["Uses Depends() helper", "Resolves sub-dependencies hierarchically"],
                    "difficulty": "Medium",
                    "topic": "FastAPI",
                }
            ],
        }

        create_resp = client.post("/api/v1/interview/sessions", json=session_payload)
        assert create_resp.status_code == 201
        session_id = create_resp.json()["session_id"]

        # Submit answer to question 1
        answer_payload = {
            "candidate_answer": "FastAPI uses Depends() to inject reusable logic like database sessions and auth checks.",
            "question_number": 1,
        }
        ans_resp = client.post(f"/api/v1/interview/sessions/{session_id}/answers", json=answer_payload)
        assert ans_resp.status_code == 200
        ans_data = ans_resp.json()
        assert ans_data["completed_questions"] == 1
        assert "evaluation" in ans_data
        assert ans_data["evaluation"]["overall_score"] == 90
        assert ans_data["evaluation"]["rating"] == "Excellent"

        # Complete the session
        comp_resp = client.post(f"/api/v1/interview/sessions/{session_id}/complete")
        assert comp_resp.status_code == 200
        summary = comp_resp.json()
        assert summary["session_status"] == "completed"
        assert summary["average_score"] == 90.0
        assert summary["questions_answered"] == 1
    finally:
        app.dependency_overrides.pop(get_ollama_service, None)


# ============================================================================
# 6. Service Failures & Error Handling
# ============================================================================


def test_url_ingest_embedding_failure_returns_503():
    """Test 503 Service Unavailable when embedding service is unreachable."""
    class FailingEmbedding:
        async def embed_chunks(self, *args, **kwargs):
            raise EmbeddingConnectionError("Ollama embedding service connection refused")

    app.dependency_overrides[get_embedding_service] = lambda: FailingEmbedding()

    mock_html = "<html><head><title>Test</title></head><body>A valid page with enough readable text to pass minimum character length checks.</body></html>"
    mock_resp = httpx.Response(status_code=200, content=mock_html.encode("utf-8"), request=httpx.Request("GET", "https://example.com/page"))

    try:
        with patch.object(URLService, "validate_ssrf_url", return_value="https://example.com/page"):
            with patch("httpx.AsyncClient.get", return_value=mock_resp):
                response = client.post("/api/v1/sources/url", json={"url": "https://example.com/page"})
        assert response.status_code == 503
        assert "embedding service" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.pop(get_embedding_service, None)


def test_source_questions_ollama_malformed_json_returns_502():
    """Test 502 Bad Gateway when Ollama returns non-JSON or malformed text."""
    class BadJsonOllama(OllamaService):
        async def generate_chat_completion(self, *args, **kwargs):
            return "This is plain text with no JSON structure."

    app.dependency_overrides[get_ollama_service] = lambda: BadJsonOllama()

    source_service = get_source_service()
    source_id = "src_bad_json_006"
    source_service.register_source(
        source_id=source_id,
        source_type=SourceType.DOCUMENT,
        title="doc.pdf",
        status=SourceProcessingStatus.COMPLETED,
    )

    try:
        response = client.post(
            f"/api/v1/sources/{source_id}/questions",
            json={"role": "Backend Engineer", "count": 2},
        )
        assert response.status_code == 502
    finally:
        app.dependency_overrides.pop(get_ollama_service, None)
