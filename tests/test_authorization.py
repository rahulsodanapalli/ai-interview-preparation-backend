"""Authorization, user isolation, IDOR prevention, and Pinecone tenant filtering tests."""

import io
import json
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pypdf
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.models.interview import InterviewSession
from app.services.ollama_service import OllamaService, get_ollama_service
from app.services.pinecone_service import PineconeService
from app.services.chunking_service import TextChunk

pytestmark = pytest.mark.real_auth


def _create_sample_pdf(text: str) -> bytes:
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
def authz_db():
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


def _create_user_and_headers(client: TestClient, email: str, name: str) -> tuple[str, dict]:
    resp = client.post(
        "/api/v1/auth/register",
        json={"name": name, "email": email, "password": "UserPassword1!"},
    )
    assert resp.status_code == 201
    data = resp.json()
    return data["user"]["id"], {"Authorization": f"Bearer {data['access_token']}"}


# ============================================================================
# 1. Unauthenticated Requests Must Return 401
# ============================================================================


@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("GET", "/api/v1/auth/me", None),
        ("GET", "/api/v1/interview/sessions", None),
        ("POST", "/api/v1/interview/sessions", {"role": "Backend", "experience_level": "Junior", "difficulty": "Easy", "topic": "API", "question_type": "Technical", "total_questions": 3}),
        ("GET", "/api/v1/interview/sessions/some-id", None),
        ("POST", "/api/v1/interview/sessions/some-id/next-question", None),
        ("POST", "/api/v1/interview/sessions/some-id/answers", {"candidate_answer": "Answer"}),
        ("POST", "/api/v1/interview/sessions/some-id/complete", None),
        ("POST", "/api/v1/interview/question", {"role": "Backend", "experience_level": "Junior", "difficulty": "Easy", "topic": "Python", "question_type": "Technical"}),
        ("POST", "/api/v1/interview/document-question", {"document_id": "doc_1"}),
        ("POST", "/api/v1/interview/source-questions", {"source_id": "src_1"}),
        ("POST", "/api/v1/interview/evaluate", {"question": "Q", "expected_answer_points": ["A"], "candidate_answer": "Ans"}),
        ("GET", "/api/v1/sources", None),
        ("GET", "/api/v1/sources/src_1/status", None),
        ("POST", "/api/v1/sources/url", {"url": "https://example.com"}),
        ("POST", "/api/v1/sources/src_1/questions", {}),
        ("POST", "/api/v1/sources/sessions", {"role": "Backend", "experience_level": "Junior", "difficulty": "Easy", "topic": "API", "question_type": "Technical", "total_questions": 3}),
        ("GET", "/api/v1/documents/doc_1/status", None),
        ("POST", "/api/v1/documents/doc_1/question", {}),
        ("POST", "/api/v1/github/validate", {"repo_url_or_name": "fastapi/fastapi"}),
        ("GET", "/api/v1/github/readme?repo=fastapi/fastapi", None),
        ("GET", "/api/v1/github/files?repo=fastapi/fastapi", None),
        ("POST", "/api/v1/github/file-content", {"repo_url_or_name": "fastapi/fastapi", "file_path": "README.md"}),
        ("POST", "/api/v1/github/question", {"repo_url_or_name": "fastapi/fastapi", "role": "Backend", "difficulty": "Easy", "experience_level": "Junior"}),
    ],
)
def test_unauthenticated_requests_rejected_401(authz_db, method, path, payload):
    """All protected endpoints reject unauthenticated requests with 401 Unauthorized."""
    client = TestClient(app)
    resp = client.request(method, path, json=payload)
    assert resp.status_code == 401


def test_public_health_endpoints_remain_accessible(authz_db):
    """Health endpoints are explicitly public and return 200 without auth."""
    client = TestClient(app)
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200


# ============================================================================
# 2. Cross-User Interview Session Ownership & IDOR Prevention
# ============================================================================


def test_user_a_cannot_access_user_b_interview_session(authz_db):
    """User A receives 403 when reading or mutating User B's interview session, and list_sessions is isolated."""
    client = TestClient(app)
    user_a_id, headers_a = _create_user_and_headers(client, "usera@example.com", "User A")
    user_b_id, headers_b = _create_user_and_headers(client, "userb@example.com", "User B")

    # Even if User B sends a forged user_id in request body, backend derives ownership from verified JWT
    create_b = client.post(
        "/api/v1/interview/sessions",
        headers=headers_b,
        json={
            "user_id": user_a_id,
            "role": "Backend Engineer",
            "experience_level": "Mid-Level",
            "difficulty": "Medium",
            "topic": "PostgreSQL",
            "question_type": "Technical",
            "total_questions": 2,
            "questions": [
                {
                    "question": "How does MVCC work in PostgreSQL?",
                    "expected_answer_points": ["Snapshot isolation", "Tuple visibility"],
                }
            ],
        },
    )
    assert create_b.status_code == 201
    b_session_id = create_b.json()["session_id"]

    # Confirm DB owner is User B, never forged User A
    row = authz_db.execute(select(InterviewSession).where(InterviewSession.id == b_session_id)).scalars().first()
    assert row is not None and row.user_id == user_b_id

    # User A's session list does NOT include User B's session
    list_a = client.get("/api/v1/interview/sessions", headers=headers_a)
    assert list_a.status_code == 200
    assert all(s["session_id"] != b_session_id for s in list_a.json())

    # User A cannot get, advance, answer, or complete User B's session
    assert client.get(f"/api/v1/interview/sessions/{b_session_id}", headers=headers_a).status_code == 403
    assert client.post(f"/api/v1/interview/sessions/{b_session_id}/next-question", headers=headers_a).status_code == 403
    assert (
        client.post(
            f"/api/v1/interview/sessions/{b_session_id}/answers",
            headers=headers_a,
            json={"candidate_answer": "Trying IDOR"},
        ).status_code
        == 403
    )
    assert client.post(f"/api/v1/interview/sessions/{b_session_id}/complete", headers=headers_a).status_code == 403


# ============================================================================
# 3. Cross-User Document, Source, Question & Pinecone Chunk Isolation
# ============================================================================


def test_user_a_cannot_access_user_b_document_or_pinecone_chunks(authz_db):
    """User A cannot read User B's document status, generate questions from it, or retrieve User B's Pinecone chunks."""
    client = TestClient(app)
    _user_a_id, headers_a = _create_user_and_headers(client, "doca@example.com", "Doc User A")
    _user_b_id, headers_b = _create_user_and_headers(client, "docb@example.com", "Doc User B")

    class MockOllama(OllamaService):
        async def generate_chat_completion(self, messages, format=None):
            return json.dumps({
                "question": "Explain your distributed caching architecture.",
                "expected_answer_points": ["Cache invalidation", "TTL"],
                "follow_up_question": "How do you prevent cache stampedes?",
                "difficulty": "Medium",
                "topic": "Caching",
            })

    app.dependency_overrides[get_ollama_service] = lambda: MockOllama()
    try:
        pdf_bytes = _create_sample_pdf("User B confidential architecture resume with Redis and Kafka.")
        upload_b = client.post(
            "/api/v1/documents/upload",
            headers=headers_b,
            files={"file": ("b_secret_resume.pdf", pdf_bytes, "application/pdf")},
        )
        assert upload_b.status_code == 200
        b_doc_id = upload_b.json()["document_id"]

        # User B sees their source in /sources
        sources_b = client.get("/api/v1/sources", headers=headers_b)
        assert sources_b.status_code == 200
        assert any(s["source_id"] == b_doc_id for s in sources_b.json())

        # User A does NOT see User B's source in /sources
        sources_a = client.get("/api/v1/sources", headers=headers_a)
        assert sources_a.status_code == 200
        assert all(s["source_id"] != b_doc_id for s in sources_a.json())

        # User A gets 403 on status, source questions, document questions, and linking a session to User B's source
        assert client.get(f"/api/v1/documents/{b_doc_id}/status", headers=headers_a).status_code == 403
        assert client.get(f"/api/v1/sources/{b_doc_id}/status", headers=headers_a).status_code == 403
        assert client.post(f"/api/v1/sources/{b_doc_id}/questions", headers=headers_a, json={}).status_code == 403
        assert client.post(f"/api/v1/documents/{b_doc_id}/questions", headers=headers_a, json={}).status_code == 403
        assert (
            client.post(
                "/api/v1/interview/source-questions",
                headers=headers_a,
                json={"source_id": b_doc_id},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/v1/interview/document-question",
                headers=headers_a,
                json={"document_id": b_doc_id},
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/v1/sources/sessions",
                headers=headers_a,
                json={
                    "source_id": b_doc_id,
                    "role": "Backend",
                    "experience_level": "Mid-Level",
                    "difficulty": "Medium",
                    "topic": "Caching",
                    "question_type": "Technical",
                    "total_questions": 2,
                },
            ).status_code
            == 403
        )
    finally:
        app.dependency_overrides.pop(get_ollama_service, None)


@pytest.mark.anyio
async def test_pinecone_service_enforces_user_id_metadata_and_query_filter():
    """Real PineconeService attaches user_id to metadata on upsert and filters by user_id + document_id on query."""
    mock_index = MagicMock()
    mock_index.upsert.return_value = {"upserted_count": 1}
    mock_index.query.return_value = {"matches": []}

    service = PineconeService(api_key="test-key", index_name="test-index")
    service._index = mock_index

    chunk = TextChunk(index=0, text="Confidential resume chunk", character_count=25, token_estimate=6)
    vector = [0.1] * 768

    await service.upsert_document_chunks(
        document_id="doc_user_b",
        chunks=[chunk],
        vectors=[vector],
        filename="resume.pdf",
        user_id="user-b-uuid",
    )

    upsert_kwargs = mock_index.upsert.call_args.kwargs
    upserted_meta = upsert_kwargs["vectors"][0][2]
    assert upserted_meta["user_id"] == "user-b-uuid"
    assert upserted_meta["document_id"] == "doc_user_b"

    await service.query_similar_chunks(
        query_vector=vector,
        document_id="doc_user_b",
        top_k=4,
        user_id="user-a-uuid",
    )

    query_kwargs = mock_index.query.call_args.kwargs
    assert query_kwargs["filter"] == {
        "document_id": {"$eq": "doc_user_b"},
        "user_id": {"$eq": "user-a-uuid"},
    }

