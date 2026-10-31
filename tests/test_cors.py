"""
Tests for CORS configuration, API contract validation, error responses, and integration readiness.
"""

import json
from fastapi.testclient import TestClient
import pytest

from app.core.config import settings
from app.main import app
from app.schemas.interview import ExperienceLevel, DifficultyLevel, QuestionType
from app.services.ollama_service import OllamaService, get_ollama_service
from app.services.session_service import get_session_service

client = TestClient(app)


# ============================================================================
# 1. CORS Configuration Tests
# ============================================================================


def test_cors_preflight_localhost_5173():
    """Verify preflight OPTIONS request for http://localhost:5173 returns allowed CORS headers."""
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "GET" in response.headers.get("access-control-allow-methods", "")
    # Credentials should remain disabled
    assert response.headers.get("access-control-allow-credentials") is None


def test_cors_preflight_127_0_0_1_5173():
    """Verify preflight OPTIONS request for http://127.0.0.1:5173 returns allowed CORS headers."""
    response = client.options(
        "/api/v1/interview/sessions",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"
    assert "POST" in response.headers.get("access-control-allow-methods", "")


def test_cors_actual_request_localhost_5173():
    """Verify standard GET request includes access-control-allow-origin header for allowed origin."""
    response = client.get(
        "/api/v1/health",
        headers={"Origin": "http://localhost:5173"},
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_disallowed_origin():
    """Verify requests from unauthorized origins do not receive access-control-allow-origin."""
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://unauthorized-domain.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    # Origin is not allowed, so access-control-allow-origin header is omitted
    assert response.headers.get("access-control-allow-origin") is None


def test_cors_custom_origin_parsing():
    """Verify Settings parses comma-separated and list CORS_ORIGINS correctly."""
    from app.core.config import Settings
    custom_settings = Settings(CORS_ORIGINS="http://localhost:3000,http://frontend.internal:5173")
    assert "http://localhost:3000" in custom_settings.CORS_ORIGINS
    assert "http://frontend.internal:5173" in custom_settings.CORS_ORIGINS


# ============================================================================
# 2. API Contract Validation (HTTP 422) Tests
# ============================================================================


def test_question_generation_invalid_payload_returns_422():
    """Verify invalid question generation payload returns 422 with validation errors."""
    # Missing required 'role' and 'topic'
    response = client.post(
        "/api/v1/interview/question",
        json={"difficulty": "Medium", "experience_level": "Junior"},
    )
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert any(err["loc"] == ["body", "role"] for err in data["detail"])


def test_question_generation_empty_string_returns_422():
    """Verify empty string or whitespace-only role returns 422."""
    response = client.post(
        "/api/v1/interview/question",
        json={
            "role": "   ",
            "topic": "Python",
            "difficulty": "Easy",
            "experience_level": "Junior",
            "question_type": "Technical",
        },
    )
    assert response.status_code == 422


def test_evaluation_invalid_payload_returns_422():
    """Verify invalid answer evaluation payload returns 422."""
    response = client.post(
        "/api/v1/interview/evaluate",
        json={"question": "What is Python?", "expected_answer_points": []},
    )
    assert response.status_code == 422


def test_create_session_out_of_bounds_total_questions_returns_422():
    """Verify total_questions < 1 or > 20 returns 422."""
    response = client.post(
        "/api/v1/interview/sessions",
        json={
            "role": "Frontend Developer",
            "experience_level": "Junior",
            "difficulty": "Easy",
            "topic": "CSS",
            "question_type": "Technical",
            "total_questions": 50,  # Max allowed is 20
        },
    )
    assert response.status_code == 422


def test_submit_answer_blank_text_returns_422():
    """Verify submitting blank candidate answer returns 422."""
    create_resp = client.post(
        "/api/v1/interview/sessions",
        json={
            "role": "QA Engineer",
            "experience_level": "Mid-Level",
            "difficulty": "Medium",
            "topic": "Selenium",
            "question_type": "Technical",
            "total_questions": 2,
        },
    )
    assert create_resp.status_code == 201
    session_id = create_resp.json()["session_id"]

    response = client.post(
        f"/api/v1/interview/sessions/{session_id}/answers",
        json={"candidate_answer": "    "},
    )
    assert response.status_code == 422


# ============================================================================
# 3. Missing Records Handling (HTTP 404) Tests
# ============================================================================


def test_missing_session_get_returns_404():
    """Verify GET /sessions/{non_existent_id} returns 404."""
    response = client.get("/api/v1/interview/sessions/non-existent-session-id")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_missing_session_next_question_returns_404():
    """Verify next-question for missing session returns 404."""
    response = client.post("/api/v1/interview/sessions/non-existent-session-id/next-question")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_missing_session_submit_answer_returns_404():
    """Verify submitting answer for missing session returns 404."""
    response = client.post(
        "/api/v1/interview/sessions/non-existent-session-id/answers",
        json={"candidate_answer": "Valid text answer."},
    )
    assert response.status_code == 404


def test_missing_session_complete_returns_404():
    """Verify completing missing session returns 404."""
    response = client.post("/api/v1/interview/sessions/non-existent-session-id/complete")
    assert response.status_code == 404


# ============================================================================
# 4. Safe Server Error Handling (HTTP 500)
# ============================================================================


def test_safe_500_error_response_no_traceback():
    """Verify unhandled exceptions return HTTP 500 JSON without leaking tracebacks or sensitive data."""
    class FailingService:
        def list_sessions(self, *args, **kwargs):
            # Simulate an unhandled database driver failure with secret-like detail
            raise RuntimeError("Database connection crashed: postgresql://admin:secret_pass@host/db")

    from app.services.session_service import get_session_service
    app.dependency_overrides[get_session_service] = lambda: FailingService()

    try:
        no_raise_client = TestClient(app, raise_server_exceptions=False)
        response = no_raise_client.get("/api/v1/interview/sessions")
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert data["detail"] == "An unexpected internal server error occurred."
        # Ensure password and database url are not in response
        assert "secret_pass" not in response.text
        assert "RuntimeError" not in response.text
        assert "Traceback" not in response.text
    finally:
        app.dependency_overrides.clear()
