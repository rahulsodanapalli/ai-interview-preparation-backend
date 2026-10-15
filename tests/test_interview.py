import json
from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from app.main import app
from app.schemas.interview import (
    EvaluationRating,
    InterviewEvaluationResponse,
)
from app.services.ollama_service import (
    OllamaConnectionError,
    OllamaService,
    OllamaTimeoutError,
    get_ollama_service,
)

client = TestClient(app)

MOCK_STRUCTURED_RESPONSE = {
    "question": "Explain how useState, useEffect, and useContext work in React.",
    "expected_answer_points": [
        "useState manages local state within functional components",
        "useEffect synchronizes side effects and responds to dependency array changes",
        "useContext provides a way to pass data through the component tree without prop drilling",
    ],
    "follow_up_question": "How do you prevent unnecessary re-renders when passing objects through React context?",
    "difficulty": "Medium",
    "topic": "React Hooks",
}

VALID_REQUEST_PAYLOAD = {
    "role": "Frontend Developer",
    "experience_level": "Junior",
    "difficulty": "Medium",
    "topic": "React Hooks",
    "question_type": "Technical",
}

MOCK_EVALUATION_RESPONSE = {
    "overall_score": 85,
    "rating": "Excellent",
    "correct_points": [
        "useState manages local state within functional components",
        "useEffect synchronizes side effects and dependency array changes",
    ],
    "missing_points": [
        "useContext provides a way to pass data without prop drilling",
    ],
    "strengths": [
        "Demonstrated solid understanding of React state and lifecycle hooks",
    ],
    "improvements": [
        "Elaborate on how useContext avoids prop drilling across nested components",
    ],
    "sample_answer": (
        "useState declares component-level reactive state. useEffect executes side effects "
        "and coordinates with dependency arrays. useContext consumes scoped React context without prop drilling."
    ),
}

VALID_EVALUATION_PAYLOAD = {
    "question": "Explain how useState, useEffect, and useContext work in React.",
    "expected_answer_points": [
        "useState manages local state within functional components",
        "useEffect synchronizes side effects and responds to dependency array changes",
        "useContext provides a way to pass data through the component tree without prop drilling",
    ],
    "candidate_answer": (
        "useState manages state in components. useEffect handles side effects when dependencies change. "
        "I am not entirely sure about useContext."
    ),
}


# ============================================================================
# Day 2 Tests: Question Generation
# ============================================================================


def test_generate_interview_question_success():
    """Test generating a structured question successfully with a mocked Ollama JSON response."""
    class MockOllamaServiceSuccess(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            return json.dumps(MOCK_STRUCTURED_RESPONSE)

    app.dependency_overrides[get_ollama_service] = lambda: MockOllamaServiceSuccess()
    try:
        response = client.post("/api/v1/interview/question", json=VALID_REQUEST_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == MOCK_STRUCTURED_RESPONSE["question"]
        assert data["expected_answer_points"] == MOCK_STRUCTURED_RESPONSE["expected_answer_points"]
        assert data["follow_up_question"] == MOCK_STRUCTURED_RESPONSE["follow_up_question"]
        assert data["difficulty"] == "Medium"
        assert data["topic"] == "React Hooks"
    finally:
        app.dependency_overrides.clear()


def test_generate_interview_question_markdown_code_block_stripping():
    """Test that markdown code fences (```json ... ```) returned by LLM are cleaned and parsed."""
    markdown_wrapped = f"```json\n{json.dumps(MOCK_STRUCTURED_RESPONSE)}\n```"

    class MockOllamaServiceMarkdown(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            return markdown_wrapped

    app.dependency_overrides[get_ollama_service] = lambda: MockOllamaServiceMarkdown()
    try:
        response = client.post("/api/v1/interview/question", json=VALID_REQUEST_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == MOCK_STRUCTURED_RESPONSE["question"]
        assert len(data["expected_answer_points"]) == 3
    finally:
        app.dependency_overrides.clear()


def test_generate_interview_question_missing_fields():
    """Test validation error when required fields are missing in request payload."""
    invalid_payload = {
        "role": "Frontend Developer",
        "experience_level": "Junior",
        "difficulty": "Medium",
    }
    response = client.post("/api/v1/interview/question", json=invalid_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_generate_interview_question_empty_strings():
    """Test validation error when fields contain only empty whitespace."""
    invalid_payload = {
        "role": "   ",
        "experience_level": "Junior",
        "difficulty": "Medium",
        "topic": "React Hooks",
        "question_type": "Technical",
    }
    response = client.post("/api/v1/interview/question", json=invalid_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_generate_interview_question_invalid_enum_values():
    """Test validation error when enum fields contain unsupported values."""
    invalid_payload = {
        "role": "Frontend Developer",
        "experience_level": "Senior-Expert",
        "difficulty": "Impossible",
        "topic": "React Hooks",
        "question_type": "Trivia",
    }
    response = client.post("/api/v1/interview/question", json=invalid_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_generate_interview_question_max_length_exceeded():
    """Test validation error when string fields exceed maximum allowed character length."""
    invalid_payload = {
        "role": "Frontend Developer " * 20,
        "experience_level": "Junior",
        "difficulty": "Medium",
        "topic": "React Hooks",
        "question_type": "Technical",
    }
    response = client.post("/api/v1/interview/question", json=invalid_payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_generate_interview_question_invalid_json_from_model():
    """Test graceful 502 error handling when LLM returns non-JSON plain text."""
    class MockOllamaServiceInvalidJson(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            return "Here is a question: Explain React hooks."

    app.dependency_overrides[get_ollama_service] = lambda: MockOllamaServiceInvalidJson()
    try:
        response = client.post("/api/v1/interview/question", json=VALID_REQUEST_PAYLOAD)
        assert response.status_code == 502
        data = response.json()
        assert data["detail"] == "The AI model returned an invalid JSON response."
    finally:
        app.dependency_overrides.clear()


def test_generate_interview_question_malformed_model_response():
    """Test graceful 502 error handling when LLM returns JSON missing required response fields."""
    class MockOllamaServiceMalformedJson(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            return json.dumps({"question": "Explain React hooks."})

    app.dependency_overrides[get_ollama_service] = lambda: MockOllamaServiceMalformedJson()
    try:
        response = client.post("/api/v1/interview/question", json=VALID_REQUEST_PAYLOAD)
        assert response.status_code == 502
        data = response.json()
        assert data["detail"] == "The AI model returned data that does not match the required question format."
    finally:
        app.dependency_overrides.clear()


def test_generate_interview_question_model_returns_json_array():
    """Test graceful 502 error handling when LLM returns a JSON array instead of a JSON object."""
    class MockOllamaServiceJsonArray(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            return json.dumps(["point 1", "point 2"])

    app.dependency_overrides[get_ollama_service] = lambda: MockOllamaServiceJsonArray()
    try:
        response = client.post("/api/v1/interview/question", json=VALID_REQUEST_PAYLOAD)
        assert response.status_code == 502
        data = response.json()
        assert data["detail"] == "The AI model returned an unexpected JSON structure."
    finally:
        app.dependency_overrides.clear()


def test_generate_interview_question_ollama_connection_failure():
    """Test graceful 503 error handling when Ollama connection fails."""
    class MockOllamaServiceConnectionError(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            raise OllamaConnectionError("Unable to connect to Ollama service.")

    app.dependency_overrides[get_ollama_service] = lambda: MockOllamaServiceConnectionError()
    try:
        response = client.post("/api/v1/interview/question", json=VALID_REQUEST_PAYLOAD)
        assert response.status_code == 503
        data = response.json()
        assert "AI inference service is currently unavailable" in data["detail"]
    finally:
        app.dependency_overrides.clear()


def test_generate_interview_question_ollama_timeout():
    """Test graceful 503 error handling when Ollama request times out."""
    class MockOllamaServiceTimeout(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            raise OllamaTimeoutError("Request to Ollama timed out.")

    app.dependency_overrides[get_ollama_service] = lambda: MockOllamaServiceTimeout()
    try:
        response = client.post("/api/v1/interview/question", json=VALID_REQUEST_PAYLOAD)
        assert response.status_code == 503
        data = response.json()
        assert "AI inference service is currently unavailable" in data["detail"]
    finally:
        app.dependency_overrides.clear()


# ============================================================================
# Day 3 Tests: Answer Evaluation
# ============================================================================


def test_evaluate_interview_answer_success():
    """Test evaluating an answer successfully with complete structured response."""
    class MockOllamaServiceEvalSuccess(OllamaService):
        async def evaluate_interview_answer(self, question, expected_answer_points, candidate_answer):
            return json.dumps(MOCK_EVALUATION_RESPONSE)

    app.dependency_overrides[get_ollama_service] = lambda: MockOllamaServiceEvalSuccess()
    try:
        response = client.post("/api/v1/interview/evaluate", json=VALID_EVALUATION_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["overall_score"] == 85
        assert data["rating"] == "Excellent"
        assert len(data["correct_points"]) == 2
        assert len(data["missing_points"]) == 1
        assert len(data["strengths"]) == 1
        assert len(data["improvements"]) == 1
        assert "useState declares component-level reactive state" in data["sample_answer"]
    finally:
        app.dependency_overrides.clear()


def test_evaluate_interview_answer_markdown_stripping():
    """Test markdown code block fences returned by LLM during evaluation are stripped cleanly."""
    markdown_wrapped = f"```json\n{json.dumps(MOCK_EVALUATION_RESPONSE)}\n```"

    class MockOllamaServiceEvalMarkdown(OllamaService):
        async def evaluate_interview_answer(self, question, expected_answer_points, candidate_answer):
            return markdown_wrapped

    app.dependency_overrides[get_ollama_service] = lambda: MockOllamaServiceEvalMarkdown()
    try:
        response = client.post("/api/v1/interview/evaluate", json=VALID_EVALUATION_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["overall_score"] == 85
        assert data["rating"] == "Excellent"
    finally:
        app.dependency_overrides.clear()


def test_evaluate_interview_answer_empty_answer():
    """Test that empty or whitespace-only candidate answers are rejected with 422."""
    payload = {
        "question": "Explain React hooks.",
        "expected_answer_points": ["State management"],
        "candidate_answer": "    ",
    }
    response = client.post("/api/v1/interview/evaluate", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_evaluate_interview_answer_missing_fields():
    """Test that missing required fields in evaluation payload are rejected with 422."""
    payload = {
        # 'question' is missing
        "expected_answer_points": ["State management"],
        "candidate_answer": "I use useState to store variables.",
    }
    response = client.post("/api/v1/interview/evaluate", json=payload)
    assert response.status_code == 422


def test_evaluate_interview_answer_max_length_exceeded():
    """Test that candidate answers exceeding 10,000 characters are rejected with 422."""
    payload = {
        "question": "Explain React hooks.",
        "expected_answer_points": ["State management"],
        "candidate_answer": "a" * 10001,
    }
    response = client.post("/api/v1/interview/evaluate", json=payload)
    assert response.status_code == 422


def test_evaluate_interview_answer_score_out_of_bounds_rejected():
    """Test that AI model scores outside 0-100 are rejected with 502 Bad Gateway."""
    invalid_score_response = dict(MOCK_EVALUATION_RESPONSE, overall_score=150)

    class MockOllamaServiceOutOfBounds(OllamaService):
        async def evaluate_interview_answer(self, question, expected_answer_points, candidate_answer):
            return json.dumps(invalid_score_response)

    app.dependency_overrides[get_ollama_service] = lambda: MockOllamaServiceOutOfBounds()
    try:
        response = client.post("/api/v1/interview/evaluate", json=VALID_EVALUATION_PAYLOAD)
        assert response.status_code == 502
        data = response.json()
        assert data["detail"] == "The AI model returned data that does not match the required evaluation format."
    finally:
        app.dependency_overrides.clear()


def test_evaluate_interview_answer_rating_normalized_by_rubric():
    """Test that ratings inconsistent with the scoring rubric are normalized."""
    # Score 85 with contradictory rating 'Needs Improvement' should normalize to 'Excellent'
    contradictory_response = dict(MOCK_EVALUATION_RESPONSE, overall_score=85, rating="Needs Improvement")

    class MockOllamaServiceMismatchedRating(OllamaService):
        async def evaluate_interview_answer(self, question, expected_answer_points, candidate_answer):
            return json.dumps(contradictory_response)

    app.dependency_overrides[get_ollama_service] = lambda: MockOllamaServiceMismatchedRating()
    try:
        response = client.post("/api/v1/interview/evaluate", json=VALID_EVALUATION_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert data["overall_score"] == 85
        assert data["rating"] == "Excellent"  # Normalized according to rubric rule
    finally:
        app.dependency_overrides.clear()


def test_evaluate_interview_answer_invalid_json_from_model():
    """Test graceful 502 handling when model returns non-JSON text."""
    class MockOllamaServiceInvalidJson(OllamaService):
        async def evaluate_interview_answer(self, question, expected_answer_points, candidate_answer):
            return "Evaluation text without JSON."

    app.dependency_overrides[get_ollama_service] = lambda: MockOllamaServiceInvalidJson()
    try:
        response = client.post("/api/v1/interview/evaluate", json=VALID_EVALUATION_PAYLOAD)
        assert response.status_code == 502
        data = response.json()
        assert data["detail"] == "The AI model returned an invalid JSON response."
    finally:
        app.dependency_overrides.clear()


def test_evaluate_interview_answer_malformed_model_response():
    """Test graceful 502 handling when model returns JSON missing required fields."""
    class MockOllamaServiceMalformed(OllamaService):
        async def evaluate_interview_answer(self, question, expected_answer_points, candidate_answer):
            return json.dumps({"overall_score": 70})

    app.dependency_overrides[get_ollama_service] = lambda: MockOllamaServiceMalformed()
    try:
        response = client.post("/api/v1/interview/evaluate", json=VALID_EVALUATION_PAYLOAD)
        assert response.status_code == 502
        data = response.json()
        assert data["detail"] == "The AI model returned data that does not match the required evaluation format."
    finally:
        app.dependency_overrides.clear()


def test_evaluate_interview_answer_ollama_connection_failure():
    """Test graceful 503 handling when Ollama connection fails."""
    class MockOllamaServiceConnFail(OllamaService):
        async def evaluate_interview_answer(self, question, expected_answer_points, candidate_answer):
            raise OllamaConnectionError("Unable to reach Ollama runtime.")

    app.dependency_overrides[get_ollama_service] = lambda: MockOllamaServiceConnFail()
    try:
        response = client.post("/api/v1/interview/evaluate", json=VALID_EVALUATION_PAYLOAD)
        assert response.status_code == 503
        data = response.json()
        assert "AI inference service is currently unavailable" in data["detail"]
    finally:
        app.dependency_overrides.clear()


def test_evaluate_interview_answer_ollama_timeout():
    """Test graceful 503 handling when Ollama times out during answer evaluation."""
    class MockOllamaServiceTimeout(OllamaService):
        async def evaluate_interview_answer(self, question, expected_answer_points, candidate_answer):
            raise OllamaTimeoutError("Evaluation timed out.")

    app.dependency_overrides[get_ollama_service] = lambda: MockOllamaServiceTimeout()
    try:
        response = client.post("/api/v1/interview/evaluate", json=VALID_EVALUATION_PAYLOAD)
        assert response.status_code == 503
        data = response.json()
        assert "AI inference service is currently unavailable" in data["detail"]
    finally:
        app.dependency_overrides.clear()


def test_evaluation_rating_rubric_boundaries():
    """Unit test rubric score boundaries for all 4 ratings."""
    assert EvaluationRating.from_score(0) == EvaluationRating.NEEDS_IMPROVEMENT
    assert EvaluationRating.from_score(39) == EvaluationRating.NEEDS_IMPROVEMENT
    assert EvaluationRating.from_score(40) == EvaluationRating.FAIR
    assert EvaluationRating.from_score(59) == EvaluationRating.FAIR
    assert EvaluationRating.from_score(60) == EvaluationRating.GOOD
    assert EvaluationRating.from_score(79) == EvaluationRating.GOOD
    assert EvaluationRating.from_score(80) == EvaluationRating.EXCELLENT
    assert EvaluationRating.from_score(100) == EvaluationRating.EXCELLENT
