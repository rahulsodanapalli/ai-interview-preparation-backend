import json
from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.schemas.interview import EvaluationRating
from app.services.ollama_service import (
    OllamaConnectionError,
    OllamaResponseError,
    OllamaService,
    OllamaTimeoutError,
    get_ollama_service,
)
from app.services.session_service import get_session_service

client = TestClient(app)

MOCK_QUESTION_1 = {
    "question": "Explain the virtual DOM in React and how reconciliation works.",
    "expected_answer_points": [
        "In-memory representation of real DOM elements",
        "Diffing algorithm compares previous and new trees",
        "Batched updates to real DOM minimize repaints and reflows",
    ],
    "follow_up_question": "What is the key prop and why is it essential in list rendering?",
    "difficulty": "Medium",
    "topic": "React Architecture",
}

MOCK_QUESTION_2 = {
    "question": "What is the purpose of useEffect dependency array and how do stale closures occur?",
    "expected_answer_points": [
        "Dependency array tells React when to re-execute effect function",
        "Omitting dependencies can cause closures over stale state or props",
        "Cleanup function runs before the next effect run or on unmount",
    ],
    "follow_up_question": "How does useCallback help prevent stale closures?",
    "difficulty": "Medium",
    "topic": "React Hooks",
}

MOCK_EVALUATION_1 = {
    "overall_score": 85,
    "rating": "Excellent",
    "correct_points": [
        "Virtual DOM is an in-memory copy of UI tree",
        "Diffing algorithm updates only changed nodes",
    ],
    "missing_points": [
        "Batching details in React 18",
    ],
    "strengths": [
        "Accurate description of reconciliation diffing",
    ],
    "improvements": [
        "Mention fibers and concurrent rendering updates",
    ],
    "sample_answer": "The virtual DOM is a lightweight JS tree representing UI nodes...",
}

MOCK_EVALUATION_2 = {
    "overall_score": 65,
    "rating": "Good",
    "correct_points": [
        "Explained dependency array and effect re-runs",
    ],
    "missing_points": [
        "Did not explain stale closure mechanics clearly",
    ],
    "strengths": [
        "Understands basic hook dependencies",
    ],
    "improvements": [
        "Deepen understanding of JS closures over old lexical environments",
    ],
    "sample_answer": "The dependency array controls effect executions...",
}

VALID_SESSION_PAYLOAD = {
    "role": "Frontend Developer",
    "experience_level": "Junior",
    "difficulty": "Medium",
    "topic": "React",
    "question_type": "Technical",
    "total_questions": 3,
}


@pytest.fixture(autouse=True)
def reset_session_store():
    """Reset in-memory session store before each test run."""
    session_service = get_session_service()
    session_service.reset_store()
    yield
    session_service.reset_store()


# ============================================================================
# Session Creation Tests
# ============================================================================


def test_create_session_success():
    """Test creating a new interview session successfully with full parameters."""
    response = client.post("/api/v1/interview/sessions", json=VALID_SESSION_PAYLOAD)
    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert len(data["session_id"]) >= 32
    assert data["role"] == "Frontend Developer"
    assert data["experience_level"] == "Junior"
    assert data["difficulty"] == "Medium"
    assert data["topic"] == "React"
    assert data["question_type"] == "Technical"
    assert data["total_questions"] == 3
    assert data["completed_questions"] == 0
    assert data["current_question_number"] == 0
    assert data["status"] == "in_progress"
    assert data["questions"] == []
    assert "created_at" in data
    assert "updated_at" in data


def test_create_session_default_total_questions():
    """Test that total_questions defaults to 5 if omitted in creation request."""
    payload = {
        "role": "Backend Engineer",
        "experience_level": "Senior",
        "difficulty": "Hard",
        "topic": "System Design",
        "question_type": "Scenario-Based",
    }
    response = client.post("/api/v1/interview/sessions", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["total_questions"] == 5


@pytest.mark.parametrize(
    "invalid_payload",
    [
        {"role": "   ", "experience_level": "Junior", "difficulty": "Medium", "topic": "React", "question_type": "Technical"},
        {"role": "Dev", "experience_level": "Junior", "difficulty": "Medium", "topic": "   ", "question_type": "Technical"},
        {"role": "Dev", "experience_level": "SuperSenior", "difficulty": "Medium", "topic": "React", "question_type": "Technical"},
        {"role": "Dev", "experience_level": "Junior", "difficulty": "Extreme", "topic": "React", "question_type": "Technical"},
        {"role": "Dev", "experience_level": "Junior", "difficulty": "Medium", "topic": "React", "question_type": "Essay"},
        {"role": "Dev", "experience_level": "Junior", "difficulty": "Medium", "topic": "React", "question_type": "Technical", "total_questions": 0},
        {"role": "Dev", "experience_level": "Junior", "difficulty": "Medium", "topic": "React", "question_type": "Technical", "total_questions": 25},
        {"role": "D" * 101, "experience_level": "Junior", "difficulty": "Medium", "topic": "React", "question_type": "Technical"},
    ],
)
def test_create_session_validation_errors(invalid_payload):
    """Test that invalid fields, empty strings, and out-of-range counts are rejected with 422."""
    response = client.post("/api/v1/interview/sessions", json=invalid_payload)
    assert response.status_code == 422


# ============================================================================
# Session Retrieval Tests
# ============================================================================


def test_get_session_details_success():
    """Test retrieving session details and progress for an existing session."""
    create_resp = client.post("/api/v1/interview/sessions", json=VALID_SESSION_PAYLOAD)
    assert create_resp.status_code == 201
    session_id = create_resp.json()["session_id"]

    get_resp = client.get(f"/api/v1/interview/sessions/{session_id}")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["session_id"] == session_id
    assert data["role"] == "Frontend Developer"
    assert data["status"] == "in_progress"


def test_get_session_not_found():
    """Test retrieving a non-existent session ID returns 404 Not Found."""
    response = client.get("/api/v1/interview/sessions/non-existent-session-id")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


# ============================================================================
# Next Question Generation in Session
# ============================================================================


def test_generate_next_question_success():
    """Test generating the first question in a session with mocked AI response."""
    class MockOllama(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            return json.dumps(MOCK_QUESTION_1)

    app.dependency_overrides[get_ollama_service] = lambda: MockOllama()
    try:
        create_resp = client.post("/api/v1/interview/sessions", json=VALID_SESSION_PAYLOAD)
        session_id = create_resp.json()["session_id"]

        next_q_resp = client.post(f"/api/v1/interview/sessions/{session_id}/next-question")
        assert next_q_resp.status_code == 200
        q_data = next_q_resp.json()
        assert q_data["session_id"] == session_id
        assert q_data["question_number"] == 1
        assert q_data["total_questions"] == 3
        assert q_data["question"] == MOCK_QUESTION_1["question"]
        assert len(q_data["expected_answer_points"]) == 3

        # Verify session state updated
        session_resp = client.get(f"/api/v1/interview/sessions/{session_id}")
        session_data = session_resp.json()
        assert session_data["current_question_number"] == 1
        assert len(session_data["questions"]) == 1
        assert session_data["questions"][0]["candidate_answer"] is None
        assert session_data["questions"][0]["evaluation"] is None
    finally:
        app.dependency_overrides.clear()


def test_generate_next_question_prevent_unanswered():
    """Test that requesting next question fails when current question is unanswered."""
    class MockOllama(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            return json.dumps(MOCK_QUESTION_1)

    app.dependency_overrides[get_ollama_service] = lambda: MockOllama()
    try:
        create_resp = client.post("/api/v1/interview/sessions", json=VALID_SESSION_PAYLOAD)
        session_id = create_resp.json()["session_id"]

        # First question generation succeeds
        resp1 = client.post(f"/api/v1/interview/sessions/{session_id}/next-question")
        assert resp1.status_code == 200

        # Attempting second question before answering question 1 must fail
        resp2 = client.post(f"/api/v1/interview/sessions/{session_id}/next-question")
        assert resp2.status_code == 400
        assert "has not been answered yet" in resp2.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_generate_next_question_cannot_exceed_total():
    """Test that requesting questions beyond total_questions is rejected with 400."""
    class MockOllama(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            return json.dumps(MOCK_QUESTION_1)

        async def evaluate_interview_answer(self, question, expected_answer_points, candidate_answer):
            return json.dumps(MOCK_EVALUATION_1)

    app.dependency_overrides[get_ollama_service] = lambda: MockOllama()
    try:
        # Create session with total_questions = 1
        payload = dict(VALID_SESSION_PAYLOAD, total_questions=1)
        create_resp = client.post("/api/v1/interview/sessions", json=payload)
        session_id = create_resp.json()["session_id"]

        # Generate question 1
        client.post(f"/api/v1/interview/sessions/{session_id}/next-question")
        # Answer question 1
        client.post(
            f"/api/v1/interview/sessions/{session_id}/answers",
            json={"candidate_answer": "Virtual DOM is in-memory DOM representation."},
        )

        # Attempt to generate question 2 when total is 1
        over_resp = client.post(f"/api/v1/interview/sessions/{session_id}/next-question")
        assert over_resp.status_code == 400
        assert "maximum total of 1 questions reached" in over_resp.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_generate_next_question_session_completed():
    """Test that generating questions in a completed session is rejected with 400."""
    create_resp = client.post("/api/v1/interview/sessions", json=VALID_SESSION_PAYLOAD)
    session_id = create_resp.json()["session_id"]

    # Complete the session immediately
    complete_resp = client.post(f"/api/v1/interview/sessions/{session_id}/complete")
    assert complete_resp.status_code == 200

    next_q_resp = client.post(f"/api/v1/interview/sessions/{session_id}/next-question")
    assert next_q_resp.status_code == 400
    assert "already completed" in next_q_resp.json()["detail"]


def test_generate_next_question_session_not_found():
    """Test generating question for non-existent session returns 404."""
    response = client.post("/api/v1/interview/sessions/non-existent-uuid/next-question")
    assert response.status_code == 404


# ============================================================================
# Answer Submission & Evaluation Tests
# ============================================================================


def test_submit_answer_success():
    """Test submitting answer for active session question and receiving evaluation."""
    class MockOllama(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            return json.dumps(MOCK_QUESTION_1)

        async def evaluate_interview_answer(self, question, expected_answer_points, candidate_answer):
            return json.dumps(MOCK_EVALUATION_1)

    app.dependency_overrides[get_ollama_service] = lambda: MockOllama()
    try:
        create_resp = client.post("/api/v1/interview/sessions", json=VALID_SESSION_PAYLOAD)
        session_id = create_resp.json()["session_id"]

        client.post(f"/api/v1/interview/sessions/{session_id}/next-question")

        answer_payload = {
            "candidate_answer": "Virtual DOM is an in-memory representation. Diffing reconciles changes.",
        }
        ans_resp = client.post(f"/api/v1/interview/sessions/{session_id}/answers", json=answer_payload)
        assert ans_resp.status_code == 200
        ans_data = ans_resp.json()
        assert ans_data["session_id"] == session_id
        assert ans_data["question_number"] == 1
        assert ans_data["completed_questions"] == 1
        assert ans_data["total_questions"] == 3
        assert ans_data["evaluation"]["overall_score"] == 85
        assert ans_data["evaluation"]["rating"] == "Excellent"

        # Verify session state reflects submitted answer and evaluation
        sess_resp = client.get(f"/api/v1/interview/sessions/{session_id}")
        sess_data = sess_resp.json()
        assert sess_data["completed_questions"] == 1
        assert sess_data["questions"][0]["candidate_answer"] == answer_payload["candidate_answer"]
        assert sess_data["questions"][0]["evaluation"]["overall_score"] == 85
    finally:
        app.dependency_overrides.clear()


def test_submit_answer_duplicate_prevented():
    """Test that submitting duplicate answers for the same question is prevented with 400."""
    class MockOllama(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            return json.dumps(MOCK_QUESTION_1)

        async def evaluate_interview_answer(self, question, expected_answer_points, candidate_answer):
            return json.dumps(MOCK_EVALUATION_1)

    app.dependency_overrides[get_ollama_service] = lambda: MockOllama()
    try:
        create_resp = client.post("/api/v1/interview/sessions", json=VALID_SESSION_PAYLOAD)
        session_id = create_resp.json()["session_id"]

        client.post(f"/api/v1/interview/sessions/{session_id}/next-question")

        answer_payload = {"candidate_answer": "First valid answer."}
        # First submission succeeds
        resp1 = client.post(f"/api/v1/interview/sessions/{session_id}/answers", json=answer_payload)
        assert resp1.status_code == 200

        # Duplicate submission must fail
        resp2 = client.post(f"/api/v1/interview/sessions/{session_id}/answers", json=answer_payload)
        assert resp2.status_code == 400
        assert "already been answered" in resp2.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_submit_answer_before_any_question():
    """Test that submitting an answer when no questions exist returns 400."""
    create_resp = client.post("/api/v1/interview/sessions", json=VALID_SESSION_PAYLOAD)
    session_id = create_resp.json()["session_id"]

    answer_payload = {"candidate_answer": "Answer with no question."}
    response = client.post(f"/api/v1/interview/sessions/{session_id}/answers", json=answer_payload)
    assert response.status_code == 400
    assert "No questions have been generated" in response.json()["detail"]


def test_submit_answer_nonexistent_question_number():
    """Test that answering a non-existent question sequence number returns 400."""
    class MockOllama(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            return json.dumps(MOCK_QUESTION_1)

    app.dependency_overrides[get_ollama_service] = lambda: MockOllama()
    try:
        create_resp = client.post("/api/v1/interview/sessions", json=VALID_SESSION_PAYLOAD)
        session_id = create_resp.json()["session_id"]

        client.post(f"/api/v1/interview/sessions/{session_id}/next-question")

        payload = {"candidate_answer": "Answer", "question_number": 9}
        response = client.post(f"/api/v1/interview/sessions/{session_id}/answers", json=payload)
        assert response.status_code == 400
        assert "Question #9 does not exist" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_submit_answer_when_session_completed():
    """Test that submitting answer after session is completed returns 400."""
    create_resp = client.post("/api/v1/interview/sessions", json=VALID_SESSION_PAYLOAD)
    session_id = create_resp.json()["session_id"]

    client.post(f"/api/v1/interview/sessions/{session_id}/complete")

    payload = {"candidate_answer": "Late answer."}
    response = client.post(f"/api/v1/interview/sessions/{session_id}/answers", json=payload)
    assert response.status_code == 400
    assert "session is completed" in response.json()["detail"]


def test_submit_answer_empty_rejected():
    """Test that empty or whitespace-only answers are rejected with 422."""
    create_resp = client.post("/api/v1/interview/sessions", json=VALID_SESSION_PAYLOAD)
    session_id = create_resp.json()["session_id"]

    response = client.post(
        f"/api/v1/interview/sessions/{session_id}/answers",
        json={"candidate_answer": "   "},
    )
    assert response.status_code == 422


# ============================================================================
# Session Completion & Summary Tests
# ============================================================================


def test_complete_session_full_lifecycle_and_summary():
    """Test complete 2-question session lifecycle and verify aggregated summary statistics."""
    class MockOllama(OllamaService):
        q_count = 0
        eval_count = 0

        async def generate_chat_completion(self, messages, model=None, format=None):
            self.q_count += 1
            if self.q_count == 1:
                return json.dumps(MOCK_QUESTION_1)
            return json.dumps(MOCK_QUESTION_2)

        async def evaluate_interview_answer(self, question, expected_answer_points, candidate_answer):
            self.eval_count += 1
            if self.eval_count == 1:
                return json.dumps(MOCK_EVALUATION_1)  # score 85
            return json.dumps(MOCK_EVALUATION_2)      # score 65

    mock_ollama = MockOllama()
    app.dependency_overrides[get_ollama_service] = lambda: mock_ollama
    try:
        payload = dict(VALID_SESSION_PAYLOAD, total_questions=2)
        create_resp = client.post("/api/v1/interview/sessions", json=payload)
        session_id = create_resp.json()["session_id"]

        # Question 1: Generate & Answer
        client.post(f"/api/v1/interview/sessions/{session_id}/next-question")
        client.post(
            f"/api/v1/interview/sessions/{session_id}/answers",
            json={"candidate_answer": "Answer 1."},
        )

        # Question 2: Generate & Answer
        client.post(f"/api/v1/interview/sessions/{session_id}/next-question")
        client.post(
            f"/api/v1/interview/sessions/{session_id}/answers",
            json={"candidate_answer": "Answer 2."},
        )

        # Complete Session
        comp_resp = client.post(f"/api/v1/interview/sessions/{session_id}/complete")
        assert comp_resp.status_code == 200
        summary = comp_resp.json()
        assert summary["session_id"] == session_id
        assert summary["role"] == "Frontend Developer"
        assert summary["topic"] == "React"
        assert summary["total_questions"] == 2
        assert summary["questions_answered"] == 2
        # Average: (85 + 65) / 2 = 75.0
        assert summary["average_score"] == 75.0
        # 75.0 falls in Good rating rubric (60-79)
        assert summary["overall_rating"] == "Good"
        assert summary["session_status"] == "completed"
        # Strengths aggregated from both answers
        assert len(summary["aggregated_strengths"]) == 2
        assert "Accurate description of reconciliation diffing" in summary["aggregated_strengths"]
        assert "Understands basic hook dependencies" in summary["aggregated_strengths"]
        # Improvement areas aggregated
        assert len(summary["aggregated_improvement_areas"]) == 2

        # Verify subsequent GET returns completed status
        get_resp = client.get(f"/api/v1/interview/sessions/{session_id}")
        assert get_resp.json()["status"] == "completed"
    finally:
        app.dependency_overrides.clear()


def test_complete_session_zero_answers_no_division_by_zero():
    """Test completing a session with no answered questions produces safe 0.0 metrics without error."""
    create_resp = client.post("/api/v1/interview/sessions", json=VALID_SESSION_PAYLOAD)
    session_id = create_resp.json()["session_id"]

    # Complete immediately without asking or answering questions
    comp_resp = client.post(f"/api/v1/interview/sessions/{session_id}/complete")
    assert comp_resp.status_code == 200
    summary = comp_resp.json()
    assert summary["session_id"] == session_id
    assert summary["total_questions"] == 3
    assert summary["questions_answered"] == 0
    assert summary["average_score"] == 0.0
    assert summary["overall_rating"] == "Needs Improvement"
    assert summary["aggregated_strengths"] == []
    assert summary["aggregated_improvement_areas"] == []
    assert summary["session_status"] == "completed"


def test_complete_session_unanswered_question_excluded_from_average():
    """Test that generated but unanswered questions do not pollute evaluated average score."""
    class MockOllama(OllamaService):
        q_count = 0

        async def generate_chat_completion(self, messages, model=None, format=None):
            self.q_count += 1
            if self.q_count == 1:
                return json.dumps(MOCK_QUESTION_1)
            return json.dumps(MOCK_QUESTION_2)

        async def evaluate_interview_answer(self, question, expected_answer_points, candidate_answer):
            return json.dumps(MOCK_EVALUATION_1)  # score 85

    mock_ollama = MockOllama()
    app.dependency_overrides[get_ollama_service] = lambda: mock_ollama
    try:
        create_resp = client.post("/api/v1/interview/sessions", json=VALID_SESSION_PAYLOAD)
        session_id = create_resp.json()["session_id"]

        # Generate & Answer Question 1 (Score 85)
        client.post(f"/api/v1/interview/sessions/{session_id}/next-question")
        client.post(
            f"/api/v1/interview/sessions/{session_id}/answers",
            json={"candidate_answer": "My answer to question 1."},
        )

        # Generate Question 2, but do NOT answer it
        client.post(f"/api/v1/interview/sessions/{session_id}/next-question")

        # Complete session early
        comp_resp = client.post(f"/api/v1/interview/sessions/{session_id}/complete")
        assert comp_resp.status_code == 200
        summary = comp_resp.json()
        assert summary["total_questions"] == 3
        assert summary["questions_answered"] == 1
        assert summary["average_score"] == 85.0
        assert summary["overall_rating"] == "Excellent"
    finally:
        app.dependency_overrides.clear()


def test_complete_session_not_found():
    """Test completing non-existent session returns 404."""
    response = client.post("/api/v1/interview/sessions/unknown-session-id/complete")
    assert response.status_code == 404


# ============================================================================
# AI Service Failure Handling in Session Flows
# ============================================================================


def test_session_next_question_ollama_unavailable():
    """Test graceful 503 error handling when Ollama fails during session question generation."""
    class MockOllamaConnFail(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            raise OllamaConnectionError("Ollama is down.")

    app.dependency_overrides[get_ollama_service] = lambda: MockOllamaConnFail()
    try:
        create_resp = client.post("/api/v1/interview/sessions", json=VALID_SESSION_PAYLOAD)
        session_id = create_resp.json()["session_id"]

        response = client.post(f"/api/v1/interview/sessions/{session_id}/next-question")
        assert response.status_code == 503
        assert "AI inference service is currently unavailable" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_session_submit_answer_ollama_timeout():
    """Test graceful 503 error handling when Ollama times out during session answer evaluation."""
    class MockOllama(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            return json.dumps(MOCK_QUESTION_1)

        async def evaluate_interview_answer(self, question, expected_answer_points, candidate_answer):
            raise OllamaTimeoutError("Timed out evaluating answer.")

    app.dependency_overrides[get_ollama_service] = lambda: MockOllama()
    try:
        create_resp = client.post("/api/v1/interview/sessions", json=VALID_SESSION_PAYLOAD)
        session_id = create_resp.json()["session_id"]

        client.post(f"/api/v1/interview/sessions/{session_id}/next-question")

        response = client.post(
            f"/api/v1/interview/sessions/{session_id}/answers",
            json={"candidate_answer": "Valid answer text."},
        )
        assert response.status_code == 503
        assert "AI inference service is currently unavailable" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()
