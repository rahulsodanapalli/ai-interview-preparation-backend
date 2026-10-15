"""
Interview Session Management Service.

NOTE ON STORAGE & PERSISTENCE:
This service implements an in-memory session store for interview sessions.
In-memory sessions are stored in process memory and are lost when the backend
server restarts. This storage mechanism is intended for development, demonstration,
and automated testing, and is NOT suitable for production persistence.
Future milestones will integrate a database (such as PostgreSQL) for persistent storage.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional
import uuid

from app.schemas.interview import (
    EvaluationRating,
    InterviewEvaluationResponse,
    InterviewQuestionResponse,
)
from app.schemas.session import (
    CreateSessionRequest,
    InterviewSessionSummaryResponse,
    SessionDetailResponse,
    SessionNextQuestionResponse,
    SessionQuestionItem,
    SessionStatus,
    SessionSubmitAnswerResponse,
)

logger = logging.getLogger(__name__)


class SessionNotFoundError(Exception):
    """Raised when an interview session ID is not found in the store."""
    pass


class SessionStateError(Exception):
    """Raised when an invalid session lifecycle transition or state rule is violated."""
    pass


class SessionService:
    """
    In-memory session manager handling interview lifecycles, question sequencing,
    answer evaluation tracking, and session summaries.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionDetailResponse] = {}

    def create_session(self, payload: CreateSessionRequest) -> SessionDetailResponse:
        """
        Create and initialize a new interview session in memory.

        Args:
            payload: Validated session configuration parameters.

        Returns:
            The newly created SessionDetailResponse in 'in_progress' status.
        """
        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        session = SessionDetailResponse(
            session_id=session_id,
            role=payload.role,
            experience_level=payload.experience_level,
            difficulty=payload.difficulty,
            topic=payload.topic,
            question_type=payload.question_type,
            total_questions=payload.total_questions,
            completed_questions=0,
            current_question_number=0,
            status=SessionStatus.IN_PROGRESS,
            questions=[],
            created_at=now,
            updated_at=now,
        )

        self._sessions[session_id] = session
        logger.info("Created interview session %s for role '%s' on topic '%s'.", session_id, payload.role, payload.topic)
        return session

    def get_session(self, session_id: str) -> SessionDetailResponse:
        """
        Retrieve session details and current progress by session ID.

        Args:
            session_id: Unique UUID string for the session.

        Returns:
            The SessionDetailResponse.

        Raises:
            SessionNotFoundError: If session_id is not in memory.
        """
        session = self._sessions.get(session_id)
        if not session:
            logger.warning("Session lookup failed: session '%s' not found.", session_id)
            raise SessionNotFoundError(f"Interview session '{session_id}' not found.")
        return session

    def validate_can_generate_question(self, session_id: str) -> SessionDetailResponse:
        """
        Validate that the session is in a valid state to generate the next question.

        Rules:
        - Session must exist.
        - Session must not be completed.
        - The current active question (if any) must already be answered before asking for the next.
        - The number of generated questions must not exceed total_questions.

        Returns:
            The active SessionDetailResponse.

        Raises:
            SessionNotFoundError: If session is missing.
            SessionStateError: If lifecycle rules prevent generating a new question.
        """
        session = self.get_session(session_id)

        if session.status == SessionStatus.COMPLETED:
            raise SessionStateError("Cannot generate next question: interview session is already completed.")

        # Check if the most recently generated question is still waiting for an answer
        unanswered = [q for q in session.questions if q.candidate_answer is None]
        if unanswered:
            current_num = unanswered[0].question_number
            raise SessionStateError(
                f"Current question #{current_num} has not been answered yet. "
                "Submit an answer before requesting the next question."
            )

        # Check if max configured questions have already been generated
        if len(session.questions) >= session.total_questions:
            raise SessionStateError(
                f"Cannot generate more questions: maximum total of {session.total_questions} "
                "questions reached for this session."
            )

        return session

    def add_question_to_session(
        self,
        session_id: str,
        question_data: InterviewQuestionResponse,
    ) -> SessionNextQuestionResponse:
        """
        Append a generated question to the session and advance the question number.

        Args:
            session_id: Unique UUID string for the session.
            question_data: Validated question generated by AI model.

        Returns:
            SessionNextQuestionResponse containing the question and progress metadata.
        """
        session = self.get_session(session_id)
        next_question_number = len(session.questions) + 1

        question_item = SessionQuestionItem(
            question_number=next_question_number,
            question=question_data.question,
            expected_answer_points=question_data.expected_answer_points,
            follow_up_question=question_data.follow_up_question,
            difficulty=question_data.difficulty,
            topic=question_data.topic,
            candidate_answer=None,
            evaluation=None,
        )

        session.questions.append(question_item)
        session.current_question_number = next_question_number
        session.updated_at = datetime.now(timezone.utc).isoformat()

        logger.info("Added question #%d to session %s.", next_question_number, session_id)

        return SessionNextQuestionResponse(
            session_id=session.session_id,
            question_number=next_question_number,
            total_questions=session.total_questions,
            question=question_item.question,
            expected_answer_points=question_item.expected_answer_points,
            follow_up_question=question_item.follow_up_question,
            difficulty=question_item.difficulty,
            topic=question_item.topic,
        )

    def validate_can_submit_answer(
        self,
        session_id: str,
        question_number: Optional[int] = None,
    ) -> SessionQuestionItem:
        """
        Validate that a candidate answer can be submitted for the specified or current question.

        Rules:
        - Session must exist.
        - Session must not be completed.
        - At least one question must exist.
        - The target question must exist and must not have been answered already (no duplicates).

        Returns:
            The target SessionQuestionItem to be answered.

        Raises:
            SessionNotFoundError: If session is missing.
            SessionStateError: If answering violates session rules.
        """
        session = self.get_session(session_id)

        if session.status == SessionStatus.COMPLETED:
            raise SessionStateError("Cannot submit answer: interview session is completed.")

        if not session.questions:
            raise SessionStateError("No questions have been generated for this session yet. Request a question first.")

        if question_number is not None:
            # Find specific question by sequence number
            matching = [q for q in session.questions if q.question_number == question_number]
            if not matching:
                raise SessionStateError(f"Question #{question_number} does not exist in this session.")
            target = matching[0]
            if target.candidate_answer is not None:
                raise SessionStateError(f"Question #{question_number} has already been answered. Duplicate submissions are not allowed.")
            return target
        else:
            # Target active question (the latest question generated)
            target = session.questions[-1]
            if target.candidate_answer is not None:
                raise SessionStateError(
                    f"Current question #{target.question_number} has already been answered. "
                    "Please request the next question."
                )
            return target

    def record_answer_and_evaluation(
        self,
        session_id: str,
        question_number: int,
        candidate_answer: str,
        evaluation: InterviewEvaluationResponse,
    ) -> SessionSubmitAnswerResponse:
        """
        Record candidate answer and AI evaluation on the target question and update progress.

        Args:
            session_id: Unique UUID string for the session.
            question_number: Sequence number of the question answered.
            candidate_answer: The candidate's answer text.
            evaluation: Validated evaluation from AI model.

        Returns:
            SessionSubmitAnswerResponse with evaluation results and updated progress.
        """
        session = self.get_session(session_id)
        matching = [q for q in session.questions if q.question_number == question_number]
        if not matching:
            raise SessionStateError(f"Question #{question_number} does not exist in this session.")

        target = matching[0]
        target.candidate_answer = candidate_answer
        target.evaluation = evaluation

        # Update completed questions count
        session.completed_questions = sum(1 for q in session.questions if q.evaluation is not None)
        session.updated_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Recorded evaluation for question #%d in session %s. Score: %d (%s). Progress: %d/%d.",
            question_number,
            session_id,
            evaluation.overall_score,
            evaluation.rating.value,
            session.completed_questions,
            session.total_questions,
        )

        return SessionSubmitAnswerResponse(
            session_id=session.session_id,
            question_number=question_number,
            completed_questions=session.completed_questions,
            total_questions=session.total_questions,
            status=session.status,
            evaluation=evaluation,
        )

    def complete_session(self, session_id: str) -> InterviewSessionSummaryResponse:
        """
        Mark the session as completed and calculate the final performance summary.

        Calculates average score, rubric-aligned rating, aggregated strengths, and
        improvement areas from actually evaluated questions. Safely handles sessions
        with zero answered questions without division-by-zero errors.

        Args:
            session_id: Unique UUID string for the session.

        Returns:
            InterviewSessionSummaryResponse.

        Raises:
            SessionNotFoundError: If session is missing.
        """
        session = self.get_session(session_id)

        session.status = SessionStatus.COMPLETED
        session.updated_at = datetime.now(timezone.utc).isoformat()

        evaluated_questions = [q for q in session.questions if q.evaluation is not None]
        questions_answered = len(evaluated_questions)

        if questions_answered == 0:
            average_score = 0.0
            overall_rating = EvaluationRating.NEEDS_IMPROVEMENT
            aggregated_strengths: List[str] = []
            aggregated_improvement_areas: List[str] = []
        else:
            total_score = sum(q.evaluation.overall_score for q in evaluated_questions)
            average_score = round(total_score / questions_answered, 2)
            overall_rating = EvaluationRating.from_score(round(average_score))

            # Aggregate unique strengths preserving order of discovery
            aggregated_strengths = []
            for q in evaluated_questions:
                for s in q.evaluation.strengths:
                    if s not in aggregated_strengths:
                        aggregated_strengths.append(s)

            # Aggregate unique improvements preserving order
            aggregated_improvement_areas = []
            for q in evaluated_questions:
                for imp in q.evaluation.improvements:
                    if imp not in aggregated_improvement_areas:
                        aggregated_improvement_areas.append(imp)

        logger.info(
            "Completed session %s. Answered: %d/%d, Average score: %.2f (%s).",
            session_id,
            questions_answered,
            session.total_questions,
            average_score,
            overall_rating.value,
        )

        return InterviewSessionSummaryResponse(
            session_id=session.session_id,
            role=session.role,
            topic=session.topic,
            total_questions=session.total_questions,
            questions_answered=questions_answered,
            average_score=average_score,
            overall_rating=overall_rating,
            aggregated_strengths=aggregated_strengths,
            aggregated_improvement_areas=aggregated_improvement_areas,
            session_status=session.status,
        )

    def reset_store(self) -> None:
        """Clear all stored sessions. Used primarily in automated tests."""
        self._sessions.clear()


# Singleton instance of SessionService for the application runtime
_session_service_instance = SessionService()


def get_session_service() -> SessionService:
    """Dependency provider function for FastAPI routes."""
    return _session_service_instance
