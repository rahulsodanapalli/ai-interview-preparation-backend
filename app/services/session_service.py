"""Interview Session Management Service with Database Persistence.

This service coordinates interview lifecycles, question sequencing, answer evaluation
tracking, and performance summaries. It delegates relational persistence to
InterviewRepository when a database session is provided, while maintaining
in-memory caching for fast lookups and offline/unit-test compatibility.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional
import uuid

from sqlalchemy.orm import Session

from app.models.interview import InterviewSession
from app.repositories.interview_repository import InterviewRepository
from app.schemas.interview import (
    DifficultyLevel,
    EvaluationRating,
    ExperienceLevel,
    InterviewEvaluationResponse,
    InterviewQuestionResponse,
    QuestionType,
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
from app.utils.identifiers import enum_to_value

logger = logging.getLogger(__name__)


class SessionNotFoundError(Exception):
    """Raised when an interview session ID is not found in the store or database."""


class SessionStateError(Exception):
    """Raised when an invalid session lifecycle transition or state rule is violated."""


def _db_model_to_session_detail(session_db: InterviewSession) -> SessionDetailResponse:
    """Convert an InterviewSession ORM model into a SessionDetailResponse schema."""
    questions_list: List[SessionQuestionItem] = []
    sorted_questions = sorted(session_db.questions, key=lambda q: q.order)

    for q in sorted_questions:
        eval_resp = None
        if q.evaluation:
            eval_resp = InterviewEvaluationResponse(
                overall_score=q.evaluation.score,
                rating=EvaluationRating(q.evaluation.rating),
                correct_points=q.evaluation.correct_points or [],
                missing_points=q.evaluation.missing_points or [],
                strengths=q.evaluation.strengths or [],
                improvements=q.evaluation.improvements or [],
                sample_answer=q.evaluation.sample_answer,
            )
        questions_list.append(
            SessionQuestionItem(
                question_number=q.order,
                question=q.question,
                expected_answer_points=q.expected_answer_points or [],
                follow_up_question=q.follow_up_question,
                difficulty=DifficultyLevel(q.difficulty),
                topic=q.topic,
                candidate_answer=q.candidate_answer,
                evaluation=eval_resp,
            )
        )

    now_iso = datetime.now(timezone.utc).isoformat()
    return SessionDetailResponse(
        session_id=session_db.id,
        role=session_db.role,
        experience_level=ExperienceLevel(session_db.experience_level),
        difficulty=DifficultyLevel(session_db.difficulty),
        topic=session_db.topic,
        question_type=QuestionType(session_db.question_type),
        total_questions=session_db.total_questions,
        completed_questions=session_db.completed_questions,
        current_question_number=session_db.current_question_number,
        source_id=getattr(session_db, "source_id", None),
        status=SessionStatus(session_db.status),
        questions=questions_list,
        created_at=session_db.created_at.isoformat() if session_db.created_at else now_iso,
        updated_at=session_db.updated_at.isoformat() if session_db.updated_at else now_iso,
    )


class SessionService:
    """Session manager handling interview lifecycles, question sequencing, and summaries."""

    DUPLICATE_WINDOW_SECONDS = 10.0

    def __init__(self, repository: Optional[InterviewRepository] = None) -> None:
        self._sessions: Dict[str, SessionDetailResponse] = {}
        self._repository = repository or InterviewRepository()

    def _find_recent_duplicate_session(
        self,
        payload: CreateSessionRequest,
    ) -> Optional[SessionDetailResponse]:
        """Return an existing in-progress source session if created within the dedup window."""
        if not payload.source_id:
            return None

        now_dt = datetime.now(timezone.utc)
        for existing in self._sessions.values():
            if (
                existing.source_id == payload.source_id
                and existing.role == payload.role
                and existing.topic == payload.topic
                and existing.status == SessionStatus.IN_PROGRESS
            ):
                try:
                    created_dt = datetime.fromisoformat(existing.created_at)
                    if (now_dt - created_dt).total_seconds() < self.DUPLICATE_WINDOW_SECONDS:
                        logger.info(
                            "Duplicate session creation detected for source '%s'; returning existing session '%s'.",
                            payload.source_id,
                            existing.session_id,
                        )
                        return existing
                except ValueError:
                    continue
        return None

    @staticmethod
    def _build_initial_questions(payload: CreateSessionRequest) -> List[SessionQuestionItem]:
        """Build SessionQuestionItem objects from optional pre-generated questions."""
        if not payload.questions:
            return []
        return [
            SessionQuestionItem(
                question_number=idx,
                question=q_input.question,
                expected_answer_points=q_input.expected_answer_points,
                follow_up_question=q_input.follow_up_question,
                difficulty=q_input.difficulty or payload.difficulty,
                topic=q_input.topic or payload.topic,
                candidate_answer=None,
                evaluation=None,
            )
            for idx, q_input in enumerate(payload.questions, start=1)
        ]

    def create_session(
        self,
        payload: CreateSessionRequest,
        db: Optional[Session] = None,
    ) -> SessionDetailResponse:
        """Create and initialize a new interview session with optional pre-generated questions."""
        duplicate = self._find_recent_duplicate_session(payload)
        if duplicate is not None:
            return duplicate

        session_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        questions_list = self._build_initial_questions(payload)
        total_questions = (
            max(len(questions_list), payload.total_questions)
            if questions_list
            else payload.total_questions
        )
        current_question_number = 1 if questions_list else 0

        session = SessionDetailResponse(
            session_id=session_id,
            role=payload.role,
            experience_level=payload.experience_level,
            difficulty=payload.difficulty,
            topic=payload.topic,
            question_type=payload.question_type,
            total_questions=total_questions,
            completed_questions=0,
            current_question_number=current_question_number,
            source_id=payload.source_id,
            status=SessionStatus.IN_PROGRESS,
            questions=questions_list,
            created_at=now,
            updated_at=now,
        )
        self._sessions[session_id] = session

        if db is not None:
            self._repository.create_session(
                db,
                session_id=session_id,
                payload=payload,
                total_questions=total_questions,
                current_question_number=current_question_number,
                questions_list=questions_list,
            )

        logger.info(
            "Created interview session %s for role '%s' on topic '%s' (source_id=%s).",
            session_id,
            payload.role,
            payload.topic,
            payload.source_id,
        )
        return session

    def get_session(
        self,
        session_id: str,
        db: Optional[Session] = None,
    ) -> SessionDetailResponse:
        """Retrieve session details and current progress by session ID."""
        if db is not None:
            session_db = self._repository.get_session_by_id(db, session_id)
            if session_db is not None:
                session_resp = _db_model_to_session_detail(session_db)
                self._sessions[session_id] = session_resp
                return session_resp

        session = self._sessions.get(session_id)
        if not session:
            logger.warning("Session lookup failed: session '%s' not found.", session_id)
            raise SessionNotFoundError(f"Interview session '{session_id}' not found.")
        return session

    def list_sessions(
        self,
        db: Optional[Session] = None,
        limit: int = 50,
        skip: int = 0,
    ) -> List[SessionDetailResponse]:
        """Retrieve list of interview sessions ordered by creation date descending."""
        if db is not None:
            sessions_db = self._repository.list_sessions(db, limit=limit, skip=skip)
            if sessions_db is not None:
                return [_db_model_to_session_detail(s) for s in sessions_db]

        all_sessions = list(self._sessions.values())
        return all_sessions[skip : skip + limit]

    def validate_can_generate_question(
        self,
        session_id: str,
        db: Optional[Session] = None,
    ) -> SessionDetailResponse:
        """Validate that the session is in a valid state to generate the next question."""
        session = self.get_session(session_id, db=db)

        if session.status == SessionStatus.COMPLETED:
            raise SessionStateError("Cannot generate next question: interview session is already completed.")

        unanswered = [q for q in session.questions if q.candidate_answer is None]
        if unanswered:
            current_num = unanswered[0].question_number
            raise SessionStateError(
                f"Current question #{current_num} has not been answered yet. "
                "Submit an answer before requesting the next question."
            )

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
        db: Optional[Session] = None,
    ) -> SessionNextQuestionResponse:
        """Append a generated question to the session and advance the question number."""
        session = self.get_session(session_id, db=db)
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

        if db is not None:
            self._repository.add_question(
                db,
                session_id=session_id,
                question_number=next_question_number,
                question_type=enum_to_value(session.question_type),
                question_data=question_data,
            )

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
        db: Optional[Session] = None,
    ) -> SessionQuestionItem:
        """Validate that a candidate answer can be submitted for the specified or current question."""
        session = self.get_session(session_id, db=db)

        if session.status == SessionStatus.COMPLETED:
            raise SessionStateError("Cannot submit answer: interview session is completed.")

        if not session.questions:
            raise SessionStateError(
                "No questions have been generated for this session yet. Request a question first."
            )

        if question_number is not None:
            matching = [q for q in session.questions if q.question_number == question_number]
            if not matching:
                raise SessionStateError(f"Question #{question_number} does not exist in this session.")
            target = matching[0]
            if target.candidate_answer is not None:
                raise SessionStateError(
                    f"Question #{question_number} has already been answered. Duplicate submissions are not allowed."
                )
            return target

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
        db: Optional[Session] = None,
    ) -> SessionSubmitAnswerResponse:
        """Record candidate answer and AI evaluation on the target question and update progress."""
        session = self.get_session(session_id, db=db)
        matching = [q for q in session.questions if q.question_number == question_number]
        if not matching:
            raise SessionStateError(f"Question #{question_number} does not exist in this session.")

        target = matching[0]
        target.candidate_answer = candidate_answer
        target.evaluation = evaluation

        session.completed_questions = sum(1 for q in session.questions if q.evaluation is not None)
        session.updated_at = datetime.now(timezone.utc).isoformat()

        if db is not None:
            self._repository.record_answer_evaluation(
                db,
                session_id=session_id,
                question_number=question_number,
                candidate_answer=candidate_answer,
                evaluation=evaluation,
                completed_questions=session.completed_questions,
            )

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

    def complete_session(
        self,
        session_id: str,
        db: Optional[Session] = None,
    ) -> InterviewSessionSummaryResponse:
        """Mark the session as completed and calculate the final performance summary."""
        session = self.get_session(session_id, db=db)

        session.status = SessionStatus.COMPLETED
        session.updated_at = datetime.now(timezone.utc).isoformat()

        if db is not None:
            self._repository.update_status(
                db,
                session_id=session_id,
                status_value=SessionStatus.COMPLETED.value,
            )

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

            aggregated_strengths = list(
                dict.fromkeys(s for q in evaluated_questions for s in q.evaluation.strengths)
            )
            aggregated_improvement_areas = list(
                dict.fromkeys(imp for q in evaluated_questions for imp in q.evaluation.improvements)
            )

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
        """Clear all in-memory stored sessions. Used in automated tests."""
        self._sessions.clear()


_session_service_instance = SessionService()


def get_session_service() -> SessionService:
    """Dependency provider function for FastAPI routes."""
    return _session_service_instance
