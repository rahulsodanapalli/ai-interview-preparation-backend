"""
Interview Session Management Service with Database Persistence.

This service coordinates interview lifecycles, question sequencing, answer evaluation
tracking, and performance summaries. It persists state to PostgreSQL when a database
session is provided, while maintaining in-memory caching and backward compatibility
for unit tests.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload


from app.models.interview import AnswerEvaluation, InterviewQuestion, InterviewSession
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

logger = logging.getLogger(__name__)


class SessionNotFoundError(Exception):
    """Raised when an interview session ID is not found in the store or database."""
    pass


class SessionStateError(Exception):
    """Raised when an invalid session lifecycle transition or state rule is violated."""
    pass


def _db_model_to_session_detail(session_db: InterviewSession) -> SessionDetailResponse:
    """Helper to convert an InterviewSession database model to a SessionDetailResponse schema."""
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
        status=SessionStatus(session_db.status),
        questions=questions_list,
        created_at=session_db.created_at.isoformat() if session_db.created_at else datetime.now(timezone.utc).isoformat(),
        updated_at=session_db.updated_at.isoformat() if session_db.updated_at else datetime.now(timezone.utc).isoformat(),
    )


class SessionService:
    """
    Session manager handling interview lifecycles, question sequencing,
    answer evaluation tracking, and session summaries with PostgreSQL persistence.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionDetailResponse] = {}

    def create_session(
        self,
        payload: CreateSessionRequest,
        db: Optional[Session] = None,
    ) -> SessionDetailResponse:
        """
        Create and initialize a new interview session.
        Persists to database when db is provided; updates in-memory cache.
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

        if db is not None:
            session_db = InterviewSession(
                id=session_id,
                role=payload.role,
                experience_level=(
                    payload.experience_level.value
                    if hasattr(payload.experience_level, "value")
                    else str(payload.experience_level)
                ),
                difficulty=(
                    payload.difficulty.value
                    if hasattr(payload.difficulty, "value")
                    else str(payload.difficulty)
                ),
                topic=payload.topic,
                question_type=(
                    payload.question_type.value
                    if hasattr(payload.question_type, "value")
                    else str(payload.question_type)
                ),
                total_questions=payload.total_questions,
                completed_questions=0,
                current_question_number=0,
                status=SessionStatus.IN_PROGRESS.value,
            )
            db.add(session_db)
            db.flush()
            logger.info("Persisted session %s to database.", session_id)

        logger.info("Created interview session %s for role '%s' on topic '%s'.", session_id, payload.role, payload.topic)
        return session

    def get_session(
        self,
        session_id: str,
        db: Optional[Session] = None,
    ) -> SessionDetailResponse:
        """
        Retrieve session details and current progress by session ID.
        Checks database first if db is provided, falling back to in-memory store.
        """
        if db is not None:
            session_db = db.get(InterviewSession, session_id)
            if session_db is not None:
                stmt = (
                    select(InterviewQuestion)
                    .options(selectinload(InterviewQuestion.evaluation))
                    .where(InterviewQuestion.session_id == session_id)
                    .order_by(InterviewQuestion.order)
                )
                questions_db = list(db.execute(stmt).scalars().all())
                session_db.questions = questions_db
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
        """
        Retrieve list of interview sessions ordered by creation date descending.
        Retrieves from database when db is provided; otherwise returns in-memory sessions.
        """
        if db is not None:
            stmt = (
                select(InterviewSession)
                .order_by(InterviewSession.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
            sessions_db = list(db.execute(stmt).scalars().all())
            result: List[SessionDetailResponse] = []
            for s_db in sessions_db:
                stmt_q = (
                    select(InterviewQuestion)
                    .options(selectinload(InterviewQuestion.evaluation))
                    .where(InterviewQuestion.session_id == s_db.id)
                    .order_by(InterviewQuestion.order)
                )
                s_db.questions = list(db.execute(stmt_q).scalars().all())
                result.append(_db_model_to_session_detail(s_db))
            return result

        all_sessions = list(self._sessions.values())
        return all_sessions[skip : skip + limit]

    def validate_can_generate_question(
        self,
        session_id: str,
        db: Optional[Session] = None,
    ) -> SessionDetailResponse:
        """
        Validate that the session is in a valid state to generate the next question.
        """
        session = self.get_session(session_id, db=db)

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
        db: Optional[Session] = None,
    ) -> SessionNextQuestionResponse:
        """
        Append a generated question to the session and advance the question number.
        Persists question to database when db is provided.
        """
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
            session_db = db.get(InterviewSession, session_id)
            question_db = InterviewQuestion(
                id=str(uuid.uuid4()),
                session_id=session_id,
                order=next_question_number,
                question=question_data.question,
                topic=question_data.topic,
                difficulty=(
                    question_data.difficulty.value
                    if hasattr(question_data.difficulty, "value")
                    else str(question_data.difficulty)
                ),
                question_type=(
                    session.question_type.value
                    if hasattr(session.question_type, "value")
                    else str(session.question_type)
                ),
                expected_answer_points=question_data.expected_answer_points,
                follow_up_question=question_data.follow_up_question,
                candidate_answer=None,
                session=session_db,
            )
            db.add(question_db)

            if session_db:
                session_db.current_question_number = next_question_number
            db.flush()
            logger.info("Persisted question #%d to database for session %s.", next_question_number, session_id)


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
        """
        Validate that a candidate answer can be submitted for the specified or current question.
        """
        session = self.get_session(session_id, db=db)

        if session.status == SessionStatus.COMPLETED:
            raise SessionStateError("Cannot submit answer: interview session is completed.")

        if not session.questions:
            raise SessionStateError("No questions have been generated for this session yet. Request a question first.")

        if question_number is not None:
            matching = [q for q in session.questions if q.question_number == question_number]
            if not matching:
                raise SessionStateError(f"Question #{question_number} does not exist in this session.")
            target = matching[0]
            if target.candidate_answer is not None:
                raise SessionStateError(f"Question #{question_number} has already been answered. Duplicate submissions are not allowed.")
            return target
        else:
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
        """
        Record candidate answer and AI evaluation on the target question and update progress.
        Persists evaluation to database when db is provided.
        """
        session = self.get_session(session_id, db=db)
        matching = [q for q in session.questions if q.question_number == question_number]
        if not matching:
            raise SessionStateError(f"Question #{question_number} does not exist in this session.")

        target = matching[0]
        target.candidate_answer = candidate_answer
        target.evaluation = evaluation

        # Update completed questions count
        session.completed_questions = sum(1 for q in session.questions if q.evaluation is not None)
        session.updated_at = datetime.now(timezone.utc).isoformat()

        if db is not None:
            stmt = select(InterviewQuestion).where(
                InterviewQuestion.session_id == session_id,
                InterviewQuestion.order == question_number,
            )
            question_db = db.execute(stmt).scalars().first()
            if question_db:
                question_db.candidate_answer = candidate_answer

                rating_val = (
                    evaluation.rating.value
                    if hasattr(evaluation.rating, "value")
                    else str(evaluation.rating)
                )

                if question_db.evaluation is not None:
                    # Update existing evaluation to avoid duplicate records
                    question_db.evaluation.candidate_answer = candidate_answer
                    question_db.evaluation.score = evaluation.overall_score
                    question_db.evaluation.rating = rating_val
                    question_db.evaluation.correct_points = evaluation.correct_points
                    question_db.evaluation.missing_points = evaluation.missing_points
                    question_db.evaluation.strengths = evaluation.strengths
                    question_db.evaluation.improvements = evaluation.improvements
                    question_db.evaluation.sample_answer = evaluation.sample_answer
                else:
                    eval_db = AnswerEvaluation(
                        id=str(uuid.uuid4()),
                        question_id=question_db.id,
                        session_id=session_id,
                        candidate_answer=candidate_answer,
                        score=evaluation.overall_score,
                        rating=rating_val,
                        correct_points=evaluation.correct_points,
                        missing_points=evaluation.missing_points,
                        strengths=evaluation.strengths,
                        improvements=evaluation.improvements,
                        sample_answer=evaluation.sample_answer,
                        question=question_db,
                    )
                    db.add(eval_db)
                    question_db.evaluation = eval_db

            session_db = db.get(InterviewSession, session_id)

            if session_db:
                session_db.completed_questions = session.completed_questions
            db.flush()
            logger.info("Persisted answer evaluation to database for question #%d in session %s.", question_number, session_id)

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
        """
        Mark the session as completed and calculate the final performance summary.
        Persists status update to database when db is provided.
        """
        session = self.get_session(session_id, db=db)

        session.status = SessionStatus.COMPLETED
        session.updated_at = datetime.now(timezone.utc).isoformat()

        if db is not None:
            session_db = db.get(InterviewSession, session_id)
            if session_db:
                session_db.status = SessionStatus.COMPLETED.value
            db.flush()
            logger.info("Persisted session completion status to database for session %s.", session_id)

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

            aggregated_strengths = []
            for q in evaluated_questions:
                for s in q.evaluation.strengths:
                    if s not in aggregated_strengths:
                        aggregated_strengths.append(s)

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
        """Clear all in-memory stored sessions. Used in automated tests."""
        self._sessions.clear()


# Singleton instance of SessionService for the application runtime
_session_service_instance = SessionService()


def get_session_service() -> SessionService:
    """Dependency provider function for FastAPI routes."""
    return _session_service_instance
