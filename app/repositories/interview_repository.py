"""Repository for persisting and querying InterviewSession, InterviewQuestion, and AnswerEvaluation."""

from datetime import datetime, timezone
import logging
from typing import List, Optional
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.models.interview import AnswerEvaluation, InterviewQuestion, InterviewSession
from app.schemas.interview import InterviewEvaluationResponse, InterviewQuestionResponse
from app.schemas.session import CreateSessionRequest, SessionQuestionItem, SessionStatus
from app.utils.identifiers import enum_to_value

logger = logging.getLogger(__name__)


def _safe_rollback(db: Session) -> None:
    """Attempt a rollback on a failed database operation without masking the original error."""
    try:
        db.rollback()
    except Exception as exc:
        logger.error("Failed to roll back database session: %s", type(exc).__name__)


class InterviewRepository:
    """Encapsulates all SQLAlchemy CRUD operations for interview sessions and evaluations."""

    def find_recent_duplicate_session(
        self,
        db: Session,
        *,
        source_id: str,
        role: str,
        topic: str,
        window_seconds: float,
        user_id: Optional[str] = None,
    ) -> Optional[InterviewSession]:
        """Return a recent in-progress session for the same user/source/role/topic within the dedup window."""
        try:
            conditions = [
                InterviewSession.source_id == source_id,
                InterviewSession.role == role,
                InterviewSession.topic == topic,
                InterviewSession.status == SessionStatus.IN_PROGRESS.value,
            ]
            if user_id is not None:
                conditions.append(InterviewSession.user_id == user_id)

            stmt = (
                select(InterviewSession)
                .options(
                    selectinload(InterviewSession.questions).selectinload(
                        InterviewQuestion.evaluation
                    )
                )
                .where(*conditions)
                .order_by(InterviewSession.created_at.desc())
                .limit(1)
            )
            existing = db.execute(stmt).scalars().first()
            if existing is None or existing.created_at is None:
                return None

            created_dt = existing.created_at
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            now_dt = datetime.now(timezone.utc)
            if (now_dt - created_dt).total_seconds() < window_seconds:
                return existing
            return None
        except Exception as exc:
            logger.error(
                "Database query failed while checking duplicate session for source '%s': %s",
                source_id,
                type(exc).__name__,
            )
            raise

    def create_session(
        self,
        db: Session,
        *,
        session_id: str,
        payload: CreateSessionRequest,
        total_questions: int,
        current_question_number: int,
        questions_list: List[SessionQuestionItem],
        user_id: Optional[str] = None,
    ) -> None:
        """Persist a new InterviewSession and any pre-generated questions."""
        try:
            session_db = InterviewSession(
                id=session_id,
                user_id=user_id,
                role=payload.role,
                experience_level=enum_to_value(payload.experience_level),
                difficulty=enum_to_value(payload.difficulty),
                topic=payload.topic,
                question_type=enum_to_value(payload.question_type),
                total_questions=total_questions,
                completed_questions=0,
                current_question_number=current_question_number,
                source_id=payload.source_id,
                status=SessionStatus.IN_PROGRESS.value,
            )
            db.add(session_db)
            db.flush()

            for q_item in questions_list:
                q_db = InterviewQuestion(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    order=q_item.question_number,
                    question=q_item.question,
                    topic=q_item.topic,
                    difficulty=enum_to_value(q_item.difficulty),
                    question_type=enum_to_value(payload.question_type),
                    expected_answer_points=q_item.expected_answer_points,
                    follow_up_question=q_item.follow_up_question,
                    candidate_answer=None,
                )
                db.add(q_db)
            db.flush()
            logger.info(
                "Persisted session %s (with %d questions) to database.",
                session_id,
                len(questions_list),
            )
        except Exception as exc:
            _safe_rollback(db)
            logger.error(
                "Failed to persist session '%s' to database: %s",
                session_id,
                type(exc).__name__,
            )
            raise

    def get_session_by_id(self, db: Session, session_id: str) -> Optional[InterviewSession]:
        """Fetch an InterviewSession with eagerly loaded questions and evaluations."""
        try:
            stmt = (
                select(InterviewSession)
                .options(
                    selectinload(InterviewSession.questions).selectinload(
                        InterviewQuestion.evaluation
                    )
                )
                .where(InterviewSession.id == session_id)
            )
            return db.execute(stmt).scalars().first()
        except Exception as exc:
            logger.error(
                "Failed to retrieve session '%s' from database: %s",
                session_id,
                type(exc).__name__,
            )
            raise

    def list_sessions(
        self,
        db: Session,
        *,
        user_id: Optional[str] = None,
        limit: int = 50,
        skip: int = 0,
    ) -> List[InterviewSession]:
        """List InterviewSessions ordered by creation time descending, scoped by user_id when provided."""
        try:
            stmt = select(InterviewSession).options(
                selectinload(InterviewSession.questions).selectinload(
                    InterviewQuestion.evaluation
                )
            )
            if user_id is not None:
                stmt = stmt.where(InterviewSession.user_id == user_id)
            stmt = stmt.order_by(InterviewSession.created_at.desc()).offset(skip).limit(limit)
            return list(db.execute(stmt).scalars().all())
        except Exception as exc:
            logger.error(
                "Failed to list sessions from database: %s",
                type(exc).__name__,
            )
            raise

    def add_question(
        self,
        db: Session,
        *,
        session_id: str,
        question_number: int,
        question_type: str,
        question_data: InterviewQuestionResponse,
    ) -> None:
        """Persist a newly generated question and advance the session's current_question_number."""
        try:
            session_db = db.get(InterviewSession, session_id)
            question_db = InterviewQuestion(
                id=str(uuid.uuid4()),
                session_id=session_id,
                order=question_number,
                question=question_data.question,
                topic=question_data.topic,
                difficulty=enum_to_value(question_data.difficulty),
                question_type=question_type,
                expected_answer_points=question_data.expected_answer_points,
                follow_up_question=question_data.follow_up_question,
                candidate_answer=None,
                session=session_db,
            )
            db.add(question_db)
            if session_db:
                session_db.current_question_number = question_number
            db.flush()
            logger.info(
                "Persisted question #%d to database for session %s.",
                question_number,
                session_id,
            )
        except Exception as exc:
            _safe_rollback(db)
            logger.error(
                "Failed to persist question #%d to database for session %s: %s",
                question_number,
                session_id,
                type(exc).__name__,
            )
            raise

    def record_answer_evaluation(
        self,
        db: Session,
        *,
        session_id: str,
        question_number: int,
        candidate_answer: str,
        evaluation: InterviewEvaluationResponse,
        completed_questions: int,
    ) -> None:
        """Persist a candidate answer and evaluation, updating session progress."""
        try:
            stmt = select(InterviewQuestion).where(
                InterviewQuestion.session_id == session_id,
                InterviewQuestion.order == question_number,
            )
            question_db = db.execute(stmt).scalars().first()
            if question_db:
                question_db.candidate_answer = candidate_answer
                rating_val = enum_to_value(evaluation.rating)

                if question_db.evaluation is not None:
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
                session_db.completed_questions = completed_questions
            db.flush()
            logger.info(
                "Persisted answer evaluation to database for question #%d in session %s.",
                question_number,
                session_id,
            )
        except Exception as exc:
            _safe_rollback(db)
            logger.error(
                "Failed to persist answer evaluation to database for session %s, question #%d: %s",
                session_id,
                question_number,
                type(exc).__name__,
            )
            raise

    def update_status(self, db: Session, *, session_id: str, status_value: str) -> None:
        """Update the lifecycle status of an InterviewSession."""
        try:
            session_db = db.get(InterviewSession, session_id)
            if session_db:
                session_db.status = status_value
            db.flush()
            logger.info(
                "Persisted session completion status to database for session %s.",
                session_id,
            )
        except Exception as exc:
            _safe_rollback(db)
            logger.error(
                "Failed to persist session completion status to database for session %s: %s",
                session_id,
                type(exc).__name__,
            )
            raise

