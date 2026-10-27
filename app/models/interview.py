import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.base import Base


class InterviewSession(Base):
    """
    SQLAlchemy model representing an interview session.
    Persists configuration parameters, lifecycle progress, and session metadata.
    """

    __tablename__ = "interview_sessions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True,
        comment="Unique UUID session identifier",
    )
    role: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Target job role for the interview",
    )
    experience_level: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Candidate experience level (Fresher, Junior, Mid-Level, Senior)",
    )
    difficulty: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Configured difficulty level (Easy, Medium, Hard)",
    )
    topic: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Interview topic or domain",
    )
    question_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Type of questions (Technical, Scenario-Based, Behavioral)",
    )
    total_questions: Mapped[int] = mapped_column(
        Integer,
        default=5,
        nullable=False,
        comment="Total questions configured for the session",
    )
    completed_questions: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Count of completed and evaluated questions",
    )
    current_question_number: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Sequence number of the current active question",
    )
    status: Mapped[str] = mapped_column(
        String(50),
        default="in_progress",
        nullable=False,
        index=True,
        comment="Session lifecycle status ('in_progress', 'completed')",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Timestamp when session was created",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Timestamp of last update",
    )

    # Relationships
    questions: Mapped[List["InterviewQuestion"]] = relationship(
        "InterviewQuestion",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="InterviewQuestion.order",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<InterviewSession(id={self.id}, role='{self.role}', "
            f"status='{self.status}', progress={self.completed_questions}/{self.total_questions})>"
        )


class InterviewQuestion(Base):
    """
    SQLAlchemy model representing a single generated question in an interview session.
    """

    __tablename__ = "interview_questions"
    __table_args__ = (
        UniqueConstraint("session_id", "order", name="uq_session_question_order"),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True,
        comment="Unique UUID question identifier",
    )
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to the parent interview session",
    )
    order: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="1-indexed sequence order of the question within the session",
    )
    question: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="The generated interview question text",
    )
    topic: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="Topic of the question",
    )
    difficulty: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Difficulty of the question",
    )
    question_type: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Question category type",
    )
    expected_answer_points: Mapped[List[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="Expected criteria points for evaluating answers",
    )
    follow_up_question: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Probing follow-up question",
    )
    candidate_answer: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Submitted candidate answer text",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Timestamp when question was generated",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Timestamp when question was last updated",
    )

    # Relationships
    session: Mapped["InterviewSession"] = relationship(
        "InterviewSession",
        back_populates="questions",
    )
    evaluation: Mapped[Optional["AnswerEvaluation"]] = relationship(
        "AnswerEvaluation",
        back_populates="question",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<InterviewQuestion(id={self.id}, session_id={self.session_id}, order={self.order})>"


class AnswerEvaluation(Base):
    """
    SQLAlchemy model representing the qualitative feedback and rubric-aligned scoring
    for a candidate's answer to a specific interview question.
    """

    __tablename__ = "answer_evaluations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True,
        comment="Unique UUID evaluation identifier",
    )
    question_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("interview_questions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
        comment="Foreign key referencing the evaluated question",
    )
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key referencing the parent interview session",
    )
    candidate_answer: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Snapshot of candidate answer evaluated",
    )
    score: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Evaluation score from 0 to 100",
    )
    rating: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Categorical rating (Needs Improvement, Fair, Good, Excellent)",
    )
    correct_points: Mapped[List[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="Concepts the candidate answered correctly",
    )
    missing_points: Mapped[List[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="Expected concepts omitted or poorly addressed",
    )
    strengths: Mapped[List[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="Strengths observed in candidate's response",
    )
    improvements: Mapped[List[str]] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="Actionable suggestions for improvement",
    )
    sample_answer: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Concise, high-quality reference answer",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Timestamp when evaluation was generated and recorded",
    )

    # Relationships
    question: Mapped["InterviewQuestion"] = relationship(
        "InterviewQuestion",
        back_populates="evaluation",
    )
    session: Mapped["InterviewSession"] = relationship(
        "InterviewSession",
    )

    def __repr__(self) -> str:
        return f"<AnswerEvaluation(id={self.id}, question_id={self.question_id}, score={self.score}, rating='{self.rating}')>"
