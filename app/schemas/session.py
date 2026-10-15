from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field, field_validator

from app.schemas.interview import (
    DifficultyLevel,
    EvaluationRating,
    ExperienceLevel,
    InterviewEvaluationResponse,
    QuestionType,
)


class SessionStatus(str, Enum):
    """Lifecycle status of an interview session."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            val_clean = value.strip().lower()
            for member in cls:
                if member.value.lower() == val_clean:
                    return member
        return None


class CreateSessionRequest(BaseModel):
    """Request payload to initialize a new structured interview session."""

    role: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Target job role for the interview session",
        examples=["Frontend Developer"],
    )
    experience_level: ExperienceLevel = Field(
        ...,
        description="Candidate experience level (Fresher, Junior, Mid-Level, Senior)",
        examples=[ExperienceLevel.JUNIOR],
    )
    difficulty: DifficultyLevel = Field(
        ...,
        description="Question difficulty level (Easy, Medium, Hard)",
        examples=[DifficultyLevel.MEDIUM],
    )
    topic: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Specific interview topic or technology domain",
        examples=["React Hooks"],
    )
    question_type: QuestionType = Field(
        ...,
        description="Type of interview questions (Technical, Scenario-Based, Behavioral)",
        examples=[QuestionType.TECHNICAL],
    )
    total_questions: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Total number of interview questions configured for this session (between 1 and 20)",
        examples=[5],
    )

    @field_validator("role", "topic", mode="before")
    @classmethod
    def validate_not_blank(cls, value: Any, info) -> Any:
        """Ensure string fields are not blank or only whitespace."""
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                raise ValueError(f"'{info.field_name}' must not be empty or whitespace only.")
            return trimmed
        return value


class SubmitAnswerRequest(BaseModel):
    """Request payload for submitting a candidate answer to the current session question."""

    candidate_answer: str = Field(
        ...,
        min_length=2,
        max_length=10000,
        description="Candidate's spoken or written answer to the interview question",
        examples=[
            "useState declares reactive state in functional components. useEffect handles side effects and synchronization with dependency arrays."
        ],
    )
    question_number: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Optional question number being answered. If omitted, targets the active unanswered question.",
        examples=[1],
    )

    @field_validator("candidate_answer", mode="before")
    @classmethod
    def validate_not_blank(cls, value: Any, info) -> Any:
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                raise ValueError(f"'{info.field_name}' must not be empty or whitespace only.")
            return trimmed
        return value


class SessionQuestionItem(BaseModel):
    """Represents a single question and its associated answer and evaluation in a session."""

    question_number: int = Field(
        ...,
        ge=1,
        le=20,
        description="1-indexed sequence number of the question in this session",
        examples=[1],
    )
    question: str = Field(
        ...,
        min_length=5,
        description="The tailored interview question text",
        examples=["Explain how useState, useEffect, and useContext work in React."],
    )
    expected_answer_points: List[str] = Field(
        ...,
        min_length=1,
        description="Key technical points a strong answer should address",
        examples=[
            "useState manages local state within functional components",
            "useEffect synchronizes side effects with dependency changes",
            "useContext shares context data without prop drilling",
        ],
    )
    follow_up_question: Optional[str] = Field(
        None,
        description="Probing follow-up question related to this topic",
        examples=["How do you avoid unnecessary re-renders when passing objects through Context?"],
    )
    difficulty: DifficultyLevel = Field(
        ...,
        description="Difficulty level of the question",
        examples=[DifficultyLevel.MEDIUM],
    )
    topic: str = Field(
        ...,
        description="Technology domain or topic of the question",
        examples=["React Hooks"],
    )
    candidate_answer: Optional[str] = Field(
        None,
        description="Candidate answer if submitted, otherwise None",
        examples=[
            "useState declares reactive state in functional components. useEffect handles side effects."
        ],
    )
    evaluation: Optional[InterviewEvaluationResponse] = Field(
        None,
        description="Structured evaluation results if evaluated, otherwise None",
    )


class SessionDetailResponse(BaseModel):
    """Comprehensive details and progress of an interview session."""

    session_id: str = Field(
        ...,
        description="Unique UUID identifier for the interview session",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    role: str = Field(..., description="Target job role", examples=["Frontend Developer"])
    experience_level: ExperienceLevel = Field(..., examples=[ExperienceLevel.JUNIOR])
    difficulty: DifficultyLevel = Field(..., examples=[DifficultyLevel.MEDIUM])
    topic: str = Field(..., description="Interview topic", examples=["React Hooks"])
    question_type: QuestionType = Field(..., examples=[QuestionType.TECHNICAL])
    total_questions: int = Field(..., ge=1, le=20, description="Configured total questions", examples=[5])
    completed_questions: int = Field(..., ge=0, description="Number of completed questions", examples=[1])
    current_question_number: int = Field(
        ...,
        ge=0,
        description="Current question number (0 if no question generated yet)",
        examples=[1],
    )
    status: SessionStatus = Field(..., description="Current session lifecycle status", examples=[SessionStatus.IN_PROGRESS])
    questions: List[SessionQuestionItem] = Field(
        default_factory=list,
        description="List of questions, candidate answers, and evaluations",
    )
    created_at: str = Field(..., description="Session creation timestamp (ISO 8601 UTC)")
    updated_at: str = Field(..., description="Last update timestamp (ISO 8601 UTC)")


class SessionNextQuestionResponse(BaseModel):
    """Response returned when the next question in a session is generated."""

    session_id: str = Field(..., description="Interview session ID")
    question_number: int = Field(..., ge=1, le=20, description="Current question sequence number")
    total_questions: int = Field(..., ge=1, le=20, description="Total questions configured for the session")
    question: str = Field(..., description="The tailored interview question")
    expected_answer_points: List[str] = Field(..., description="Criteria expected in a strong answer")
    follow_up_question: Optional[str] = Field(None, description="Probing follow-up question")
    difficulty: DifficultyLevel = Field(...)
    topic: str = Field(...)


class SessionSubmitAnswerResponse(BaseModel):
    """Response returned after candidate answer submission and evaluation."""

    session_id: str = Field(..., description="Interview session ID")
    question_number: int = Field(..., ge=1, le=20, description="Question sequence number answered")
    completed_questions: int = Field(..., ge=0, description="Number of questions completed so far")
    total_questions: int = Field(..., ge=1, le=20, description="Total configured questions")
    status: SessionStatus = Field(..., description="Session lifecycle status")
    evaluation: InterviewEvaluationResponse = Field(..., description="Evaluation result for this answer")


class InterviewSessionSummaryResponse(BaseModel):
    """Final interview summary produced upon session completion."""

    session_id: str = Field(..., description="Interview session ID")
    role: str = Field(..., description="Target job role")
    topic: str = Field(..., description="Interview topic")
    total_questions: int = Field(..., ge=1, le=20, description="Total questions configured")
    questions_answered: int = Field(..., ge=0, description="Total questions actually answered and evaluated")
    average_score: float = Field(..., ge=0.0, le=100.0, description="Calculated average evaluation score (0.0 to 100.0)")
    overall_rating: EvaluationRating = Field(
        ...,
        description="Rubric rating based on the average score (Needs Improvement, Fair, Good, Excellent)",
    )
    aggregated_strengths: List[str] = Field(
        default_factory=list,
        description="Aggregated positive strengths identified across candidate answers",
    )
    aggregated_improvement_areas: List[str] = Field(
        default_factory=list,
        description="Aggregated constructive improvement areas across candidate answers",
    )
    session_status: SessionStatus = Field(
        ...,
        description="Session status upon completion ('completed')",
    )
