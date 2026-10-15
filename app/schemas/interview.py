from enum import Enum
from typing import Any, List
from pydantic import BaseModel, Field, field_validator, model_validator


class ExperienceLevel(str, Enum):
    """Candidate experience level options."""

    FRESHER = "Fresher"
    JUNIOR = "Junior"
    MID_LEVEL = "Mid-Level"
    SENIOR = "Senior"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            val_clean = value.strip().lower()
            for member in cls:
                if member.value.lower() == val_clean:
                    return member
        return None


class DifficultyLevel(str, Enum):
    """Question difficulty level options."""

    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            val_clean = value.strip().lower()
            for member in cls:
                if member.value.lower() == val_clean:
                    return member
        return None


class QuestionType(str, Enum):
    """Type of interview question requested."""

    TECHNICAL = "Technical"
    SCENARIO_BASED = "Scenario-Based"
    BEHAVIORAL = "Behavioral"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            val_clean = value.strip().lower()
            for member in cls:
                if member.value.lower() == val_clean:
                    return member
        return None


class EvaluationRating(str, Enum):
    """Categorical performance rating based on overall score."""

    NEEDS_IMPROVEMENT = "Needs Improvement"
    FAIR = "Fair"
    GOOD = "Good"
    EXCELLENT = "Excellent"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            val_clean = value.strip().lower()
            for member in cls:
                if member.value.lower() == val_clean:
                    return member
        return None

    @classmethod
    def from_score(cls, score: int) -> "EvaluationRating":
        """
        Return the corresponding evaluation rating based on the official rubric:
        - 0-39: Needs Improvement
        - 40-59: Fair
        - 60-79: Good
        - 80-100: Excellent
        """
        if score >= 80:
            return cls.EXCELLENT
        elif score >= 60:
            return cls.GOOD
        elif score >= 40:
            return cls.FAIR
        else:
            return cls.NEEDS_IMPROVEMENT


class InterviewQuestionRequest(BaseModel):
    """Validated request payload for generating a tailored interview question."""

    role: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Target job role for the interview",
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
        description="Type of question (Technical, Scenario-Based, Behavioral)",
        examples=[QuestionType.TECHNICAL],
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


class InterviewQuestionResponse(BaseModel):
    """Structured response containing interview question and evaluation criteria."""

    question: str = Field(
        ...,
        min_length=5,
        description="The tailored interview question",
        examples=["Explain how useState, useEffect, and useContext work in React."],
    )
    expected_answer_points: List[str] = Field(
        ...,
        min_length=1,
        description="Key points that a strong answer should cover",
        examples=[
            "Understanding of useState for managing state in functional components",
            "Handling side effects and dependency arrays using useEffect",
            "Using useContext to consume global state without prop drilling",
        ],
    )
    follow_up_question: str = Field(
        ...,
        min_length=5,
        description="A related follow-up question to probe deeper into candidate knowledge",
        examples=["How would you handle race conditions when fetching data inside useEffect?"],
    )
    difficulty: DifficultyLevel = Field(
        ...,
        description="The difficulty level of the question",
        examples=[DifficultyLevel.MEDIUM],
    )
    topic: str = Field(
        ...,
        min_length=1,
        description="The requested topic",
        examples=["React Hooks"],
    )

    @field_validator("expected_answer_points")
    @classmethod
    def validate_points(cls, points: List[str]) -> List[str]:
        cleaned = [p.strip() for p in points if isinstance(p, str) and p.strip()]
        if not cleaned:
            raise ValueError("expected_answer_points must contain at least one non-empty string point.")
        return cleaned


class InterviewEvaluationRequest(BaseModel):
    """Request payload for evaluating a candidate's answer against expected criteria."""

    question: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="The interview question that was asked",
        examples=["Explain how useState, useEffect, and useContext work in React."],
    )
    expected_answer_points: List[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of key points expected in a strong answer",
        examples=[
            "useState manages local state in functional components",
            "useEffect handles side effects and synchronization with dependencies",
            "useContext allows sharing global/context data without prop drilling",
        ],
    )
    candidate_answer: str = Field(
        ...,
        min_length=2,
        max_length=10000,
        description="The candidate's spoken or written answer to the interview question",
        examples=[
            "useState creates state in components. useEffect runs side effects when dependencies change. useContext consumes data from React Context."
        ],
    )

    @field_validator("question", "candidate_answer", mode="before")
    @classmethod
    def validate_not_blank(cls, value: Any, info) -> Any:
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                raise ValueError(f"'{info.field_name}' must not be empty or whitespace only.")
            return trimmed
        return value

    @field_validator("expected_answer_points")
    @classmethod
    def validate_expected_points(cls, points: List[str]) -> List[str]:
        cleaned = [p.strip() for p in points if isinstance(p, str) and p.strip()]
        if not cleaned:
            raise ValueError("expected_answer_points must contain at least one non-empty string.")
        return cleaned


class InterviewEvaluationResponse(BaseModel):
    """Structured response containing qualitative feedback and rubric-aligned scoring."""

    overall_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Overall evaluation score from 0 to 100",
        examples=[85],
    )
    rating: EvaluationRating = Field(
        ...,
        description="Categorical rating based on score: Needs Improvement (0-39), Fair (40-59), Good (60-79), Excellent (80-100)",
        examples=[EvaluationRating.EXCELLENT],
    )
    correct_points: List[str] = Field(
        default_factory=list,
        description="Key concepts the candidate explained correctly",
        examples=["Correctly explained useState for component-level state"],
    )
    missing_points: List[str] = Field(
        default_factory=list,
        description="Important expected concepts not adequately covered or omitted",
        examples=["Did not mention cleanup functions or dependency array nuances in useEffect"],
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Positive observations regarding clarity, precision, and accuracy",
        examples=["Clear, succinct explanations of hook primitives"],
    )
    improvements: List[str] = Field(
        default_factory=list,
        description="Actionable, constructive recommendations for strengthening the response",
        examples=["Provide a concrete code or architectural scenario when using useContext"],
    )
    sample_answer: str = Field(
        ...,
        min_length=5,
        description="A concise, high-quality reference answer addressing all expected points",
        examples=[
            "useState allows functional components to maintain state across renders. useEffect enables managing lifecycle side effects with dependency tracking. useContext provides a mechanism to consume scoped context without prop-drilling."
        ],
    )

    @field_validator("correct_points", "missing_points", "strengths", "improvements", mode="before")
    @classmethod
    def clean_string_lists(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return value

    @model_validator(mode="after")
    def sync_rating_with_rubric(self) -> "InterviewEvaluationResponse":
        """
        Normalize the rating to strictly match the scoring rubric:
        - 0-39: Needs Improvement
        - 40-59: Fair
        - 60-79: Good
        - 80-100: Excellent
        """
        expected_rating = EvaluationRating.from_score(self.overall_score)
        if self.rating != expected_rating:
            self.rating = expected_rating
        return self
