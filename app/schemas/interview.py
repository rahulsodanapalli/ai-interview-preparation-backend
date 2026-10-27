from enum import Enum
from typing import Any, Dict, List, Optional
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

    @field_validator("expected_answer_points", mode="before")
    @classmethod
    def coerce_expected_points(cls, value: Any) -> Any:
        """Coerce a single string into a list for frontend resilience."""
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed:
                return [trimmed]
            return []
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


# ============================================================================
# RAG Document-Grounded Question Schemas (Phase 5)
# ============================================================================


class RetrievedChunkMetadata(BaseModel):
    """Metadata describing a retrieved document chunk citation used as RAG context."""

    chunk_id: str = Field(..., description="Unique chunk identifier in Pinecone")
    chunk_index: int = Field(..., description="0-indexed position of chunk in the source document")
    similarity_score: float = Field(..., description="Cosine similarity score of the match")
    excerpt: str = Field(..., description="Safe excerpt of the retrieved chunk text")


class DocumentQuestionRequest(BaseModel):
    """Request payload for generating an interview question grounded in an uploaded document."""

    document_id: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=100,
        description="Unique document identifier returned from document upload",
        examples=["doc_a1b2c3d4e5f60718"],
    )
    role: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=100,
        description="Target job role for the interview",
        examples=["Backend Developer"],
    )
    experience_level: ExperienceLevel = Field(
        default=ExperienceLevel.MID_LEVEL,
        description="Candidate experience level (Fresher, Junior, Mid-Level, Senior)",
        examples=[ExperienceLevel.MID_LEVEL],
    )
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.MEDIUM,
        description="Question difficulty level (Easy, Medium, Hard)",
        examples=[DifficultyLevel.MEDIUM],
    )
    topic: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=100,
        description="Specific interview topic or technology domain to focus on",
        examples=["FastAPI and PostgreSQL"],
    )
    question_type: QuestionType = Field(
        default=QuestionType.TECHNICAL,
        description="Type of question (Technical, Scenario-Based, Behavioral)",
        examples=[QuestionType.TECHNICAL],
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of relevant document chunks to retrieve as context",
        examples=[4],
    )
    min_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum cosine similarity threshold for context chunk retrieval (defaults to system configured threshold if omitted)",
        examples=[0.55],
    )

    @field_validator("document_id", mode="before")
    @classmethod
    def validate_document_id_not_blank(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str):
            trimmed = v.strip()
            if not trimmed:
                raise ValueError("'document_id' must not be empty or whitespace only.")
            if "#" in trimmed:
                trimmed = trimmed.split("#")[0].strip()
                if not trimmed:
                    raise ValueError("'document_id' must not be empty or whitespace only.")
            return trimmed
        raise ValueError("'document_id' must be a valid string.")


class DocumentQuestionResponse(BaseModel):
    """Structured response containing interview question grounded in candidate document context."""

    question: str = Field(
        ...,
        min_length=5,
        description="The interview question grounded in the candidate's document background",
    )
    expected_answer_points: List[str] = Field(
        ...,
        min_length=1,
        description="Key technical criteria or concepts a strong answer should address",
    )
    follow_up_question: str = Field(
        ...,
        min_length=5,
        description="A follow-up question to probe deeper into the candidate's experience",
    )
    difficulty: DifficultyLevel = Field(
        ...,
        description="The difficulty level of the question",
    )
    topic: str = Field(
        ...,
        description="The topic or technical skill area targeted",
    )
    document_id: str = Field(
        ...,
        description="Identifier of the document from which context was retrieved",
    )
    context_grounded: bool = Field(
        default=True,
        description=(
            "Indicates whether the question is supported by relevant candidate document evidence meeting "
            "the semantic relevance threshold. When false, the question is a foundational role-based question "
            "because no relevant document evidence was available."
        ),
    )
    retrieved_chunks: List[RetrievedChunkMetadata] = Field(
        default_factory=list,
        description="Citations and similarity scores of the document chunks used as context",
    )
