"""Pydantic schemas for RAG Evaluation metrics, audit results, and benchmark scoring."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvaluationRating(str, Enum):
    """Quality rating band for evaluated RAG outputs."""

    EXCELLENT = "Excellent"
    GOOD = "Good"
    ACCEPTABLE = "Acceptable"
    NEEDS_IMPROVEMENT = "Needs Improvement"


class MetricScore(BaseModel):
    """Individual evaluation metric score with explanation."""

    metric_name: str = Field(..., description="Name of the evaluated metric (e.g. Faithfulness, Context Relevance)")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized score between 0.0 and 1.0")
    reasoning: str = Field(..., description="Justification and analysis from the evaluator")
    passed: bool = Field(..., description="Whether the metric exceeds the acceptable quality threshold")


class RAGEvaluationRequest(BaseModel):
    """Request payload to evaluate a generated question against retrieved context chunks."""

    query: str = Field(..., min_length=2, description="Retrieval query or interview topic")
    question: str = Field(..., min_length=5, description="Generated interview question")
    expected_answer_points: List[str] = Field(default_factory=list, description="Key technical answer points")
    context_chunks: List[str] = Field(..., min_length=1, description="Retrieved candidate document chunk excerpts")
    role: Optional[str] = Field(default="Software Engineer", description="Target job role")
    difficulty: Optional[str] = Field(default="Medium", description="Target difficulty level")


class RAGEvaluationResponse(BaseModel):
    """Structured evaluation report assessing retrieval quality and generation accuracy."""

    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall RAG Triad aggregate score")
    rating: EvaluationRating = Field(..., description="Overall qualitative rating band")
    faithfulness: MetricScore = Field(..., description="Groundedness: question adherence to retrieved evidence")
    context_relevance: MetricScore = Field(..., description="Retrieval precision: context pertinence to the query")
    question_relevance: MetricScore = Field(..., description="Question quality: alignment with role and topic")
    summary: str = Field(..., description="High-level evaluation summary and recommendation")

