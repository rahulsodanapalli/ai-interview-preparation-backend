"""Pydantic schemas for request and response validation."""

from app.schemas.document import DocumentUploadResponse
from app.schemas.health import (
    DatabaseHealthResponse,
    HealthCheckResponse,
    OllamaHealthResponse,
)
from app.schemas.interview import (
    DifficultyLevel,
    DocumentQuestionRequest,
    DocumentQuestionResponse,
    EvaluationRating,
    ExperienceLevel,
    InterviewEvaluationRequest,
    InterviewEvaluationResponse,
    InterviewQuestionRequest,
    InterviewQuestionResponse,
    QuestionType,
    RetrievedChunkMetadata,
)
from app.schemas.session import (
    CreateSessionRequest,
    InterviewSessionSummaryResponse,
    SessionDetailResponse,
    SessionNextQuestionResponse,
    SessionQuestionItem,
    SessionStatus,
    SessionSubmitAnswerResponse,
    SubmitAnswerRequest,
)

from app.schemas.github import (
    RepoEvidence,
    RepoFileContentResponse,
    RepoFileItem,
    RepoFileRequest,
    RepoFilesListResponse,
    RepoIdentifierRequest,
    RepoMetadataResponse,
    RepoQuestionRequest,
    RepoQuestionResponse,
    RepoReadmeResponse,
    parse_repo_identifier,
)

__all__ = [
    "DocumentUploadResponse",
    "DatabaseHealthResponse",
    "HealthCheckResponse",
    "OllamaHealthResponse",

    "DifficultyLevel",
    "EvaluationRating",
    "ExperienceLevel",
    "InterviewEvaluationRequest",
    "InterviewEvaluationResponse",
    "InterviewQuestionRequest",
    "InterviewQuestionResponse",
    "QuestionType",
    "RetrievedChunkMetadata",
    "DocumentQuestionRequest",
    "DocumentQuestionResponse",
    "CreateSessionRequest",
    "InterviewSessionSummaryResponse",
    "SessionDetailResponse",
    "SessionNextQuestionResponse",
    "SessionQuestionItem",
    "SessionStatus",
    "SessionSubmitAnswerResponse",
    "SubmitAnswerRequest",

    # GitHub MCP schemas
    "RepoEvidence",
    "RepoFileContentResponse",
    "RepoFileItem",
    "RepoFileRequest",
    "RepoFilesListResponse",
    "RepoIdentifierRequest",
    "RepoMetadataResponse",
    "RepoQuestionRequest",
    "RepoQuestionResponse",
    "RepoReadmeResponse",
    "parse_repo_identifier",
]
