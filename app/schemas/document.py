"""Pydantic schemas for document and URL ingestion, processing status, and source questions."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.interview import (
    DifficultyLevel,
    ExperienceLevel,
    QuestionType,
    RetrievedChunkMetadata,
)


class SourceType(str, Enum):
    """Supported source ingestion types."""

    DOCUMENT = "document"
    URL = "url"
    GITHUB = "github"


class SourceProcessingStatus(str, Enum):
    """Processing and indexing lifecycle status."""

    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentUploadResponse(BaseModel):
    """Response returned upon successful processing and text extraction of an uploaded document."""

    success: bool = Field(
        default=True,
        description="Indicates whether the document text was successfully extracted and indexed.",
    )
    filename: str = Field(
        ...,
        description="Sanitized representation of the uploaded document filename.",
    )
    page_count: int = Field(
        default=1,
        description="Total number of pages or sections in the document.",
        ge=1,
    )
    character_count: int = Field(
        ...,
        description="Total number of characters extracted across the document.",
        ge=1,
    )
    preview: str = Field(
        ...,
        description="Safe preview of the extracted document text (capped at 500 characters).",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Unique document identifier used for vector retrieval isolation.",
    )
    source_id: Optional[str] = Field(
        default=None,
        description="Unique source identifier representing this document for retrieval and interview sessions.",
    )
    status: str = Field(
        default="completed",
        description="Processing status of the document ('completed', 'processing', 'failed').",
    )
    chunk_count: Optional[int] = Field(
        default=None,
        description="Total number of text chunks created from the document.",
        ge=0,
    )
    vector_count: Optional[int] = Field(
        default=None,
        description="Total number of vector embeddings stored in Pinecone.",
        ge=0,
    )
    message: str = Field(
        default="Document text extracted and processed successfully.",
        description="Informational status message summarizing the extraction and storage result.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "filename": "candidate_resume.pdf",
                "page_count": 2,
                "character_count": 1420,
                "preview": "Jane Doe\nSenior Software Engineer\n\nExperience:\n- Lead Developer...",
                "document_id": "doc_1234567890abcdef",
                "source_id": "doc_1234567890abcdef",
                "status": "completed",
                "chunk_count": 3,
                "vector_count": 3,
                "message": "Document text extracted and processed successfully.",
            }
        }
    )


class UrlIngestRequest(BaseModel):
    """Request payload for ingesting and indexing a public webpage or documentation URL."""

    url: str = Field(
        ...,
        min_length=8,
        max_length=2048,
        description="Publicly accessible HTTP or HTTPS URL of the webpage, documentation, or GitHub repository.",
        examples=["https://docs.python.org/3/tutorial/index.html"],
    )

    @field_validator("url", mode="before")
    @classmethod
    def validate_url_string(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("URL must not be empty or whitespace only.")
        clean_url = v.strip()
        lower = clean_url.lower()
        if not (lower.startswith("http://") or lower.startswith("https://")):
            raise ValueError("URL scheme must be either HTTP or HTTPS.")
        return clean_url


class SourceUploadResponse(BaseModel):
    """Response returned upon successful ingestion of a document or URL source."""

    success: bool = Field(
        default=True,
        description="Indicates whether the source was successfully fetched, extracted, and indexed.",
    )
    source_id: str = Field(
        ...,
        description="Unique source identifier to reference this source in retrieval and session creation.",
    )
    source_type: SourceType = Field(
        ...,
        description="The classified type of the source ('document', 'url', 'github').",
    )
    status: SourceProcessingStatus = Field(
        default=SourceProcessingStatus.COMPLETED,
        description="Processing and indexing status ('completed', 'processing', 'failed').",
    )
    url: Optional[str] = Field(
        default=None,
        description="Canonical URL of the ingested webpage or repository.",
    )
    title: str = Field(
        ...,
        description="Extracted page title, repository name, or sanitized filename.",
    )
    character_count: int = Field(
        ...,
        description="Total character count extracted from the source content.",
        ge=0,
    )
    chunk_count: int = Field(
        default=0,
        description="Total number of chunks created and indexed.",
        ge=0,
    )
    vector_count: int = Field(
        default=0,
        description="Total number of 768-dim vector embeddings stored in Pinecone.",
        ge=0,
    )
    preview: str = Field(
        default="",
        description="Safe excerpt preview of the extracted content (up to 500 characters).",
    )
    message: str = Field(
        default="Source content extracted, embedded, and indexed successfully.",
        description="Informational status message summarizing processing results.",
    )


class SourceStatusResponse(BaseModel):
    """Detailed status and metadata for a registered source."""

    source_id: str = Field(..., description="Unique source identifier")
    status: SourceProcessingStatus = Field(..., description="Current processing status")
    source_type: SourceType = Field(..., description="Type of source")
    title: str = Field(..., description="Source title or filename")
    url: Optional[str] = Field(default=None, description="Canonical URL if source is web-based")
    filename: Optional[str] = Field(default=None, description="Filename if source is an uploaded document")
    character_count: int = Field(default=0, description="Extracted characters")
    chunk_count: int = Field(default=0, description="Indexed text chunks")
    vector_count: int = Field(default=0, description="Indexed vector embeddings in Pinecone")
    error_message: Optional[str] = Field(default=None, description="Error details if status is failed")
    created_at: str = Field(..., description="Timestamp when source was created (ISO 8601 UTC)")


class SourceQuestionItem(BaseModel):
    """Structured interview question with stable ID grounded in source context."""

    id: str = Field(
        ...,
        description="Stable unique question identifier (e.g. 'q_1', 'q_2', or UUID)",
        examples=["q_1"],
    )
    order: int = Field(
        ...,
        ge=1,
        description="1-indexed sequence number of the question",
        examples=[1],
    )
    question: str = Field(
        ...,
        min_length=5,
        description="The source-grounded interview question text",
    )
    expected_answer_points: List[str] = Field(
        ...,
        min_length=1,
        description="Key technical criteria or concepts a strong answer should address",
    )
    follow_up_question: Optional[str] = Field(
        default=None,
        description="Probing follow-up question related to this topic",
    )
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.MEDIUM,
        description="Configured or generated difficulty level",
    )
    topic: str = Field(
        default="Technical Context",
        description="Technical topic or domain targeted by this question",
    )


class SourceQuestionsRequest(BaseModel):
    """Request payload for generating multiple questions grounded in a specific source."""

    source_id: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=100,
        description="Unique source ID (if not provided in the URL path)",
    )
    role: Optional[str] = Field(
        default="Software Engineer",
        min_length=2,
        max_length=100,
        description="Target job role for question generation",
        examples=["Full Stack Developer"],
    )
    experience_level: ExperienceLevel = Field(
        default=ExperienceLevel.MID_LEVEL,
        description="Candidate experience level (Fresher, Junior, Mid-Level, Senior)",
    )
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.MEDIUM,
        description="Target difficulty level (Easy, Medium, Hard)",
    )
    topic: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=100,
        description="Optional technical topic focus area",
    )
    question_type: QuestionType = Field(
        default=QuestionType.TECHNICAL,
        description="Question category type",
    )
    count: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of structured questions to generate (between 1 and 10, default 3)",
    )


class SourceQuestionsResponse(BaseModel):
    """Structured response containing generated questions grounded in source content."""

    source_id: str = Field(..., description="Source ID from which content was retrieved")
    context_grounded: bool = Field(
        default=True,
        description="Indicates whether questions are directly grounded in source context",
    )
    total_questions: int = Field(..., ge=1, description="Count of generated questions")
    questions: List[SourceQuestionItem] = Field(
        ...,
        min_length=1,
        description="List of structured interview questions with stable IDs",
    )
    retrieved_chunks: List[RetrievedChunkMetadata] = Field(
        default_factory=list,
        description="Citations and excerpts of source chunks retrieved as context",
    )
