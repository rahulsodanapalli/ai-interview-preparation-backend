"""Pydantic schemas for document upload and ingestion."""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class DocumentUploadResponse(BaseModel):
    """Response returned upon successful processing and text extraction of an uploaded document."""

    success: bool = Field(
        default=True,
        description="Indicates whether the document text was successfully extracted.",
    )
    filename: str = Field(
        ...,
        description="Sanitized representation of the uploaded document filename.",
    )
    page_count: int = Field(
        ...,
        description="Total number of pages in the PDF document.",
        ge=1,
    )
    character_count: int = Field(
        ...,
        description="Total number of characters extracted across all pages.",
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
        default="PDF text extracted and processed successfully.",
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
                "message": "PDF text extracted successfully.",
            }
        }
    )
