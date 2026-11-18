"""API routes for document and URL upload, status checks, and document-grounded question generation."""

from typing import Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.dependencies.auth import require_active_user
from app.core.exceptions import (
    handle_document_upload_errors,
    handle_rag_question_errors,
    handle_url_ingestion_errors,
)
from app.db.session import get_db_optional
from app.models.auth import User
from app.schemas.document import (
    DocumentUploadResponse,
    SourceQuestionsRequest,
    SourceQuestionsResponse,
    SourceStatusResponse,
    SourceUploadResponse,
    UrlIngestRequest,
)
from app.schemas.interview import DocumentQuestionRequest, DocumentQuestionResponse
from app.services.document_service import DocumentService, get_document_service
from app.services.rag_question_service import RAGQuestionService, get_rag_question_service
from app.services.source_service import SourceForbiddenError, SourceNotFoundError, SourceService, get_source_service
from app.services.url_service import URLService, get_url_service
from app.utils.identifiers import normalize_document_id, supports_kwarg

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload, Chunk, Embed, and Store PDF Document Vectors in Pinecone",
    description=(
        "Upload a PDF resume or technical document (up to 10 MB) to validate its structure, "
        "extract text content, generate semantic chunks and embeddings, and store vector representations "
        "in Pinecone with document-scoped isolation metadata. Returns document metadata, source ID, chunk count, and vector count."
    ),
    responses={
        200: {
            "description": "PDF processed, embedded, and stored in Pinecone successfully",
            "model": DocumentUploadResponse,
        },
        400: {"description": "Invalid, corrupted, empty, unencrypted, or text-empty PDF document"},
        413: {"description": "Uploaded file exceeds configured maximum size limit"},
        500: {"description": "Internal error during embedding or Pinecone vector persistence"},
        503: {"description": "External Ollama embedding or Pinecone service is unavailable"},
    },
)
async def upload_document(
    file: UploadFile = File(..., description="Multipart PDF file to upload and extract text from"),
    document_service: DocumentService = Depends(get_document_service),
    db: Optional[Session] = Depends(get_db_optional),
    current_user: User = Depends(require_active_user),
) -> DocumentUploadResponse:
    """Accepts a multipart PDF upload, validates structure, extracts text, chunks, embeds, and stores in Pinecone."""
    try:
        with handle_document_upload_errors():
            kwargs = {"file": file, "db": db}
            if supports_kwarg(document_service.ingest_pdf, "user_id"):
                kwargs["user_id"] = current_user.id
            return await document_service.ingest_pdf(**kwargs)
    finally:
        await file.close()


@router.post(
    "/url",
    response_model=SourceUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest, Chunk, Embed, and Store Webpage Content in Pinecone",
    description=(
        "Ingests content from a public webpage or documentation URL with strict SSRF protection, "
        "redirect re-validation, response size limits, and HTML extraction. Reuses GitHub MCP for "
        "GitHub repository URLs."
    ),
    responses={
        200: {"description": "URL content processed and indexed successfully"},
        400: {"description": "Invalid URL format, SSRF target, unsupported content, or empty page"},
        403: {"description": "Access denied by target server (paywall / login)"},
        413: {"description": "Webpage content exceeds response size limits"},
        500: {"description": "Internal vector processing error"},
        503: {"description": "External Ollama or Pinecone service unavailable"},
    },
)
async def ingest_url(
    payload: UrlIngestRequest,
    url_service: URLService = Depends(get_url_service),
    db: Optional[Session] = Depends(get_db_optional),
    current_user: User = Depends(require_active_user),
) -> SourceUploadResponse:
    """Ingest a public URL or GitHub repository, validate SSRF, extract text, chunk, and index."""
    with handle_url_ingestion_errors(
        unexpected_detail="An unexpected error occurred while processing the URL."
    ):
        kwargs = {"url": payload.url, "db": db}
        if supports_kwarg(url_service.ingest_url, "user_id"):
            kwargs["user_id"] = current_user.id
        return await url_service.ingest_url(**kwargs)


@router.get(
    "/{document_id}/status",
    response_model=SourceStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Check Document or Source Processing Status",
    description="Check the current processing and indexing status for a document or source ID.",
    responses={
        200: {"description": "Source status retrieved successfully"},
        404: {"description": "Source not found"},
    },
)
async def get_document_status(
    document_id: str,
    source_service: SourceService = Depends(get_source_service),
    db: Optional[Session] = Depends(get_db_optional),
    current_user: User = Depends(require_active_user),
) -> SourceStatusResponse:
    """Retrieve indexing status and metadata for a document or source ID."""
    clean_id = (document_id or "").strip()
    try:
        return source_service.get_source(clean_id, db=db, user_id=current_user.id)
    except SourceForbiddenError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except SourceNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post(
    "/{document_id}/questions",
    response_model=SourceQuestionsResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Multiple Document-Grounded Interview Questions",
    description=(
        "Retrieves the most relevant chunks for the given document or source ID from Pinecone "
        "and generates multiple tailored interview questions with stable IDs grounded in context."
    ),
    responses={
        200: {"description": "Source-grounded questions generated successfully"},
        400: {"description": "Source has not finished processing or contains insufficient content"},
        404: {"description": "Source not found"},
        502: {"description": "Bad gateway from AI model"},
        503: {"description": "Pinecone or Ollama service is unavailable"},
    },
)
async def generate_questions_for_document(
    document_id: str,
    payload: Optional[SourceQuestionsRequest] = None,
    rag_service: RAGQuestionService = Depends(get_rag_question_service),
    db: Optional[Session] = Depends(get_db_optional),
    current_user: User = Depends(require_active_user),
) -> SourceQuestionsResponse:
    """Generate multiple structured interview questions grounded in document context."""
    clean_id = normalize_document_id(document_id)
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="document_id must not be empty.")

    req = payload or SourceQuestionsRequest(source_id=clean_id)
    if req.source_id != clean_id:
        req = req.model_copy(update={"source_id": clean_id})

    with handle_rag_question_errors(
        unexpected_detail="An unexpected internal error occurred while generating interview questions."
    ):
        kwargs = {"db": db}
        if supports_kwarg(rag_service.generate_source_questions, "user_id"):
            kwargs["user_id"] = current_user.id
        return await rag_service.generate_source_questions(req, **kwargs)


@router.post(
    "/{document_id}/question",
    response_model=DocumentQuestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Document-Grounded Interview Question for Specific Document (RAG)",
    description=(
        "Retrieves the most relevant technical chunks for the given document ID from Pinecone "
        "and generates a tailored interview question grounded in the document context."
    ),
    responses={
        200: {
            "description": "Document-grounded question generated successfully",
            "model": DocumentQuestionResponse,
        },
        400: {"description": "Invalid document ID or query parameters"},
        422: {"description": "Validation error in request payload"},
        500: {"description": "Internal model output or vector processing error"},
        502: {"description": "Bad gateway or malformed JSON from Ollama model"},
        503: {"description": "Pinecone or Ollama service is unavailable"},
    },
)
async def generate_question_for_document(
    document_id: str,
    payload: Optional[DocumentQuestionRequest] = None,
    rag_service: RAGQuestionService = Depends(get_rag_question_service),
    db: Optional[Session] = Depends(get_db_optional),
    current_user: User = Depends(require_active_user),
) -> DocumentQuestionResponse:
    """Convenience endpoint generating a single grounded question for a specific path document_id."""
    clean_id = normalize_document_id(document_id)
    if not clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="document_id in URL path must not be empty.",
        )

    req = payload or DocumentQuestionRequest(document_id=clean_id)
    if req.document_id != clean_id:
        req = req.model_copy(update={"document_id": clean_id})

    with handle_rag_question_errors(
        unexpected_detail="An unexpected internal error occurred while generating the document question.",
        detailed_service_messages=True,
    ):
        kwargs = {}
        if supports_kwarg(rag_service.generate_document_question, "db"):
            kwargs["db"] = db
        if supports_kwarg(rag_service.generate_document_question, "user_id"):
            kwargs["user_id"] = current_user.id
        return await rag_service.generate_document_question(req, **kwargs)
