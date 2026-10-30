"""API routes for unified source management, URL ingestion, status checks, question generation, and session initiation."""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db_optional
from app.schemas.document import (
    SourceQuestionsRequest,
    SourceQuestionsResponse,
    SourceStatusResponse,
    SourceUploadResponse,
    UrlIngestRequest,
)
from app.schemas.session import CreateSessionRequest, SessionDetailResponse
from app.services.embedding_service import EmbeddingConnectionError, EmbeddingTimeoutError
from app.services.ollama_service import (
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from app.services.pinecone_service import PineconeConnectionError, PineconeTimeoutError
from app.services.rag_question_service import (
    RAGQuestionService,
    SourceInsufficientContentError,
    get_rag_question_service,
)
from app.services.session_service import SessionService, get_session_service
from app.services.source_service import (
    SourceNotFoundError,
    SourceNotReadyError,
    SourceService,
    get_source_service,
)
from app.services.url_service import (
    ContentTooLargeError,
    EmptyContentError,
    InvalidURLError,
    ProtectedURLError,
    SSRFSecurityError,
    UnsupportedContentTypeError,
    URLFetchError,
    URLService,
    get_url_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sources", tags=["Sources"])


@router.post(
    "/url",
    response_model=SourceUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest URL or GitHub Repository",
    description=(
        "Validate, fetch, chunk, embed, and index content from a public webpage or GitHub repository "
        "with SSRF protection and redirect validation."
    ),
    responses={
        200: {"description": "Source ingested and indexed successfully"},
        400: {"description": "Invalid URL, SSRF rejection, unsupported format, or empty content"},
        403: {"description": "Access forbidden by destination (paywall / login)"},
        413: {"description": "Response content too large"},
        500: {"description": "Internal processing error"},
        503: {"description": "Ollama or Pinecone unavailable"},
    },
)
async def ingest_source_url(
    payload: UrlIngestRequest,
    url_service: URLService = Depends(get_url_service),
    db: Optional[Session] = Depends(get_db_optional),
) -> SourceUploadResponse:
    """Ingest a public URL or GitHub repository as an interview context source."""
    try:
        return await url_service.ingest_url(url=payload.url, db=db)
    except (InvalidURLError, SSRFSecurityError, UnsupportedContentTypeError, EmptyContentError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ProtectedURLError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ContentTooLargeError as exc:
        status_413 = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413)
        raise HTTPException(status_code=status_413, detail=str(exc)) from exc
    except URLFetchError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (EmbeddingConnectionError, EmbeddingTimeoutError) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Embedding service unavailable: {exc}") from exc
    except (PineconeConnectionError, PineconeTimeoutError) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Pinecone service unavailable: {exc}") from exc
    except Exception as exc:
        logger.error("Unexpected error ingesting source URL: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the source URL.",
        ) from exc


@router.get(
    "/{source_id}/status",
    response_model=SourceStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Check Source Processing Status",
    description="Check the processing and indexing status of a source by source ID.",
    responses={
        200: {"description": "Source status retrieved successfully"},
        404: {"description": "Source not found"},
    },
)
async def get_source_status(
    source_id: str,
    source_service: SourceService = Depends(get_source_service),
    db: Optional[Session] = Depends(get_db_optional),
) -> SourceStatusResponse:
    """Retrieve indexing status and metrics for a source ID."""
    clean_id = (source_id or "").strip()
    try:
        return source_service.get_source(clean_id, db=db)
    except SourceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get(
    "",
    response_model=List[SourceStatusResponse],
    status_code=status.HTTP_200_OK,
    summary="List Ingested Sources",
    description="Retrieve a list of ingested documents and URLs ordered by most recent.",
)
async def list_sources(
    limit: int = Query(50, ge=1, le=100),
    skip: int = Query(0, ge=0),
    source_service: SourceService = Depends(get_source_service),
    db: Optional[Session] = Depends(get_db_optional),
) -> List[SourceStatusResponse]:
    """List all registered sources with optional pagination."""
    return source_service.list_sources(db=db, limit=limit, skip=skip)


@router.post(
    "/{source_id}/questions",
    response_model=SourceQuestionsResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Source-Grounded Interview Questions",
    description="Retrieve relevant chunks from the source and generate tailored interview questions with stable IDs.",
    responses={
        200: {"description": "Grounded interview questions generated successfully"},
        400: {"description": "Source has not finished processing or contains insufficient content"},
        404: {"description": "Source not found"},
        502: {"description": "Bad gateway from AI model"},
        503: {"description": "Pinecone or Ollama service is unavailable"},
    },
)
async def generate_source_questions(
    source_id: str,
    payload: Optional[SourceQuestionsRequest] = None,
    rag_service: RAGQuestionService = Depends(get_rag_question_service),
    db: Optional[Session] = Depends(get_db_optional),
) -> SourceQuestionsResponse:
    """Generate structured interview questions grounded in source content."""
    clean_id = (source_id or "").strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="source_id must not be empty.")

    req = payload or SourceQuestionsRequest(source_id=clean_id)
    if req.source_id != clean_id:
        req = req.model_copy(update={"source_id": clean_id})

    try:
        return await rag_service.generate_source_questions(req, db=db)
    except SourceNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (SourceNotReadyError, SourceInsufficientContentError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (EmbeddingConnectionError, EmbeddingTimeoutError) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Embedding service unavailable: {exc}") from exc
    except (PineconeConnectionError, PineconeTimeoutError) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Pinecone service unavailable: {exc}") from exc
    except (OllamaConnectionError, OllamaTimeoutError) as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"AI model service unavailable: {exc}") from exc
    except OllamaResponseError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unexpected error generating questions for source '%s': %s", clean_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred while generating source interview questions.",
        ) from exc


@router.post(
    "/sessions",
    response_model=SessionDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start Interview Session Linked to Source",
    description=(
        "Initialize an interview session associated with a source ID and pre-populated with "
        "generated questions, preventing accidental duplicate session creation."
    ),
    responses={
        201: {"description": "Interview session initialized successfully"},
        400: {"description": "Invalid source state, missing questions, or validation error"},
        404: {"description": "Source not found"},
    },
)
async def create_source_session(
    payload: CreateSessionRequest,
    session_service: SessionService = Depends(get_session_service),
    source_service: SourceService = Depends(get_source_service),
    db: Optional[Session] = Depends(get_db_optional),
) -> SessionDetailResponse:
    """Start an interview session linked to a source ID and optional pre-generated questions."""
    if payload.source_id:
        try:
            source = source_service.get_source(payload.source_id, db=db)
            if source.status != "completed":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Source '{payload.source_id}' is not ready (status: {source.status}).",
                )
        except SourceNotFoundError:
            logger.info("Source ID '%s' not registered in database; proceeding with session creation.", payload.source_id)

    return session_service.create_session(payload, db=db)

