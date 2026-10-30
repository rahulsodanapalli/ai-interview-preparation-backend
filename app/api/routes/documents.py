"""API routes for document and URL upload, validation, chunking, embedding, and Pinecone storage."""

import logging
from typing import Optional
import uuid
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.db.session import get_db_optional
from app.schemas.document import (
    DocumentUploadResponse,
    SourceProcessingStatus,
    SourceQuestionsRequest,
    SourceQuestionsResponse,
    SourceStatusResponse,
    SourceType,
    SourceUploadResponse,
    UrlIngestRequest,
)
from app.schemas.interview import DocumentQuestionRequest, DocumentQuestionResponse
from app.services.chunking_service import TextChunkingService, get_chunking_service
from app.services.embedding_service import (
    EmbeddingConnectionError,
    EmbeddingService,
    EmbeddingServiceError,
    EmbeddingTimeoutError,
    get_embedding_service,
)
from app.services.pdf_service import (
    PDFEmptyFileError,
    PDFEncryptedError,
    PDFInvalidError,
    PDFNoTextError,
    PDFPageLimitError,
    PDFService,
    PDFSizeLimitError,
    get_pdf_service,
)
from app.services.pinecone_service import (
    PineconeConfigurationError,
    PineconeConnectionError,
    PineconeDimensionMismatchError,
    PineconeIndexNotFoundError,
    PineconeResponseError,
    PineconeService,
    PineconeServiceError,
    PineconeTimeoutError,
    get_pinecone_service,
)
from app.services.rag_question_service import (
    RAGQuestionService,
    SourceInsufficientContentError,
    get_rag_question_service,
)
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
        400: {
            "description": "Invalid, corrupted, empty, unencrypted, or text-empty PDF document",
        },
        413: {
            "description": "Uploaded file exceeds configured maximum size limit",
        },
        500: {
            "description": "Internal error during embedding or Pinecone vector persistence",
        },
        503: {
            "description": "External Ollama embedding or Pinecone service is unavailable",
        },
    },
)
async def upload_document(
    file: UploadFile = File(..., description="Multipart PDF file to upload and extract text from"),
    pdf_service: PDFService = Depends(get_pdf_service),
    chunking_service: TextChunkingService = Depends(get_chunking_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    pinecone_service: PineconeService = Depends(get_pinecone_service),
    source_service: SourceService = Depends(get_source_service),
    db: Optional[Session] = Depends(get_db_optional),
) -> DocumentUploadResponse:
    """Accepts a multipart PDF upload, validates structure, extracts text, chunks, embeds, and stores in Pinecone."""
    try:
        # 1. Read, validate stream, and extract text and metadata
        filename, full_text, page_count, character_count, preview = await pdf_service.extract_text_and_metadata(file)

        # 2. Generate a unique, document-scoped identifier for retrieval isolation
        document_id = f"doc_{uuid.uuid4().hex[:16]}"

        # 3. Create discrete semantic chunks
        chunks = chunking_service.create_chunks(full_text)
        chunk_count = len(chunks)

        # 4. Generate embeddings for all text chunks
        chunk_texts = [c.text for c in chunks]
        vectors = await embedding_service.embed_chunks(chunk_texts)

        # 5. Upsert chunk vectors and safe metadata into Pinecone
        vector_count = await pinecone_service.upsert_document_chunks(
            document_id=document_id,
            chunks=chunks,
            vectors=vectors,
            filename=filename,
        )

        # 6. Persist source metadata to PostgreSQL and in-memory source tracker
        source_service.register_source(
            source_id=document_id,
            source_type=SourceType.DOCUMENT,
            title=filename,
            filename=filename,
            status=SourceProcessingStatus.COMPLETED,
            character_count=character_count,
            chunk_count=chunk_count,
            vector_count=vector_count,
            db=db,
        )

        logger.info(
            "Document '%s' successfully ingested into Pinecone: id='%s', pages=%d, chunks=%d, vectors=%d.",
            filename,
            document_id,
            page_count,
            chunk_count,
            vector_count,
        )

        return DocumentUploadResponse(
            success=True,
            filename=filename,
            document_id=document_id,
            source_id=document_id,
            status="completed",
            page_count=page_count,
            character_count=character_count,
            chunk_count=chunk_count,
            vector_count=vector_count,
            preview=preview,
            message=(
                f"Successfully extracted {character_count} characters across {page_count} page(s) "
                f"and stored {vector_count} vector(s) in Pinecone."
            ),
        )

    except PDFSizeLimitError as exc:
        logger.warning("Upload size limit exceeded: %s", exc)
        status_413 = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413)
        raise HTTPException(
            status_code=status_413,
            detail=str(exc),
        ) from exc
    except (
        PDFEmptyFileError,
        PDFInvalidError,
        PDFEncryptedError,
        PDFPageLimitError,
        PDFNoTextError,
    ) as exc:
        logger.warning("PDF processing failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except (EmbeddingConnectionError, EmbeddingTimeoutError) as exc:
        logger.error("Embedding service unavailable during document upload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service is currently unavailable: {exc}",
        ) from exc
    except (PineconeConnectionError, PineconeTimeoutError) as exc:
        logger.error("Pinecone vector storage service unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Pinecone vector storage service is currently unavailable: {exc}",
        ) from exc
    except (
        PineconeConfigurationError,
        PineconeIndexNotFoundError,
        PineconeDimensionMismatchError,
        PineconeResponseError,
        EmbeddingServiceError,
    ) as exc:
        logger.error("Vector storage processing error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected error during PDF processing and vector storage: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing and storing the document.",
        ) from exc
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
) -> SourceUploadResponse:
    """Ingest a public URL or GitHub repository, validate SSRF, extract text, chunk, and index."""
    try:
        return await url_service.ingest_url(url=payload.url, db=db)
    except (InvalidURLError, SSRFSecurityError, UnsupportedContentTypeError, EmptyContentError) as exc:
        logger.warning("URL ingestion rejected: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ProtectedURLError as exc:
        logger.warning("URL destination protected: %s", exc)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ContentTooLargeError as exc:
        status_413 = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413)
        raise HTTPException(status_code=status_413, detail=str(exc)) from exc
    except URLFetchError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (EmbeddingConnectionError, EmbeddingTimeoutError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service is currently unavailable: {exc}",
        ) from exc
    except (PineconeConnectionError, PineconeTimeoutError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Pinecone vector storage service is currently unavailable: {exc}",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error during URL ingestion: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the URL.",
        ) from exc


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
) -> SourceStatusResponse:
    """Retrieve indexing status and metadata for a document or source ID."""
    clean_id = (document_id or "").strip()
    try:
        return source_service.get_source(clean_id, db=db)
    except SourceNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


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
) -> SourceQuestionsResponse:
    """Generate multiple structured interview questions grounded in document context."""
    clean_id = (document_id or "").strip()
    if "#" in clean_id:
        clean_id = clean_id.split("#")[0].strip()
    if not clean_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="document_id must not be empty.")

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
        logger.error("Unexpected error generating document questions: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred while generating interview questions.",
        ) from exc


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
) -> DocumentQuestionResponse:
    """Convenience endpoint generating a single grounded question for a specific path document_id."""
    clean_id = (document_id or "").strip()
    if "#" in clean_id:
        clean_id = clean_id.split("#")[0].strip()
    if not clean_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="document_id in URL path must not be empty.",
        )
    req = payload or DocumentQuestionRequest(document_id=clean_id)
    if req.document_id != clean_id:
        req = req.model_copy(update={"document_id": clean_id})

    try:
        return await rag_service.generate_document_question(req)
    except (EmbeddingConnectionError, EmbeddingTimeoutError) as exc:
        logger.error("Embedding service unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service is currently unavailable. Please verify Ollama is running.",
        ) from exc
    except (PineconeConnectionError, PineconeTimeoutError) as exc:
        logger.error("Pinecone service unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pinecone vector storage service is currently unavailable.",
        ) from exc
    except (
        PineconeConfigurationError,
        PineconeDimensionMismatchError,
        PineconeIndexNotFoundError,
        PineconeResponseError,
    ) as exc:
        logger.error("Pinecone storage error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pinecone vector storage error: {exc}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected error during document question generation: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred while generating the document question.",
        ) from exc
