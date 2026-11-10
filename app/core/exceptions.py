"""Shared HTTP exception translation helpers for FastAPI route handlers.

Centralizes mapping of domain and integration exceptions into consistent
FastAPI HTTPException responses while preserving exact status codes and
API error contracts across all endpoints.
"""

from contextlib import contextmanager
import logging
from typing import Iterator, Optional
from fastapi import HTTPException, status
from pydantic import ValidationError

from app.services.answer_evaluation_service import EvaluationOutputFormatError
from app.services.embedding_service import (
    EmbeddingConnectionError,
    EmbeddingServiceError,
    EmbeddingTimeoutError,
)
from app.services.github_mcp_service import (
    GitHubConnectionError,
    GitHubFileNotFoundError,
    GitHubFileTooLargeError,
    GitHubPermissionError,
    GitHubRateLimitError,
    GitHubRepoNotFoundError,
    GitHubServiceError,
    GitHubTimeoutError,
    GitHubValidationError,
)
from app.services.ollama_service import (
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from app.services.pdf_service import (
    PDFEmptyFileError,
    PDFEncryptedError,
    PDFInvalidError,
    PDFNoTextError,
    PDFPageLimitError,
    PDFSizeLimitError,
)
from app.services.pinecone_service import (
    PineconeConfigurationError,
    PineconeConnectionError,
    PineconeDimensionMismatchError,
    PineconeIndexNotFoundError,
    PineconeResponseError,
    PineconeTimeoutError,
)
from app.services.question_generation_service import QuestionOutputFormatError
from app.services.rag_question_service import SourceInsufficientContentError
from app.services.retrieval_service import EmptyQueryError, InvalidDocumentIdError
from app.services.session_service import SessionNotFoundError, SessionStateError
from app.services.source_service import SourceNotFoundError, SourceNotReadyError
from app.services.url_service import (
    ContentTooLargeError,
    EmptyContentError,
    InvalidURLError,
    ProtectedURLError,
    SSRFSecurityError,
    UnsupportedContentTypeError,
    URLFetchError,
)

logger = logging.getLogger(__name__)

HTTP_413_STATUS = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413)


@contextmanager
def handle_github_errors(
    *,
    file_not_found_detail: Optional[str] = None,
    wrap_generation_messages: bool = False,
) -> Iterator[None]:
    """Translate GitHub MCP and Ollama exceptions into standard HTTPExceptions."""
    try:
        yield
    except HTTPException:
        raise
    except (ValueError, GitHubValidationError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except GitHubFileTooLargeError as exc:
        raise HTTPException(status_code=HTTP_413_STATUS, detail=str(exc)) from exc
    except GitHubRepoNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except GitHubFileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=file_not_found_detail or str(exc),
        ) from exc
    except GitHubPermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except GitHubRateLimitError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except (OllamaConnectionError, GitHubConnectionError) as exc:
        detail = f"Service unavailable: {exc}" if wrap_generation_messages else str(exc)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail) from exc
    except (OllamaTimeoutError, GitHubTimeoutError) as exc:
        detail = f"Request timed out: {exc}" if wrap_generation_messages else str(exc)
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=detail) from exc
    except OllamaResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI model generation error: {exc}",
        ) from exc
    except GitHubServiceError as exc:
        detail = f"GitHub service error: {exc}" if wrap_generation_messages else str(exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail) from exc
    except Exception as exc:
        logger.error("Unexpected error in GitHub endpoint: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred while processing the GitHub request.",
        ) from exc


@contextmanager
def handle_url_ingestion_errors(
    *,
    unexpected_detail: str = "An unexpected error occurred while processing the source URL.",
) -> Iterator[None]:
    """Translate URL ingestion, SSRF, embedding, and Pinecone exceptions into HTTPExceptions."""
    try:
        yield
    except HTTPException:
        raise
    except (
        InvalidURLError,
        SSRFSecurityError,
        UnsupportedContentTypeError,
        EmptyContentError,
        URLFetchError,
    ) as exc:
        logger.warning("URL ingestion rejected: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except ProtectedURLError as exc:
        logger.warning("URL destination protected: %s", exc)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ContentTooLargeError as exc:
        raise HTTPException(status_code=HTTP_413_STATUS, detail=str(exc)) from exc
    except (EmbeddingConnectionError, EmbeddingTimeoutError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service unavailable: {exc}",
        ) from exc
    except (PineconeConnectionError, PineconeTimeoutError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Pinecone service unavailable: {exc}",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error during URL ingestion: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=unexpected_detail,
        ) from exc


@contextmanager
def handle_document_upload_errors() -> Iterator[None]:
    """Translate PDF upload, embedding, and Pinecone exceptions into HTTPExceptions."""
    try:
        yield
    except HTTPException:
        raise
    except PDFSizeLimitError as exc:
        logger.warning("Upload size limit exceeded: %s", exc)
        raise HTTPException(status_code=HTTP_413_STATUS, detail=str(exc)) from exc
    except (
        PDFEmptyFileError,
        PDFInvalidError,
        PDFEncryptedError,
        PDFPageLimitError,
        PDFNoTextError,
    ) as exc:
        logger.warning("PDF processing failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Unexpected error during PDF processing and vector storage: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing and storing the document.",
        ) from exc


@contextmanager
def handle_rag_question_errors(
    *,
    unexpected_detail: str = "An unexpected internal error occurred while generating interview questions.",
    detailed_service_messages: bool = False,
) -> Iterator[None]:
    """Translate RAG retrieval, source readiness, Pinecone, and Ollama errors into HTTPExceptions."""
    try:
        yield
    except HTTPException:
        raise
    except SourceNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (
        SourceNotReadyError,
        SourceInsufficientContentError,
        InvalidDocumentIdError,
        EmptyQueryError,
    ) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except (EmbeddingConnectionError, EmbeddingTimeoutError) as exc:
        detail = (
            "Embedding service is currently unavailable. Please verify Ollama is running."
            if detailed_service_messages
            else f"Embedding service unavailable: {exc}"
        )
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail) from exc
    except (PineconeConnectionError, PineconeTimeoutError) as exc:
        detail = (
            "Pinecone vector storage service is currently unavailable."
            if detailed_service_messages
            else f"Pinecone service unavailable: {exc}"
        )
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail) from exc
    except (OllamaConnectionError, OllamaTimeoutError) as exc:
        detail = (
            "AI inference service is currently unavailable. Please verify Ollama is running."
            if detailed_service_messages
            else f"AI model service unavailable: {exc}"
        )
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail) from exc
    except (
        PineconeConfigurationError,
        PineconeDimensionMismatchError,
        PineconeIndexNotFoundError,
        PineconeResponseError,
    ) as exc:
        logger.error("Pinecone storage error during RAG question generation: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pinecone vector storage error: {exc}",
        ) from exc
    except OllamaResponseError as exc:
        detail = (
            "Failed to generate interview question from the local AI model."
            if detailed_service_messages
            else str(exc)
        )
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail) from exc
    except ValidationError as exc:
        detail = (
            "The AI model returned data that does not match the required question format."
            if detailed_service_messages
            else str(exc)
        )
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail) from exc
    except Exception as exc:
        logger.error("Unexpected error during RAG question generation: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=unexpected_detail,
        ) from exc


@contextmanager
def handle_session_errors() -> Iterator[None]:
    """Translate session lookup and lifecycle state errors into HTTPExceptions."""
    try:
        yield
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except SessionStateError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@contextmanager
def handle_interview_ai_errors(
    *,
    operation: str = "question generation",
    upstream_error_detail: str = "Failed to generate interview question from the local AI model.",
) -> Iterator[None]:
    """Translate Ollama connection, timeout, response, and output format errors into HTTPExceptions."""
    try:
        yield
    except HTTPException:
        raise
    except (OllamaConnectionError, OllamaTimeoutError) as exc:
        logger.error("Ollama service unavailable during %s: %s", operation, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI inference service is currently unavailable. Please verify Ollama is running.",
        ) from exc
    except (QuestionOutputFormatError, EvaluationOutputFormatError) as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    except OllamaResponseError as exc:
        logger.error("Ollama error during %s: %s", operation, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=upstream_error_detail,
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error during %s: %s", operation, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the request.",
        ) from exc

