"""Semantic Retrieval Service for RAG vector search in Pinecone.

Coordinates embedding generation for query strings, executes isolated vector similarity
searches scoped to specific documents in Pinecone, and returns structured chunk citations.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional
from fastapi import Depends

from app.core.config import settings
from app.services.embedding_service import (
    EmbeddingConnectionError,
    EmbeddingService,
    EmbeddingServiceError,
    EmbeddingTimeoutError,
    get_embedding_service,
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

logger = logging.getLogger(__name__)


# ============================================================================
# Retrieval Custom Exceptions
# ============================================================================


class RetrievalServiceError(Exception):
    """Base exception for semantic retrieval operations."""

    pass


class InvalidDocumentIdError(RetrievalServiceError):
    """Raised when document ID is missing, blank, or malformed."""

    pass


class EmptyQueryError(RetrievalServiceError):
    """Raised when the search query string is empty or whitespace-only."""

    pass


# ============================================================================
# Retrieved Chunk Data Structure
# ============================================================================


@dataclass(frozen=True)
class RetrievedChunk:
    """Represents a discrete document chunk retrieved via semantic similarity search."""

    chunk_id: str
    chunk_index: int
    score: float
    text: str
    document_id: str
    filename: str = "document.pdf"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def safe_excerpt(self) -> str:
        """Returns a safe, capped text excerpt suitable for API responses (max 200 chars)."""
        clean = " ".join(self.text.split())
        return clean[:200] + ("..." if len(clean) > 200 else "")


# ============================================================================
# Semantic Retrieval Service Implementation
# ============================================================================


class RetrievalService:
    """Orchestrates query embedding generation and document-scoped vector retrieval.

    Attributes:
        embedding_service: Service for generating 768-dim embeddings via Ollama.
        pinecone_service: Service for executing vector queries against Pinecone.
        min_score: Configured minimum cosine similarity score threshold for relevance.
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        pinecone_service: Optional[PineconeService] = None,
        min_score: Optional[float] = None,
    ):
        self.embedding_service = embedding_service or get_embedding_service()
        self.pinecone_service = pinecone_service or get_pinecone_service()
        self.min_score = (
            min_score
            if min_score is not None
            else getattr(settings, "RAG_MIN_RELEVANCE_SCORE", 0.55)
        )

    async def retrieve_chunks(
        self,
        query_text: str,
        document_id: str,
        top_k: int = 4,
        namespace: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """Retrieves semantically relevant chunks for a query, isolated to a specific document and filtered by threshold.

        Security Note (Document-Scoped Isolation):
            Enforces a metadata filter `document_id == document_id` inside Pinecone,
            guaranteeing that vectors belonging to other documents are never retrieved.
            Neither query vectors nor full resume text are ever written to application logs.

        Args:
            query_text: Plain text search query or topic focus area.
            document_id: Unique document identifier to isolate search scope.
            top_k: Number of most similar chunks to retrieve (default: 4).
            namespace: Optional Pinecone namespace override.
            min_score: Optional minimum relevance score threshold overriding instance default.

        Returns:
            List of RetrievedChunk objects meeting or exceeding the minimum relevance score.

        Raises:
            InvalidDocumentIdError: If document_id is blank or not a string.
            EmptyQueryError: If query_text is empty or whitespace only.
            EmbeddingServiceError: If Ollama embedding generation fails.
            PineconeServiceError: If Pinecone index query fails.
        """
        # 1. Validate inputs
        if not document_id or not str(document_id).strip():
            raise InvalidDocumentIdError("document_id must not be empty or whitespace only.")
        clean_doc_id = str(document_id).strip()
        if "#" in clean_doc_id:
            clean_doc_id = clean_doc_id.split("#")[0].strip()
        if not clean_doc_id:
            raise InvalidDocumentIdError("document_id must not be empty or whitespace only.")

        if not query_text or not str(query_text).strip():
            raise EmptyQueryError("query_text must not be empty or whitespace only.")
        clean_query = str(query_text).strip()

        if top_k <= 0:
            top_k = 4

        effective_min_score = min_score if min_score is not None else self.min_score

        logger.info(
            "Initiating semantic retrieval for document '%s' (top_k=%d, min_score=%.2f, query_length=%d chars).",
            clean_doc_id,
            top_k,
            effective_min_score,
            len(clean_query),
        )

        # 2. Generate dense query embedding vector (768 dimensions)
        query_vector = await self.embedding_service.embed_text(clean_query)

        # 3. Query Pinecone with document-scoped isolation filter
        raw_matches = await self.pinecone_service.query_similar_chunks(
            query_vector=query_vector,
            document_id=clean_doc_id,
            top_k=top_k,
            namespace=namespace,
        )

        if not raw_matches:
            logger.info("No matching chunks retrieved for document '%s'.", clean_doc_id)
            return []

        # 4. Convert to structured RetrievedChunk instances and filter by relevance threshold
        retrieved: List[RetrievedChunk] = []
        for match in raw_matches:
            score = float(match.get("score", 0.0)) if match.get("score") is not None else 0.0
            if effective_min_score is not None and score < effective_min_score:
                logger.debug(
                    "Excluding chunk '%s' for document '%s' with score %.4f below minimum relevance threshold %.2f.",
                    match.get("id", ""),
                    clean_doc_id,
                    score,
                    effective_min_score,
                )
                continue

            meta = match.get("metadata", {})
            chunk = RetrievedChunk(
                chunk_id=match.get("id", ""),
                chunk_index=match.get("chunk_index", meta.get("chunk_index", 0)),
                score=score,
                text=match.get("text", meta.get("text", "")),
                document_id=clean_doc_id,
                filename=match.get("filename", meta.get("filename", "document.pdf")),
                metadata=meta,
            )
            retrieved.append(chunk)

        if not retrieved and raw_matches:
            logger.info(
                "All %d retrieved chunks for document '%s' fell below minimum relevance threshold %.2f.",
                len(raw_matches),
                clean_doc_id,
                effective_min_score,
            )
        else:
            logger.info(
                "Successfully retrieved %d relevant chunks for document '%s' (top score=%.4f).",
                len(retrieved),
                clean_doc_id,
                retrieved[0].score if retrieved else 0.0,
            )
        return retrieved


def get_retrieval_service(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    pinecone_service: PineconeService = Depends(get_pinecone_service),
) -> RetrievalService:
    """FastAPI dependency provider for RetrievalService with injected dependencies."""
    return RetrievalService(
        embedding_service=embedding_service,
        pinecone_service=pinecone_service,
    )
