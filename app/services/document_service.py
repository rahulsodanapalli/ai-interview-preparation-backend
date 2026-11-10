"""Document Ingestion Service orchestrating PDF extraction, chunking, embedding, Pinecone indexing, and source registration."""

import logging
from typing import Optional
import uuid

from fastapi import Depends, UploadFile
from sqlalchemy.orm import Session

from app.schemas.document import (
    DocumentUploadResponse,
    SourceProcessingStatus,
    SourceType,
)
from app.services.chunking_service import TextChunkingService, get_chunking_service
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.pdf_service import PDFService, get_pdf_service
from app.services.pinecone_service import PineconeService, get_pinecone_service
from app.services.source_service import SourceService, get_source_service

logger = logging.getLogger(__name__)


class DocumentService:
    """Coordinates the end-to-end PDF ingestion pipeline into Pinecone and source tracking."""

    def __init__(
        self,
        pdf_service: Optional[PDFService] = None,
        chunking_service: Optional[TextChunkingService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        pinecone_service: Optional[PineconeService] = None,
        source_service: Optional[SourceService] = None,
    ) -> None:
        self.pdf_service = pdf_service or get_pdf_service()
        self.chunking_service = chunking_service or get_chunking_service()
        self.embedding_service = embedding_service or get_embedding_service()
        self.pinecone_service = pinecone_service or get_pinecone_service()
        self.source_service = source_service or get_source_service()

    async def ingest_pdf(
        self,
        file: UploadFile,
        db: Optional[Session] = None,
    ) -> DocumentUploadResponse:
        """Validate, extract text, chunk, embed, index in Pinecone, and register a PDF upload."""
        filename, full_text, page_count, character_count, preview = (
            await self.pdf_service.extract_text_and_metadata(file)
        )

        document_id = f"doc_{uuid.uuid4().hex[:16]}"
        chunks = self.chunking_service.create_chunks(full_text)
        chunk_count = len(chunks)

        chunk_texts = [c.text for c in chunks]
        vectors = await self.embedding_service.embed_chunks(chunk_texts)

        vector_count = await self.pinecone_service.upsert_document_chunks(
            document_id=document_id,
            chunks=chunks,
            vectors=vectors,
            filename=filename,
        )

        self.source_service.register_source(
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


def get_document_service(
    pdf_service: PDFService = Depends(get_pdf_service),
    chunking_service: TextChunkingService = Depends(get_chunking_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    pinecone_service: PineconeService = Depends(get_pinecone_service),
    source_service: SourceService = Depends(get_source_service),
) -> DocumentService:
    """FastAPI dependency provider for DocumentService."""
    return DocumentService(
        pdf_service=pdf_service,
        chunking_service=chunking_service,
        embedding_service=embedding_service,
        pinecone_service=pinecone_service,
        source_service=source_service,
    )

