"""Source Management Service for document and URL metadata tracking and persistence."""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.interview import DocumentSource
from app.schemas.document import SourceProcessingStatus, SourceStatusResponse, SourceType

logger = logging.getLogger(__name__)


class SourceServiceError(Exception):
    """Base exception for source service errors."""
    pass


class SourceNotFoundError(SourceServiceError):
    """Raised when a requested source ID is not found."""
    pass


class SourceNotReadyError(SourceServiceError):
    """Raised when an operation requires a completed source, but processing is pending or failed."""
    pass


def _db_to_source_status(src_db: DocumentSource) -> SourceStatusResponse:
    """Helper to convert a DocumentSource database model into a SourceStatusResponse schema."""
    created_str = (
        src_db.created_at.isoformat()
        if src_db.created_at
        else datetime.now(timezone.utc).isoformat()
    )
    return SourceStatusResponse(
        source_id=src_db.id,
        status=SourceProcessingStatus(src_db.status),
        source_type=SourceType(src_db.source_type),
        title=src_db.title,
        url=src_db.url,
        filename=src_db.filename,
        character_count=src_db.character_count or 0,
        chunk_count=src_db.chunk_count or 0,
        vector_count=src_db.vector_count or 0,
        error_message=src_db.error_message,
        created_at=created_str,
    )


class SourceService:
    """Service managing metadata, indexing state, and persistence for documents and URLs."""

    def __init__(self) -> None:
        self._sources: Dict[str, SourceStatusResponse] = {}

    def reset_store(self) -> None:
        """Clear the in-memory cache for test isolation."""
        self._sources.clear()

    def register_source(
        self,
        source_id: str,
        source_type: SourceType,
        title: str,
        url: Optional[str] = None,
        filename: Optional[str] = None,
        status: SourceProcessingStatus = SourceProcessingStatus.COMPLETED,
        character_count: int = 0,
        chunk_count: int = 0,
        vector_count: int = 0,
        error_message: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> SourceStatusResponse:
        """Register a new source in in-memory cache and persist to PostgreSQL if db is available."""
        now = datetime.now(timezone.utc).isoformat()
        source_resp = SourceStatusResponse(
            source_id=source_id,
            status=status,
            source_type=source_type,
            title=title,
            url=url,
            filename=filename,
            character_count=character_count,
            chunk_count=chunk_count,
            vector_count=vector_count,
            error_message=error_message,
            created_at=now,
        )

        self._sources[source_id] = source_resp

        if db is not None:
            try:
                db_source = DocumentSource(
                    id=source_id,
                    source_type=source_type.value if hasattr(source_type, "value") else str(source_type),
                    title=title[:255],
                    url=url,
                    filename=filename[:255] if filename else None,
                    status=status.value if hasattr(status, "value") else str(status),
                    character_count=character_count,
                    chunk_count=chunk_count,
                    vector_count=vector_count,
                    error_message=error_message,
                )
                db.add(db_source)
                db.flush()
                logger.info("Persisted source record '%s' to database.", source_id)
            except Exception as exc:
                logger.warning("Failed to persist source '%s' to database: %s", source_id, exc)

        return source_resp

    def update_source(
        self,
        source_id: str,
        status: Optional[SourceProcessingStatus] = None,
        character_count: Optional[int] = None,
        chunk_count: Optional[int] = None,
        vector_count: Optional[int] = None,
        error_message: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> SourceStatusResponse:
        """Update an existing source's indexing metrics and lifecycle status."""
        source = self.get_source(source_id, db=db)
        updated_dict = source.model_dump()

        if status is not None:
            updated_dict["status"] = status
        if character_count is not None:
            updated_dict["character_count"] = character_count
        if chunk_count is not None:
            updated_dict["chunk_count"] = chunk_count
        if vector_count is not None:
            updated_dict["vector_count"] = vector_count
        if error_message is not None:
            updated_dict["error_message"] = error_message

        new_source = SourceStatusResponse(**updated_dict)
        self._sources[source_id] = new_source

        if db is not None:
            try:
                src_db = db.get(DocumentSource, source_id)
                if src_db:
                    if status is not None:
                        src_db.status = status.value if hasattr(status, "value") else str(status)
                    if character_count is not None:
                        src_db.character_count = character_count
                    if chunk_count is not None:
                        src_db.chunk_count = chunk_count
                    if vector_count is not None:
                        src_db.vector_count = vector_count
                    if error_message is not None:
                        src_db.error_message = error_message
                    db.flush()
            except Exception as exc:
                logger.warning("Failed to update database source '%s': %s", source_id, exc)

        return new_source

    def get_source(
        self,
        source_id: str,
        db: Optional[Session] = None,
    ) -> SourceStatusResponse:
        """Retrieve source status and metadata, checking database first if available."""
        clean_id = (source_id or "").strip()
        if not clean_id:
            raise SourceNotFoundError("source_id must not be empty.")

        if db is not None:
            try:
                src_db = db.get(DocumentSource, clean_id)
                if src_db is not None:
                    resp = _db_to_source_status(src_db)
                    self._sources[clean_id] = resp
                    return resp
            except Exception as exc:
                logger.debug("Database lookup for source '%s' failed: %s", clean_id, exc)

        if clean_id in self._sources:
            return self._sources[clean_id]

        raise SourceNotFoundError(f"Source with ID '{clean_id}' was not found.")

    def list_sources(
        self,
        db: Optional[Session] = None,
        limit: int = 50,
        skip: int = 0,
    ) -> List[SourceStatusResponse]:
        """List registered sources ordered by most recent."""
        if db is not None:
            try:
                stmt = (
                    select(DocumentSource)
                    .order_by(DocumentSource.created_at.desc())
                    .offset(skip)
                    .limit(limit)
                )
                db_sources = list(db.execute(stmt).scalars().all())
                return [_db_to_source_status(s) for s in db_sources]
            except Exception as exc:
                logger.warning("Failed to list sources from database: %s", exc)

        all_sources = list(self._sources.values())
        return all_sources[skip : skip + limit]


_global_source_service = SourceService()


def get_source_service() -> SourceService:
    """FastAPI dependency yielding singleton SourceService."""
    return _global_source_service

