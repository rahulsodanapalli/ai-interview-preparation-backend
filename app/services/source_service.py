"""Source Management Service for document and URL metadata tracking and persistence."""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from app.models.interview import DocumentSource
from app.repositories.source_repository import SourceRepository
from app.schemas.document import SourceProcessingStatus, SourceStatusResponse, SourceType
from app.utils.identifiers import enum_to_value

logger = logging.getLogger(__name__)


class SourceServiceError(Exception):
    """Base exception for source service errors."""


class SourceNotFoundError(SourceServiceError):
    """Raised when a requested source ID is not found."""


class SourceNotReadyError(SourceServiceError):
    """Raised when an operation requires a completed source, but processing is pending or failed."""


def _db_to_source_status(src_db: DocumentSource) -> SourceStatusResponse:
    """Convert a DocumentSource database model into a SourceStatusResponse schema."""
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

    def __init__(self, repository: Optional[SourceRepository] = None) -> None:
        self._sources: Dict[str, SourceStatusResponse] = {}
        self._repository = repository or SourceRepository()

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
            self._repository.create(
                db,
                source_id=source_id,
                source_type=enum_to_value(source_type),
                title=title,
                url=url,
                filename=filename,
                status=enum_to_value(status),
                character_count=character_count,
                chunk_count=chunk_count,
                vector_count=vector_count,
                error_message=error_message,
            )

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
            self._repository.update(
                db,
                source_id=source_id,
                status=enum_to_value(status) if status is not None else None,
                character_count=character_count,
                chunk_count=chunk_count,
                vector_count=vector_count,
                error_message=error_message,
            )

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
            src_db = self._repository.get_by_id(db, clean_id)
            if src_db is not None:
                resp = _db_to_source_status(src_db)
                self._sources[clean_id] = resp
                return resp

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
            db_sources = self._repository.list_recent(db, limit=limit, skip=skip)
            if db_sources is not None:
                return [_db_to_source_status(s) for s in db_sources]

        all_sources = list(self._sources.values())
        return all_sources[skip : skip + limit]


_global_source_service = SourceService()


def get_source_service() -> SourceService:
    """FastAPI dependency yielding singleton SourceService."""
    return _global_source_service
