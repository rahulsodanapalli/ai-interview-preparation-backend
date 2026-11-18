"""Source Management Service for document and URL metadata tracking and persistence.

In production, PostgreSQL via SourceRepository is the single source of truth; the
in-memory store is strictly limited to non-production development/testing when no
database session is configured.
"""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional

from sqlalchemy import Connection, Engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.session import DatabaseConfigurationError, commit_session, is_production_env
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


class SourceForbiddenError(SourceServiceError):
    """Raised when a user attempts to access another user's document or URL source."""


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
        # Development/testing-only in-memory store used strictly when db is None in non-production.
        self._dev_sources: Dict[str, SourceStatusResponse] = {}
        self._dev_source_owners: Dict[str, Optional[str]] = {}
        self._repository = repository or SourceRepository()

    @property
    def _sources(self) -> Dict[str, SourceStatusResponse]:
        """Backward-compatible alias for development-only in-memory store."""
        return self._dev_sources

    @staticmethod
    def _require_db_in_production(db: Optional[Session]) -> None:
        """Fail fast if a database session is absent in production."""
        if db is None and is_production_env():
            raise DatabaseConfigurationError(
                "Database session is required in production. In-memory source fallback is disabled."
            )

    @staticmethod
    def _open_independent_session(db: Session) -> Optional[Session]:
        """Create an independent SQLAlchemy Session sharing the same Engine/Connection."""
        try:
            bind = db.get_bind()
        except Exception:
            return None
        if isinstance(bind, (Engine, Connection)):
            factory = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=bind,
                expire_on_commit=False,
            )
            return factory()
        return None

    def reset_store(self) -> None:
        """Clear the development-only in-memory store for test isolation."""
        self._dev_sources.clear()
        self._dev_source_owners.clear()

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
        commit_immediately: bool = False,
        user_id: Optional[str] = None,
    ) -> SourceStatusResponse:
        """Register a new source in PostgreSQL (or dev-only memory store when db is None)."""
        self._require_db_in_production(db)

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

        if db is not None:
            if commit_immediately:
                ind_db = self._open_independent_session(db)
                if ind_db is not None:
                    try:
                        db_source = self._repository.create(
                            ind_db,
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
                            user_id=user_id,
                        )
                        ind_db.commit()
                        return _db_to_source_status(db_source)
                    except Exception:
                        try:
                            ind_db.rollback()
                        except Exception:
                            pass
                        raise
                    finally:
                        ind_db.close()

            db_source = self._repository.create(
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
                user_id=user_id,
            )
            commit_session(db)
            if isinstance(db_source, DocumentSource):
                return _db_to_source_status(db_source)
            return source_resp

        self._dev_sources[source_id] = source_resp
        self._dev_source_owners[source_id] = user_id
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
        self._require_db_in_production(db)

        if db is not None:
            src_db = self._repository.get_by_id(db, source_id)
            if src_db is not None:
                updated_db = self._repository.update(
                    db,
                    source_id=source_id,
                    status=enum_to_value(status) if status is not None else None,
                    character_count=character_count,
                    chunk_count=chunk_count,
                    vector_count=vector_count,
                    error_message=error_message,
                )
                target_db = updated_db or src_db
                resp = _db_to_source_status(target_db)
                commit_session(db)
                return resp

            if not is_production_env() and source_id in self._dev_sources:
                return self._update_dev_source(
                    source_id=source_id,
                    status=status,
                    character_count=character_count,
                    chunk_count=chunk_count,
                    vector_count=vector_count,
                    error_message=error_message,
                )

            raise SourceNotFoundError(f"Source with ID '{source_id}' was not found.")

        return self._update_dev_source(
            source_id=source_id,
            status=status,
            character_count=character_count,
            chunk_count=chunk_count,
            vector_count=vector_count,
            error_message=error_message,
        )

    def _update_dev_source(
        self,
        *,
        source_id: str,
        status: Optional[SourceProcessingStatus] = None,
        character_count: Optional[int] = None,
        chunk_count: Optional[int] = None,
        vector_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> SourceStatusResponse:
        """Update a source in the development-only in-memory store."""
        source = self._dev_sources.get(source_id)
        if source is None:
            raise SourceNotFoundError(f"Source with ID '{source_id}' was not found.")

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
        self._dev_sources[source_id] = new_source
        return new_source

    def mark_source_failed(
        self,
        source_id: str,
        error_message: str,
        *,
        db: Optional[Session] = None,
        source_type: Optional[SourceType] = None,
        title: Optional[str] = None,
        url: Optional[str] = None,
        filename: Optional[str] = None,
        character_count: int = 0,
        user_id: Optional[str] = None,
    ) -> None:
        """Persist FAILED status in an independent transaction so outer rollbacks never erase it."""
        self._require_db_in_production(db)
        safe_error = (error_message or "Source processing failed.")[:2000]

        if db is not None:
            # Release any uncommitted transaction state/locks on the request session first.
            try:
                db.rollback()
            except Exception:
                pass

            ind_db = self._open_independent_session(db)
            if ind_db is not None:
                try:
                    existing = self._repository.get_by_id(ind_db, source_id)
                    if existing is not None:
                        self._repository.update(
                            ind_db,
                            source_id=source_id,
                            status=SourceProcessingStatus.FAILED.value,
                            error_message=safe_error,
                        )
                    elif source_type is not None and title is not None:
                        self._repository.create(
                            ind_db,
                            source_id=source_id,
                            source_type=enum_to_value(source_type),
                            title=title,
                            url=url,
                            filename=filename,
                            status=SourceProcessingStatus.FAILED.value,
                            character_count=character_count,
                            chunk_count=0,
                            vector_count=0,
                            error_message=safe_error,
                            user_id=user_id,
                        )
                    ind_db.commit()
                except Exception as persist_exc:
                    try:
                        ind_db.rollback()
                    except Exception:
                        pass
                    logger.error(
                        "Failed to persist FAILED status for source '%s': %s",
                        source_id,
                        type(persist_exc).__name__,
                    )
                finally:
                    ind_db.close()
            else:
                try:
                    self._repository.update(
                        db,
                        source_id=source_id,
                        status=SourceProcessingStatus.FAILED.value,
                        error_message=safe_error,
                    )
                    if callable(getattr(db, "commit", None)):
                        db.commit()
                except Exception as persist_exc:
                    logger.error(
                        "Failed to persist FAILED status for source '%s': %s",
                        source_id,
                        type(persist_exc).__name__,
                    )

        if not is_production_env() and (db is None or source_id in self._dev_sources):
            if source_id in self._dev_sources:
                self._update_dev_source(
                    source_id=source_id,
                    status=SourceProcessingStatus.FAILED,
                    error_message=safe_error,
                )
            elif db is None and source_type is not None and title is not None:
                self._dev_sources[source_id] = SourceStatusResponse(
                    source_id=source_id,
                    status=SourceProcessingStatus.FAILED,
                    source_type=source_type,
                    title=title,
                    url=url,
                    filename=filename,
                    character_count=character_count,
                    chunk_count=0,
                    vector_count=0,
                    error_message=safe_error,
                    created_at=datetime.now(timezone.utc).isoformat(),
                )

    def get_source(
        self,
        source_id: str,
        db: Optional[Session] = None,
        user_id: Optional[str] = None,
    ) -> SourceStatusResponse:
        """Retrieve source status and metadata from PostgreSQL (or dev-only memory store), enforcing user ownership."""
        self._require_db_in_production(db)

        clean_id = (source_id or "").strip()
        if not clean_id:
            raise SourceNotFoundError("source_id must not be empty.")

        if db is not None:
            src_db = self._repository.get_by_id(db, clean_id)
            if src_db is not None:
                if user_id is not None and src_db.user_id != user_id:
                    logger.warning(
                        "Authorization denied: user '%s' attempted access to source '%s'.",
                        user_id,
                        clean_id,
                    )
                    raise SourceForbiddenError("You do not have permission to access this source.")
                return _db_to_source_status(src_db)
            if not is_production_env() and clean_id in self._dev_sources:
                if user_id is not None and self._dev_source_owners.get(clean_id) != user_id:
                    raise SourceForbiddenError("You do not have permission to access this source.")
                return self._dev_sources[clean_id]
            raise SourceNotFoundError(f"Source with ID '{clean_id}' was not found.")

        if clean_id in self._dev_sources:
            if user_id is not None and self._dev_source_owners.get(clean_id) != user_id:
                raise SourceForbiddenError("You do not have permission to access this source.")
            return self._dev_sources[clean_id]

        raise SourceNotFoundError(f"Source with ID '{clean_id}' was not found.")

    def list_sources(
        self,
        db: Optional[Session] = None,
        limit: int = 50,
        skip: int = 0,
        user_id: Optional[str] = None,
    ) -> List[SourceStatusResponse]:
        """List registered sources ordered by most recent, scoped by user_id when provided."""
        self._require_db_in_production(db)

        if db is not None:
            db_sources = self._repository.list_recent(db, user_id=user_id, limit=limit, skip=skip)
            return [_db_to_source_status(s) for s in db_sources]

        all_sources = [
            s
            for s in self._dev_sources.values()
            if user_id is None or self._dev_source_owners.get(s.source_id) == user_id
        ]
        return all_sources[skip : skip + limit]


_global_source_service = SourceService()


def get_source_service() -> SourceService:
    """FastAPI dependency yielding singleton SourceService."""
    return _global_source_service
