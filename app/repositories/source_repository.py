"""Repository for persisting and querying DocumentSource records."""

import logging
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.interview import DocumentSource

logger = logging.getLogger(__name__)


def _safe_rollback(db: Session) -> None:
    """Attempt a rollback on a failed database operation without masking the original error."""
    try:
        db.rollback()
    except Exception:
        pass


class SourceRepository:
    """Encapsulates SQLAlchemy CRUD operations for DocumentSource records."""

    def create(
        self,
        db: Session,
        *,
        source_id: str,
        source_type: str,
        title: str,
        url: Optional[str],
        filename: Optional[str],
        status: str,
        character_count: int,
        chunk_count: int,
        vector_count: int,
        error_message: Optional[str],
    ) -> None:
        """Persist a new DocumentSource record."""
        try:
            db_source = DocumentSource(
                id=source_id,
                source_type=source_type,
                title=title[:255],
                url=url,
                filename=filename[:255] if filename else None,
                status=status,
                character_count=character_count,
                chunk_count=chunk_count,
                vector_count=vector_count,
                error_message=error_message,
            )
            db.add(db_source)
            db.flush()
            logger.info("Persisted source record '%s' to database.", source_id)
        except Exception as exc:
            _safe_rollback(db)
            logger.warning("Failed to persist source '%s' to database: %s", source_id, exc)

    def update(
        self,
        db: Session,
        *,
        source_id: str,
        status: Optional[str] = None,
        character_count: Optional[int] = None,
        chunk_count: Optional[int] = None,
        vector_count: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update metrics and status for an existing DocumentSource record."""
        try:
            src_db = db.get(DocumentSource, source_id)
            if src_db:
                if status is not None:
                    src_db.status = status
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
            _safe_rollback(db)
            logger.warning("Failed to update database source '%s': %s", source_id, exc)

    def get_by_id(self, db: Session, source_id: str) -> Optional[DocumentSource]:
        """Retrieve a DocumentSource by its ID, or None if missing/failed."""
        try:
            return db.get(DocumentSource, source_id)
        except Exception as exc:
            logger.debug("Database lookup for source '%s' failed: %s", source_id, exc)
            return None

    def list_recent(
        self,
        db: Session,
        *,
        limit: int = 50,
        skip: int = 0,
    ) -> Optional[List[DocumentSource]]:
        """List DocumentSource records ordered by creation timestamp descending."""
        try:
            stmt = (
                select(DocumentSource)
                .order_by(DocumentSource.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
            return list(db.execute(stmt).scalars().all())
        except Exception as exc:
            logger.warning("Failed to list sources from database: %s", exc)
            return None

