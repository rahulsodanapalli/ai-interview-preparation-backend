"""Repository for persisting and querying ProjectAnalysis records."""

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.project_explainer import ProjectAnalysis

logger = logging.getLogger(__name__)


def _safe_rollback(db: Session) -> None:
    """Attempt a rollback on a failed database operation without masking the original error."""
    try:
        db.rollback()
    except Exception as exc:
        logger.error("Failed to roll back database session: %s", type(exc).__name__)


class ProjectExplainerRepository:
    """Encapsulates SQLAlchemy CRUD operations for ProjectAnalysis records."""

    def create(
        self,
        db: Session,
        *,
        analysis_id: str,
        user_id: Optional[str],
        repository: str,
        repo_url: str,
        ref: Optional[str],
        source_id: str,
        status: str,
        progress_percent: int,
        current_step: str,
        metadata_json: Optional[Dict[str, Any]] = None,
    ) -> ProjectAnalysis:
        try:
            record = ProjectAnalysis(
                id=analysis_id,
                user_id=user_id,
                repository=repository,
                repo_url=repo_url,
                ref=ref,
                source_id=source_id,
                status=status,
                progress_percent=progress_percent,
                current_step=current_step,
                metadata_json=metadata_json,
            )
            db.add(record)
            db.flush()
            return record
        except Exception as exc:
            _safe_rollback(db)
            logger.error("Failed to create ProjectAnalysis '%s': %s", analysis_id, type(exc).__name__)
            raise

    def get_by_id(self, db: Session, analysis_id: str) -> Optional[ProjectAnalysis]:
        try:
            return db.get(ProjectAnalysis, analysis_id)
        except Exception as exc:
            logger.error("Database lookup for ProjectAnalysis '%s' failed: %s", analysis_id, type(exc).__name__)
            raise

    def find_active_for_repo(
        self,
        db: Session,
        *,
        repository: str,
        user_id: Optional[str],
        ref: Optional[str] = None,
    ) -> Optional[ProjectAnalysis]:
        try:
            stmt = select(ProjectAnalysis).where(
                ProjectAnalysis.repository == repository,
                ProjectAnalysis.status.in_(["pending", "ingesting", "analyzing"]),
            )
            if user_id is not None:
                stmt = stmt.where(ProjectAnalysis.user_id == user_id)
            else:
                stmt = stmt.where(ProjectAnalysis.user_id.is_(None))
            if ref is not None:
                stmt = stmt.where(ProjectAnalysis.ref == ref)
            else:
                stmt = stmt.where(ProjectAnalysis.ref.is_(None))
            stmt = stmt.order_by(ProjectAnalysis.created_at.desc()).limit(1)
            return db.execute(stmt).scalars().first()
        except Exception as exc:
            logger.error("Failed to query active ProjectAnalysis for '%s': %s", repository, type(exc).__name__)
            raise

    def find_latest_completed_for_repo(
        self,
        db: Session,
        *,
        repository: str,
        user_id: Optional[str],
        ref: Optional[str] = None,
    ) -> Optional[ProjectAnalysis]:
        try:
            stmt = select(ProjectAnalysis).where(
                ProjectAnalysis.repository == repository,
                ProjectAnalysis.status == "completed",
            )
            if user_id is not None:
                stmt = stmt.where(ProjectAnalysis.user_id == user_id)
            else:
                stmt = stmt.where(ProjectAnalysis.user_id.is_(None))
            if ref is not None:
                stmt = stmt.where(ProjectAnalysis.ref == ref)
            else:
                stmt = stmt.where(ProjectAnalysis.ref.is_(None))
            stmt = stmt.order_by(ProjectAnalysis.created_at.desc()).limit(1)
            return db.execute(stmt).scalars().first()
        except Exception as exc:
            logger.error("Failed to query completed ProjectAnalysis for '%s': %s", repository, type(exc).__name__)
            raise

    def update_progress(
        self,
        db: Session,
        analysis_id: str,
        *,
        status: Optional[str] = None,
        progress_percent: Optional[int] = None,
        current_step: Optional[str] = None,
        metadata_json: Optional[Dict[str, Any]] = None,
        ingestion_stats_json: Optional[Dict[str, Any]] = None,
    ) -> Optional[ProjectAnalysis]:
        try:
            record = db.get(ProjectAnalysis, analysis_id)
            if record is None:
                return None
            if status is not None:
                record.status = status
            if progress_percent is not None:
                record.progress_percent = progress_percent
            if current_step is not None:
                record.current_step = current_step[:120]
            if metadata_json is not None:
                record.metadata_json = metadata_json
            if ingestion_stats_json is not None:
                record.ingestion_stats_json = ingestion_stats_json
            db.flush()
            return record
        except Exception as exc:
            _safe_rollback(db)
            logger.error("Failed to update ProjectAnalysis '%s': %s", analysis_id, type(exc).__name__)
            raise

    def mark_completed(
        self,
        db: Session,
        analysis_id: str,
        *,
        metadata_json: Optional[Dict[str, Any]],
        ingestion_stats_json: Dict[str, Any],
        analysis_json: Dict[str, Any],
    ) -> Optional[ProjectAnalysis]:
        try:
            record = db.get(ProjectAnalysis, analysis_id)
            if record is None:
                return None
            record.status = "completed"
            record.progress_percent = 100
            record.current_step = "Analysis completed"
            record.metadata_json = metadata_json
            record.ingestion_stats_json = ingestion_stats_json
            record.analysis_json = analysis_json
            record.error_message = None
            db.flush()
            return record
        except Exception as exc:
            _safe_rollback(db)
            logger.error("Failed to mark ProjectAnalysis '%s' completed: %s", analysis_id, type(exc).__name__)
            raise

    def mark_failed(
        self,
        db: Session,
        analysis_id: str,
        *,
        error_message: str,
        metadata_json: Optional[Dict[str, Any]] = None,
        ingestion_stats_json: Optional[Dict[str, Any]] = None,
    ) -> Optional[ProjectAnalysis]:
        try:
            record = db.get(ProjectAnalysis, analysis_id)
            if record is None:
                return None
            record.status = "failed"
            record.current_step = "Analysis failed"
            record.error_message = error_message[:2000]
            if metadata_json is not None:
                record.metadata_json = metadata_json
            if ingestion_stats_json is not None:
                record.ingestion_stats_json = ingestion_stats_json
            db.flush()
            return record
        except Exception as exc:
            _safe_rollback(db)
            logger.error("Failed to mark ProjectAnalysis '%s' failed: %s", analysis_id, type(exc).__name__)
            raise

    def list_by_user(
        self,
        db: Session,
        *,
        user_id: Optional[str],
        limit: int = 20,
        skip: int = 0,
    ) -> List[ProjectAnalysis]:
        try:
            stmt = select(ProjectAnalysis)
            if user_id is not None:
                stmt = stmt.where(ProjectAnalysis.user_id == user_id)
            stmt = stmt.order_by(ProjectAnalysis.created_at.desc()).offset(skip).limit(limit)
            return list(db.execute(stmt).scalars().all())
        except Exception as exc:
            logger.error("Failed to list ProjectAnalysis records: %s", type(exc).__name__)
            raise

