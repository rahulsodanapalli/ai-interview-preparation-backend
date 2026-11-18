"""SQLAlchemy ORM model for GitHub Project Explainer analyses and progress tracking."""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.db.base import Base


class ProjectAnalysis(Base):
    """Persisted repository analysis job, ingestion metrics, and structured project explanation."""

    __tablename__ = "project_analyses"

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: f"proj_{uuid.uuid4().hex[:16]}",
        index=True,
        comment="Unique project analysis identifier",
    )
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
        comment="Owner user UUID",
    )
    repository: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Normalized GitHub repository full name (owner/repo)",
    )
    repo_url: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Canonical GitHub repository URL",
    )
    ref: Mapped[Optional[str]] = mapped_column(
        String(120),
        nullable=True,
        comment="Optional Git branch, tag, or commit SHA",
    )
    source_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="Deterministic Pinecone vector scope identifier for this repository and user",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
        index=True,
        comment="Lifecycle status: pending, ingesting, analyzing, completed, failed",
    )
    progress_percent: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Execution progress percentage (0-100)",
    )
    current_step: Mapped[str] = mapped_column(
        String(120),
        nullable=False,
        default="Queued for analysis",
        comment="Human-readable description of the current pipeline stage",
    )
    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Validated GitHub repository metadata snapshot",
    )
    ingestion_stats_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Ingestion statistics, analyzed file paths, and skipped file summary",
    )
    analysis_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Structured 9-section project explanation generated from repository evidence",
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Sanitized error message if analysis failed",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Timestamp when analysis was created",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="Timestamp of last analysis update",
    )

    def __repr__(self) -> str:
        return (
            f"<ProjectAnalysis(id='{self.id}', repository='{self.repository}', "
            f"status='{self.status}', progress={self.progress_percent})>"
        )

