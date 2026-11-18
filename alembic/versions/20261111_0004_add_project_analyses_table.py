"""add project_analyses table for Project Explainer feature

Revision ID: 20261111_0004
Revises: 20261110_0003
Create Date: 2026-11-11 15:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20261111_0004"
down_revision: Union[str, None] = "20261110_0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "project_analyses",
        sa.Column("id", sa.String(length=64), nullable=False, comment="Unique project analysis identifier"),
        sa.Column("user_id", sa.String(length=36), nullable=True, comment="Owner user UUID"),
        sa.Column("repository", sa.String(length=255), nullable=False, comment="Normalized GitHub repository full name (owner/repo)"),
        sa.Column("repo_url", sa.Text(), nullable=False, comment="Canonical GitHub repository URL"),
        sa.Column("ref", sa.String(length=120), nullable=True, comment="Optional Git branch, tag, or commit SHA"),
        sa.Column("source_id", sa.String(length=64), nullable=False, comment="Deterministic Pinecone vector scope identifier for this repository and user"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending", comment="Lifecycle status: pending, ingesting, analyzing, completed, failed"),
        sa.Column("progress_percent", sa.Integer(), nullable=False, server_default="0", comment="Execution progress percentage (0-100)"),
        sa.Column("current_step", sa.String(length=120), nullable=False, server_default="Queued for analysis", comment="Human-readable description of the current pipeline stage"),
        sa.Column("metadata_json", sa.JSON(), nullable=True, comment="Validated GitHub repository metadata snapshot"),
        sa.Column("ingestion_stats_json", sa.JSON(), nullable=True, comment="Ingestion statistics, analyzed file paths, and skipped file summary"),
        sa.Column("analysis_json", sa.JSON(), nullable=True, comment="Structured 9-section project explanation generated from repository evidence"),
        sa.Column("error_message", sa.Text(), nullable=True, comment="Sanitized error message if analysis failed"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False, comment="Timestamp when analysis was created"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False, comment="Timestamp of last analysis update"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_project_analyses_id"), "project_analyses", ["id"], unique=False)
    op.create_index(op.f("ix_project_analyses_user_id"), "project_analyses", ["user_id"], unique=False)
    op.create_index(op.f("ix_project_analyses_repository"), "project_analyses", ["repository"], unique=False)
    op.create_index(op.f("ix_project_analyses_source_id"), "project_analyses", ["source_id"], unique=False)
    op.create_index(op.f("ix_project_analyses_status"), "project_analyses", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_project_analyses_status"), table_name="project_analyses")
    op.drop_index(op.f("ix_project_analyses_source_id"), table_name="project_analyses")
    op.drop_index(op.f("ix_project_analyses_repository"), table_name="project_analyses")
    op.drop_index(op.f("ix_project_analyses_user_id"), table_name="project_analyses")
    op.drop_index(op.f("ix_project_analyses_id"), table_name="project_analyses")
    op.drop_table("project_analyses")

