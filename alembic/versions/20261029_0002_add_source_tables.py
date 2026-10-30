"""add document_sources table and source_id to interview_sessions

Revision ID: 20261029_0002
Revises: 20261016_0001
Create Date: 2026-10-29 15:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '20261029_0002'
down_revision: Union[str, None] = '20261016_0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create document_sources table
    op.create_table(
        'document_sources',
        sa.Column('id', sa.String(length=64), nullable=False, comment='Unique source identifier'),
        sa.Column('source_type', sa.String(length=50), nullable=False, server_default='document', comment="Source type: 'document', 'url', or 'github'"),
        sa.Column('title', sa.String(length=255), nullable=False, comment='Title or filename of the source'),
        sa.Column('url', sa.Text(), nullable=True, comment='Canonical URL if source is a webpage or repository'),
        sa.Column('filename', sa.String(length=255), nullable=True, comment='Filename if source is an uploaded document'),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='completed', comment="Processing status: 'processing', 'completed', 'failed'"),
        sa.Column('character_count', sa.Integer(), nullable=False, server_default='0', comment='Total characters extracted from source'),
        sa.Column('chunk_count', sa.Integer(), nullable=False, server_default='0', comment='Total chunks indexed into Pinecone'),
        sa.Column('vector_count', sa.Integer(), nullable=False, server_default='0', comment='Total vectors upserted to Pinecone'),
        sa.Column('error_message', sa.Text(), nullable=True, comment='Error message if processing failed'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False, comment='Timestamp when source was registered'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False, comment='Timestamp of last update'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_document_sources_id'), 'document_sources', ['id'], unique=False)
    op.create_index(op.f('ix_document_sources_status'), 'document_sources', ['status'], unique=False)

    # 2. Add source_id column to interview_sessions
    op.add_column(
        'interview_sessions',
        sa.Column('source_id', sa.String(length=255), nullable=True, comment='Source identifier linked to this session'),
    )
    op.create_index(op.f('ix_interview_sessions_source_id'), 'interview_sessions', ['source_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_interview_sessions_source_id'), table_name='interview_sessions')
    op.drop_column('interview_sessions', 'source_id')

    op.drop_index(op.f('ix_document_sources_status'), table_name='document_sources')
    op.drop_index(op.f('ix_document_sources_id'), table_name='document_sources')
    op.drop_table('document_sources')

