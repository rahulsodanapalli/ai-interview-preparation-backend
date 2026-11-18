"""add authentication tables and user ownership columns

Revision ID: 20261110_0003
Revises: 20261029_0002
Create Date: 2026-11-10 16:06:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "20261110_0003"
down_revision: Union[str, None] = "20261029_0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create users table
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), nullable=False, comment="Unique UUID user identifier"),
        sa.Column("name", sa.String(length=120), nullable=False, comment="User display name"),
        sa.Column("email", sa.String(length=255), nullable=False, comment="Normalized lowercase email address"),
        sa.Column("password_hash", sa.String(length=255), nullable=True, comment="Argon2id password hash (null for OAuth-only users)"),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False, comment="Whether the user account is active"),
        sa.Column("email_verified", sa.Boolean(), server_default=sa.false(), nullable=False, comment="Whether the user's email address has been verified"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False, comment="Timestamp when account was created"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False, comment="Timestamp of last account update"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index(op.f("ix_users_id"), "users", ["id"], unique=False)
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    # 2. Create oauth_accounts table
    op.create_table(
        "oauth_accounts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("provider", sa.String(length=50), nullable=False, comment="OAuth provider identifier (e.g., 'google')"),
        sa.Column("provider_user_id", sa.String(length=255), nullable=False, comment="Immutable unique subject identifier from the provider"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider", "provider_user_id", name="uq_oauth_provider_user"),
        sa.UniqueConstraint("user_id", "provider", name="uq_oauth_user_provider"),
    )
    op.create_index(op.f("ix_oauth_accounts_id"), "oauth_accounts", ["id"], unique=False)
    op.create_index(op.f("ix_oauth_accounts_user_id"), "oauth_accounts", ["user_id"], unique=False)
    op.create_index("ix_oauth_provider_user_id", "oauth_accounts", ["provider", "provider_user_id"], unique=False)

    # 3. Create refresh_tokens table
    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False, comment="SHA-256 hex digest of the opaque refresh token"),
        sa.Column("family_id", sa.String(length=36), nullable=False, comment="Token rotation family identifier for reuse detection"),
        sa.Column("replaced_by_token_id", sa.String(length=36), nullable=True, comment="ID of the replacement refresh token created on rotation"),
        sa.Column("remember_me", sa.Boolean(), server_default=sa.false(), nullable=False, comment="Whether the session uses extended remember-me refresh lifetime"),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_hash", name="uq_refresh_tokens_token_hash"),
    )
    op.create_index(op.f("ix_refresh_tokens_id"), "refresh_tokens", ["id"], unique=False)
    op.create_index(op.f("ix_refresh_tokens_user_id"), "refresh_tokens", ["user_id"], unique=False)
    op.create_index(op.f("ix_refresh_tokens_token_hash"), "refresh_tokens", ["token_hash"], unique=True)
    op.create_index(op.f("ix_refresh_tokens_family_id"), "refresh_tokens", ["family_id"], unique=False)
    op.create_index(op.f("ix_refresh_tokens_expires_at"), "refresh_tokens", ["expires_at"], unique=False)

    # 4. Add user_id ownership column to interview_sessions
    with op.batch_alter_table("interview_sessions") as batch_op:
        batch_op.add_column(
            sa.Column("user_id", sa.String(length=36), nullable=True, comment="Owner user UUID")
        )
        batch_op.create_index(
            op.f("ix_interview_sessions_user_id"), ["user_id"], unique=False
        )
        batch_op.create_foreign_key(
            "fk_interview_sessions_user_id_users",
            "users",
            ["user_id"],
            ["id"],
            ondelete="CASCADE",
        )

    # 5. Add user_id ownership column to document_sources
    with op.batch_alter_table("document_sources") as batch_op:
        batch_op.add_column(
            sa.Column("user_id", sa.String(length=36), nullable=True, comment="Owner user UUID")
        )
        batch_op.create_index(
            op.f("ix_document_sources_user_id"), ["user_id"], unique=False
        )
        batch_op.create_foreign_key(
            "fk_document_sources_user_id_users",
            "users",
            ["user_id"],
            ["id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    with op.batch_alter_table("document_sources") as batch_op:
        batch_op.drop_constraint("fk_document_sources_user_id_users", type_="foreignkey")
        batch_op.drop_index(op.f("ix_document_sources_user_id"))
        batch_op.drop_column("user_id")

    with op.batch_alter_table("interview_sessions") as batch_op:
        batch_op.drop_constraint("fk_interview_sessions_user_id_users", type_="foreignkey")
        batch_op.drop_index(op.f("ix_interview_sessions_user_id"))
        batch_op.drop_column("user_id")

    op.drop_index(op.f("ix_refresh_tokens_expires_at"), table_name="refresh_tokens")
    op.drop_index(op.f("ix_refresh_tokens_family_id"), table_name="refresh_tokens")
    op.drop_index(op.f("ix_refresh_tokens_token_hash"), table_name="refresh_tokens")
    op.drop_index(op.f("ix_refresh_tokens_user_id"), table_name="refresh_tokens")
    op.drop_index(op.f("ix_refresh_tokens_id"), table_name="refresh_tokens")
    op.drop_table("refresh_tokens")

    op.drop_index("ix_oauth_provider_user_id", table_name="oauth_accounts")
    op.drop_index(op.f("ix_oauth_accounts_user_id"), table_name="oauth_accounts")
    op.drop_index(op.f("ix_oauth_accounts_id"), table_name="oauth_accounts")
    op.drop_table("oauth_accounts")

    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_index(op.f("ix_users_id"), table_name="users")
    op.drop_table("users")

