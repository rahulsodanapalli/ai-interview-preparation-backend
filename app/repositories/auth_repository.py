"""Repository for User, OAuthAccount, and RefreshToken persistence."""

from datetime import datetime, timezone
import logging
from typing import Optional
import uuid

from sqlalchemy import select, update
from sqlalchemy.orm import Session, selectinload

from app.models.auth import OAuthAccount, RefreshToken, User

logger = logging.getLogger(__name__)


class AuthRepository:
    """Encapsulates SQLAlchemy persistence for authentication entities."""

    def get_user_by_id(self, db: Session, user_id: str) -> Optional[User]:
        stmt = (
            select(User)
            .options(selectinload(User.oauth_accounts))
            .where(User.id == user_id)
        )
        return db.execute(stmt).scalars().first()

    def get_user_by_email(self, db: Session, email: str) -> Optional[User]:
        stmt = (
            select(User)
            .options(selectinload(User.oauth_accounts))
            .where(User.email == email)
        )
        return db.execute(stmt).scalars().first()

    def create_user(
        self,
        db: Session,
        *,
        name: str,
        email: str,
        password_hash: Optional[str],
        is_active: bool = True,
        email_verified: bool = False,
    ) -> User:
        user = User(
            id=str(uuid.uuid4()),
            name=name,
            email=email,
            password_hash=password_hash,
            is_active=is_active,
            email_verified=email_verified,
        )
        db.add(user)
        db.flush()
        return user

    def get_oauth_account(
        self,
        db: Session,
        *,
        provider: str,
        provider_user_id: str,
    ) -> Optional[OAuthAccount]:
        stmt = (
            select(OAuthAccount)
            .options(selectinload(OAuthAccount.user).selectinload(User.oauth_accounts))
            .where(
                OAuthAccount.provider == provider,
                OAuthAccount.provider_user_id == provider_user_id,
            )
        )
        return db.execute(stmt).scalars().first()

    def create_oauth_account(
        self,
        db: Session,
        *,
        user_id: str,
        provider: str,
        provider_user_id: str,
    ) -> OAuthAccount:
        oauth_acc = OAuthAccount(
            id=str(uuid.uuid4()),
            user_id=user_id,
            provider=provider,
            provider_user_id=provider_user_id,
        )
        db.add(oauth_acc)
        db.flush()
        return oauth_acc

    def create_refresh_token(
        self,
        db: Session,
        *,
        user_id: str,
        token_hash: str,
        family_id: str,
        expires_at: datetime,
        remember_me: bool = False,
    ) -> RefreshToken:
        rt = RefreshToken(
            id=str(uuid.uuid4()),
            user_id=user_id,
            token_hash=token_hash,
            family_id=family_id,
            expires_at=expires_at,
            remember_me=remember_me,
        )
        db.add(rt)
        db.flush()
        return rt

    def get_refresh_token_by_hash(
        self, db: Session, token_hash: str, *, for_update: bool = False
    ) -> Optional[RefreshToken]:
        stmt = select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        if for_update:
            stmt = stmt.with_for_update()
        return db.execute(stmt).scalars().first()

    def revoke_refresh_token(
        self,
        db: Session,
        token_record: RefreshToken,
        *,
        replaced_by_token_id: Optional[str] = None,
    ) -> None:
        token_record.revoked_at = datetime.now(timezone.utc)
        if replaced_by_token_id is not None:
            token_record.replaced_by_token_id = replaced_by_token_id
        db.flush()

    def revoke_token_family(self, db: Session, family_id: str) -> int:
        now = datetime.now(timezone.utc)
        stmt = (
            update(RefreshToken)
            .where(
                RefreshToken.family_id == family_id,
                RefreshToken.revoked_at.is_(None),
            )
            .values(revoked_at=now)
        )
        result = db.execute(stmt)
        db.flush()
        return int(result.rowcount or 0)

    def revoke_all_user_tokens(self, db: Session, user_id: str) -> int:
        now = datetime.now(timezone.utc)
        stmt = (
            update(RefreshToken)
            .where(
                RefreshToken.user_id == user_id,
                RefreshToken.revoked_at.is_(None),
            )
            .values(revoked_at=now)
        )
        result = db.execute(stmt)
        db.flush()
        return int(result.rowcount or 0)

