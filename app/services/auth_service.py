"""Authentication service handling registration, login, token rotation, logout, and OAuth linking."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import secrets
from typing import Optional
import uuid

from fastapi import Depends
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import (
    TokenValidationError,
    create_access_token,
    create_oauth_state_token,
    generate_csrf_token,
    generate_refresh_token,
    hash_password,
    hash_token,
    normalize_email,
    password_needs_rehash,
    verify_oauth_state_token,
    verify_password,
)
from app.db.session import commit_session
from app.models.auth import User
from app.repositories.auth_repository import AuthRepository
from app.schemas.auth import AuthResponse, LoginRequest, RegisterRequest, UserResponse
from app.services.google_oauth_service import GoogleOAuthService, get_google_oauth_service

logger = logging.getLogger(__name__)


class AuthError(Exception):
    """Base authentication exception."""


class EmailAlreadyExistsError(AuthError):
    """Raised when registering with an already-used email address."""


class InvalidCredentialsError(AuthError):
    """Raised when email/password authentication fails."""


class InactiveUserError(AuthError):
    """Raised when an inactive user attempts to authenticate or refresh."""


class InvalidRefreshTokenError(AuthError):
    """Raised when a refresh token is missing, invalid, expired, or revoked."""


class UnsafeAccountLinkingError(AuthError):
    """Raised when OAuth account linking would risk account takeover."""


@dataclass
class AuthResult:
    """Internal result carrying both API AuthResponse and raw refresh token for HttpOnly cookie."""

    response: AuthResponse
    refresh_token: str
    refresh_max_age_seconds: int


def user_to_response(user: User) -> UserResponse:
    """Convert a User ORM instance to a safe API UserResponse."""
    providers: list[str] = []
    if user.password_hash:
        providers.append("password")
    for oa in getattr(user, "oauth_accounts", []) or []:
        if oa.provider not in providers:
            providers.append(oa.provider)

    created_str = (
        user.created_at.isoformat()
        if user.created_at
        else datetime.now(timezone.utc).isoformat()
    )
    return UserResponse(
        id=user.id,
        name=user.name,
        email=user.email,
        is_active=user.is_active,
        email_verified=user.email_verified,
        auth_providers=providers,
        created_at=created_str,
    )


class AuthService:
    """Coordinates user lifecycle, Argon2id hashing, JWT/refresh token issuance, and Google OAuth."""

    def __init__(
        self,
        repository: Optional[AuthRepository] = None,
        google_service: Optional[GoogleOAuthService] = None,
    ) -> None:
        self._repo = repository or AuthRepository()
        self._google = google_service or get_google_oauth_service()

    def _issue_tokens(
        self,
        db: Session,
        *,
        user: User,
        remember_me: bool = False,
        family_id: Optional[str] = None,
    ) -> AuthResult:
        """Issue a new access token, persisted hashed refresh token, and CSRF token."""
        access_token, expires_in = create_access_token(
            user_id=user.id,
            email=user.email,
        )
        raw_refresh = generate_refresh_token()
        token_hash = hash_token(raw_refresh)
        ttl_days = (
            settings.JWT_REFRESH_REMEMBER_ME_DAYS
            if remember_me
            else settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS
        )
        expires_at = datetime.now(timezone.utc) + timedelta(days=ttl_days)
        effective_family = family_id or str(uuid.uuid4())

        rt_record = self._repo.create_refresh_token(
            db,
            user_id=user.id,
            token_hash=token_hash,
            family_id=effective_family,
            expires_at=expires_at,
            remember_me=remember_me,
        )
        csrf_token = generate_csrf_token()
        max_age = ttl_days * 24 * 60 * 60

        return AuthResult(
            response=AuthResponse(
                access_token=access_token,
                token_type="bearer",
                expires_in=expires_in,
                csrf_token=csrf_token,
                user=user_to_response(user),
            ),
            refresh_token=raw_refresh,
            refresh_max_age_seconds=max_age,
        ), rt_record

    def register(self, db: Session, payload: RegisterRequest) -> AuthResult:
        """Register a new email/password account with Argon2id hashing."""
        email = normalize_email(payload.email)
        existing = self._repo.get_user_by_email(db, email)
        if existing is not None:
            raise EmailAlreadyExistsError("An account with this email already exists.")

        pw_hash = hash_password(payload.password)
        try:
            user = self._repo.create_user(
                db,
                name=payload.name,
                email=email,
                password_hash=pw_hash,
                is_active=True,
                email_verified=False,
            )
            auth_result, _ = self._issue_tokens(db, user=user, remember_me=False)
            commit_session(db)
        except IntegrityError as exc:
            db.rollback()
            raise EmailAlreadyExistsError("An account with this email already exists.") from exc

        logger.info("Registered new user account id=%s.", user.id)
        return auth_result

    def login(self, db: Session, payload: LoginRequest) -> AuthResult:
        """Authenticate a user via email and password with generic error responses."""
        email = normalize_email(payload.email)
        user = self._repo.get_user_by_email(db, email)

        password_valid = verify_password(
            payload.password,
            user.password_hash if user is not None else None,
        )
        if user is None or not password_valid:
            raise InvalidCredentialsError("Invalid email or password.")

        if not user.is_active:
            raise InactiveUserError("This account is inactive.")

        if user.password_hash and password_needs_rehash(user.password_hash):
            user.password_hash = hash_password(payload.password)

        auth_result, _ = self._issue_tokens(
            db,
            user=user,
            remember_me=payload.remember_me,
        )
        commit_session(db)
        logger.info("User authenticated via password id=%s.", user.id)
        return auth_result

    def refresh(self, db: Session, raw_refresh_token: Optional[str]) -> AuthResult:
        """Rotate a refresh token, detecting reuse of revoked tokens and revoking the family."""
        if not raw_refresh_token or not raw_refresh_token.strip():
            raise InvalidRefreshTokenError("Refresh token is missing.")

        token_hash = hash_token(raw_refresh_token.strip())
        existing = self._repo.get_refresh_token_by_hash(db, token_hash, for_update=True)
        if existing is None:
            raise InvalidRefreshTokenError("Invalid refresh token.")

        if existing.revoked_at is not None:
            # Token reuse detected: revoke entire family to protect against token theft
            revoked_count = self._repo.revoke_token_family(db, existing.family_id)
            commit_session(db)
            logger.warning(
                "Refresh token reuse detected for user_id=%s family_id=%s; revoked %d active token(s).",
                existing.user_id,
                existing.family_id,
                revoked_count,
            )
            raise InvalidRefreshTokenError("Refresh token has been revoked.")

        expires_at = existing.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= datetime.now(timezone.utc):
            self._repo.revoke_refresh_token(db, existing)
            commit_session(db)
            raise InvalidRefreshTokenError("Refresh token has expired.")

        user = self._repo.get_user_by_id(db, existing.user_id)
        if user is None:
            raise InvalidRefreshTokenError("User associated with refresh token no longer exists.")
        if not user.is_active:
            self._repo.revoke_token_family(db, existing.family_id)
            commit_session(db)
            raise InactiveUserError("This account is inactive.")

        auth_result, new_record = self._issue_tokens(
            db,
            user=user,
            remember_me=existing.remember_me,
            family_id=existing.family_id,
        )
        self._repo.revoke_refresh_token(
            db,
            existing,
            replaced_by_token_id=new_record.id,
        )
        commit_session(db)
        return auth_result

    def logout(self, db: Session, raw_refresh_token: Optional[str]) -> None:
        """Revoke the provided refresh token (and its family) on logout."""
        if not raw_refresh_token or not raw_refresh_token.strip():
            return

        token_hash = hash_token(raw_refresh_token.strip())
        existing = self._repo.get_refresh_token_by_hash(db, token_hash, for_update=True)
        if existing is not None:
            self._repo.revoke_token_family(db, existing.family_id)
            commit_session(db)
            logger.info("Revoked refresh token family on logout for user_id=%s.", existing.user_id)

    def create_google_authorization(self) -> tuple[str, str, str]:
        """Generate Google OAuth authorization URL, signed state token, and nonce."""
        _cid, _secret, redirect_uri = self._google.ensure_configured()
        nonce = uuid.uuid4().hex
        state = create_oauth_state_token(nonce=nonce, redirect_uri=redirect_uri)
        auth_url = self._google.build_authorization_url(state=state, nonce=nonce)
        return auth_url, state, nonce

    async def complete_google_oauth(
        self,
        db: Session,
        *,
        code: str,
        state: str,
        cookie_nonce: Optional[str] = None,
    ) -> AuthResult:
        """Verify OAuth state, exchange Google auth code, safely link/create user, and issue tokens."""
        _cid, _secret, redirect_uri = self._google.ensure_configured()
        cleaned_code = (code or "").strip()
        cleaned_state = (state or "").strip()
        if not cleaned_code:
            raise InvalidCredentialsError("Google authorization code is missing.")
        if not cleaned_state:
            raise InvalidCredentialsError("OAuth state parameter is missing.")

        try:
            state_claims = verify_oauth_state_token(cleaned_state, expected_redirect_uri=redirect_uri)
        except TokenValidationError as exc:
            raise InvalidCredentialsError(str(exc)) from exc

        expected_nonce = str(state_claims.get("nonce") or "").strip()
        if not expected_nonce:
            raise InvalidCredentialsError("OAuth state validation failed.")
        if cookie_nonce is not None:
            cleaned_cookie = cookie_nonce.strip()
            if not cleaned_cookie or not secrets.compare_digest(cleaned_cookie, expected_nonce):
                raise InvalidCredentialsError("OAuth state validation failed.")

        identity = await self._google.authenticate_code(
            code=cleaned_code,
            expected_nonce=expected_nonce,
        )

        # 1. Check for existing OAuth account link
        oauth_account = self._repo.get_oauth_account(
            db,
            provider="google",
            provider_user_id=identity.sub,
        )
        if oauth_account is not None:
            user = oauth_account.user
            if not user.is_active:
                raise InactiveUserError("This account is inactive.")
            auth_result, _ = self._issue_tokens(db, user=user, remember_me=True)
            commit_session(db)
            logger.info("User authenticated via Google OAuth id=%s.", user.id)
            return auth_result

        # 2. Check for existing user with matching normalized email
        existing_user = self._repo.get_user_by_email(db, identity.email)
        if existing_user is not None:
            if not existing_user.is_active:
                raise InactiveUserError("This account is inactive.")

            # Prevent account takeover: never link Google onto an unverified local account
            if not existing_user.email_verified:
                raise UnsafeAccountLinkingError(
                    "An unverified account with this email already exists. "
                    "Please sign in with your password and verify your email before linking Google."
                )

            # Prevent linking if user already has a different Google subject bound
            for existing_oa in existing_user.oauth_accounts:
                if existing_oa.provider == "google" and existing_oa.provider_user_id != identity.sub:
                    raise UnsafeAccountLinkingError(
                        "This account is already linked to a different Google identity."
                    )

            try:
                self._repo.create_oauth_account(
                    db,
                    user_id=existing_user.id,
                    provider="google",
                    provider_user_id=identity.sub,
                )
            except IntegrityError as exc:
                db.rollback()
                linked = self._repo.get_oauth_account(
                    db, provider="google", provider_user_id=identity.sub
                )
                if linked is None or linked.user_id != existing_user.id:
                    raise UnsafeAccountLinkingError(
                        "This account is already linked to a different Google identity."
                    ) from exc
            db.refresh(existing_user)
            user = existing_user
            auth_result, _ = self._issue_tokens(db, user=user, remember_me=True)
            commit_session(db)
            logger.info("Safely linked Google OAuth to verified user id=%s.", user.id)
            return auth_result

        # 3. Create new verified Google user + OAuth link
        try:
            user = self._repo.create_user(
                db,
                name=identity.name,
                email=identity.email,
                password_hash=None,
                is_active=True,
                email_verified=True,
            )
            self._repo.create_oauth_account(
                db,
                user_id=user.id,
                provider="google",
                provider_user_id=identity.sub,
            )
            db.refresh(user)
            auth_result, _ = self._issue_tokens(db, user=user, remember_me=True)
            commit_session(db)
            logger.info("Created new user via Google OAuth id=%s.", user.id)
            return auth_result
        except IntegrityError as exc:
            db.rollback()
            raced_oauth = self._repo.get_oauth_account(
                db, provider="google", provider_user_id=identity.sub
            )
            if raced_oauth is not None:
                if not raced_oauth.user.is_active:
                    raise InactiveUserError("This account is inactive.") from exc
                auth_result, _ = self._issue_tokens(db, user=raced_oauth.user, remember_me=True)
                commit_session(db)
                return auth_result
            raise UnsafeAccountLinkingError(
                "An account with this email already exists and cannot be linked automatically."
            ) from exc


def get_auth_service(
    google_service: GoogleOAuthService = Depends(get_google_oauth_service),
) -> AuthService:
    """FastAPI dependency provider for AuthService."""
    return AuthService(google_service=google_service)
