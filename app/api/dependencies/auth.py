"""Reusable FastAPI authentication, authorization, and CSRF dependencies."""

import logging
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import TokenValidationError, decode_access_token, verify_csrf_tokens
from app.db.session import get_db_optional
from app.models.auth import User
from app.repositories.auth_repository import AuthRepository

logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)
_auth_repo = AuthRepository()


def _resolve_user_from_token(token: str, db: Optional[Session]) -> User:
    """Validate a JWT access token and load the corresponding User from PostgreSQL."""
    try:
        payload = decode_access_token(token)
    except TokenValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user_id = str(payload.get("sub") or "").strip()
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token payload.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is required for authentication.",
        )

    user = _auth_repo.get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authenticated user account no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


async def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    db: Optional[Session] = Depends(get_db_optional),
) -> Optional[User]:
    """Return the authenticated User if a valid Bearer token is provided, or None if unauthenticated."""
    if credentials is None or not credentials.credentials:
        return None
    if (credentials.scheme or "").lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _resolve_user_from_token(credentials.credentials, db)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
    db: Optional[Session] = Depends(get_db_optional),
) -> User:
    """Require and return the authenticated User from the verified Bearer JWT access token."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if (credentials.scheme or "").lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _resolve_user_from_token(credentials.credentials, db)


async def require_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Ensure the authenticated user account is active; otherwise return 403 Forbidden."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )
    return current_user


async def require_csrf_protection(request: Request) -> None:
    """Enforce Origin check and double-submit CSRF cookie/header validation on cookie-based endpoints."""
    origin = request.headers.get("Origin")
    if origin:
        normalized_origin = origin.strip().rstrip("/")
        if normalized_origin not in settings.allowed_cors_origins:
            logger.warning("Blocked cookie auth request from disallowed origin '%s'.", normalized_origin)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cross-origin request forbidden.",
            )

    cookie_token = request.cookies.get(settings.AUTH_CSRF_COOKIE_NAME)
    header_token = request.headers.get("X-CSRF-Token")
    if not verify_csrf_tokens(cookie_token, header_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="CSRF validation failed.",
        )

