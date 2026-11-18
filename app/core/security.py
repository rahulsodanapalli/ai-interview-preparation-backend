"""Cryptographic and token security primitives for authentication, passwords, JWTs, and CSRF."""

from datetime import datetime, timedelta, timezone
import hashlib
import logging
import re
import secrets
from typing import Any, Dict, Optional
import uuid

from argon2 import PasswordHasher, Type as Argon2Type
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
import jwt
from jwt.exceptions import InvalidTokenError

from app.core.config import settings

logger = logging.getLogger(__name__)

# Argon2id password hasher configured with OWASP-recommended parameters
_password_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=65536,
    parallelism=4,
    hash_len=32,
    salt_len=16,
    type=Argon2Type.ID,
)

# Ephemeral dev-only secret generated per process when JWT_SECRET_KEY is unset in non-production
_DEV_EPHEMERAL_JWT_SECRET: str = secrets.token_urlsafe(48)
_DEV_SECRET_WARNING_LOGGED: bool = False

# Pre-computed Argon2id hash for constant-time dummy verification on unknown emails
DUMMY_PASSWORD_HASH: str = _password_hasher.hash("DummyPassword#2026!TimingGuard")

_EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+$")


class TokenValidationError(Exception):
    """Raised when an access token or OAuth state token is invalid or expired."""


def get_jwt_signing_key() -> str:
    """Return the configured JWT secret key, or a secure ephemeral key in non-production."""
    global _DEV_SECRET_WARNING_LOGGED
    configured = (settings.JWT_SECRET_KEY or "").strip()
    if configured:
        return configured

    if settings.is_production:
        raise RuntimeError("JWT_SECRET_KEY is not configured in production.")

    if not _DEV_SECRET_WARNING_LOGGED:
        logger.warning(
            "JWT_SECRET_KEY is not set; using an ephemeral process-local key for development."
        )
        _DEV_SECRET_WARNING_LOGGED = True
    return _DEV_EPHEMERAL_JWT_SECRET


def normalize_email(email: str) -> str:
    """Normalize an email address by stripping whitespace and lowercasing."""
    return (email or "").strip().lower()


def validate_email_format(email: str) -> str:
    """Validate and normalize an email address."""
    normalized = normalize_email(email)
    if not normalized or len(normalized) > 255 or not _EMAIL_REGEX.match(normalized):
        raise ValueError("Please provide a valid email address.")
    return normalized


def validate_password_strength(password: str) -> str:
    """Enforce password complexity requirements without mutating the password."""
    if not isinstance(password, str) or len(password) < 8:
        raise ValueError("Password must be at least 8 characters long.")
    if len(password) > 128:
        raise ValueError("Password must not exceed 128 characters.")
    if not any(c.isupper() for c in password):
        raise ValueError("Password must contain at least one uppercase letter.")
    if not any(c.islower() for c in password):
        raise ValueError("Password must contain at least one lowercase letter.")
    if not any(c.isdigit() for c in password):
        raise ValueError("Password must contain at least one digit.")
    if not any(not c.isalnum() and not c.isspace() for c in password):
        raise ValueError("Password must contain at least one special character.")
    return password


def hash_password(password: str) -> str:
    """Hash a plaintext password using Argon2id."""
    return _password_hasher.hash(password)


def verify_password(plain_password: str, password_hash: Optional[str]) -> bool:
    """Verify a plaintext password against an Argon2id hash in constant-safe manner."""
    if not plain_password or not password_hash:
        # Still run dummy check to keep timing uniform
        try:
            _password_hasher.verify(DUMMY_PASSWORD_HASH, plain_password or "invalid")
        except Exception:
            pass
        return False

    try:
        return bool(_password_hasher.verify(password_hash, plain_password))
    except (VerifyMismatchError, InvalidHashError, VerificationError):
        return False


def password_needs_rehash(password_hash: str) -> bool:
    """Check whether an existing password hash should be upgraded to current Argon2id parameters."""
    try:
        return _password_hasher.check_needs_rehash(password_hash)
    except Exception:
        return False


def create_access_token(
    *,
    user_id: str,
    email: str,
    expires_delta: Optional[timedelta] = None,
) -> tuple[str, int]:
    """Create a short-lived signed JWT access token and return (token, expires_in_seconds)."""
    now = datetime.now(timezone.utc)
    delta = expires_delta or timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    expires_at = now + delta
    payload: Dict[str, Any] = {
        "sub": user_id,
        "email": email,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
        "jti": str(uuid.uuid4()),
    }
    encoded = jwt.encode(payload, get_jwt_signing_key(), algorithm=settings.JWT_ALGORITHM)
    return encoded, max(1, int(delta.total_seconds()))


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT access token, enforcing signature, expiration, and token type."""
    if not token or not token.strip():
        raise TokenValidationError("Access token is missing.")
    try:
        payload = jwt.decode(
            token,
            get_jwt_signing_key(),
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": ["sub", "exp", "iat", "type"]},
        )
    except InvalidTokenError as exc:
        raise TokenValidationError("Invalid or expired access token.") from exc

    if payload.get("type") != "access" or not payload.get("sub"):
        raise TokenValidationError("Invalid access token claims.")
    return payload


def create_oauth_state_token(
    *,
    nonce: str,
    redirect_uri: str,
    expires_minutes: int = 10,
) -> str:
    """Create a short-lived signed state JWT binding OAuth authorization requests to a nonce."""
    now = datetime.now(timezone.utc)
    payload = {
        "type": "oauth_state",
        "nonce": nonce,
        "redirect_uri": redirect_uri,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_minutes)).timestamp()),
        "jti": str(uuid.uuid4()),
    }
    return jwt.encode(payload, get_jwt_signing_key(), algorithm=settings.JWT_ALGORITHM)


def verify_oauth_state_token(state: str, expected_redirect_uri: str) -> Dict[str, Any]:
    """Validate an OAuth state token's signature, expiry, type, and redirect URI."""
    if not state or not state.strip():
        raise TokenValidationError("OAuth state parameter is missing.")
    try:
        payload = jwt.decode(
            state,
            get_jwt_signing_key(),
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": ["exp", "iat", "type", "nonce", "redirect_uri"]},
        )
    except InvalidTokenError as exc:
        raise TokenValidationError("Invalid or expired OAuth state.") from exc

    if payload.get("type") != "oauth_state":
        raise TokenValidationError("Invalid OAuth state token type.")
    if payload.get("redirect_uri") != expected_redirect_uri:
        raise TokenValidationError("OAuth state redirect URI mismatch.")
    return payload


def generate_refresh_token() -> str:
    """Generate a high-entropy opaque refresh token."""
    return secrets.token_urlsafe(48)


def hash_token(raw_token: str) -> str:
    """Compute a SHA-256 hex digest of an opaque token for safe database persistence."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def generate_csrf_token() -> str:
    """Generate a high-entropy CSRF token for double-submit cookie verification."""
    return secrets.token_urlsafe(32)


def verify_csrf_tokens(cookie_token: Optional[str], header_token: Optional[str]) -> bool:
    """Verify CSRF cookie and header tokens match in constant time."""
    if not cookie_token or not header_token:
        return False
    return secrets.compare_digest(cookie_token.strip(), header_token.strip())

