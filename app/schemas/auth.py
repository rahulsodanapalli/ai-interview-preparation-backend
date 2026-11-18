"""Pydantic request and response schemas for user authentication and Google OAuth."""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from app.core.security import validate_email_format, validate_password_strength


class RegisterRequest(BaseModel):
    """Payload for registering a new user account with email and password."""

    name: str = Field(..., min_length=2, max_length=120, description="User display name")
    email: str = Field(..., max_length=255, description="Unique email address")
    password: str = Field(..., min_length=8, max_length=128, description="Strong password")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        cleaned = " ".join((v or "").split())
        if len(cleaned) < 2:
            raise ValueError("Name must be at least 2 characters long.")
        return cleaned

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return validate_email_format(v)

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        return validate_password_strength(v)


class LoginRequest(BaseModel):
    """Payload for authenticating with email and password."""

    email: str = Field(..., max_length=255, description="Registered email address")
    password: str = Field(..., min_length=1, max_length=128, description="Account password")
    remember_me: bool = Field(
        default=False,
        description="Extend refresh session lifetime securely via HttpOnly cookie",
    )

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        return validate_email_format(v)


class GoogleCallbackRequest(BaseModel):
    """Payload for completing Google OAuth 2.0 Authorization Code exchange."""

    code: str = Field(..., min_length=1, max_length=2048, description="Authorization code from Google")
    state: str = Field(..., min_length=1, max_length=2048, description="Signed OAuth state token")

    @field_validator("code", "state")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        cleaned = (v or "").strip()
        if not cleaned:
            raise ValueError("Field must not be empty.")
        return cleaned


class UserResponse(BaseModel):
    """Authenticated user profile representation (never exposes password hashes or secrets)."""

    id: str
    name: str
    email: str
    is_active: bool
    email_verified: bool
    auth_providers: List[str] = Field(default_factory=list)
    created_at: str


class AuthResponse(BaseModel):
    """Authentication token response returned upon login, registration, refresh, or OAuth."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    csrf_token: str
    user: UserResponse


class CsrfTokenResponse(BaseModel):
    """Response carrying the double-submit CSRF token for browser clients."""

    csrf_token: str


class GoogleAuthorizeUrlResponse(BaseModel):
    """Optional JSON response containing Google OAuth authorization redirect URL."""

    authorization_url: str
    state: str


class MessageResponse(BaseModel):
    """Generic status message response."""

    message: str

