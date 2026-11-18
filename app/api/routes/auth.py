"""Authentication and Google OAuth 2.0 API routes."""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.api.dependencies.auth import require_active_user, require_csrf_protection
from app.core.config import settings
from app.core.rate_limit import rate_limit_auth
from app.core.security import generate_csrf_token
from app.db.session import get_db
from app.models.auth import User
from app.schemas.auth import (
    AuthResponse,
    CsrfTokenResponse,
    GoogleAuthorizeUrlResponse,
    GoogleCallbackRequest,
    LoginRequest,
    MessageResponse,
    RegisterRequest,
    UserResponse,
)
from app.services.auth_service import (
    AuthService,
    EmailAlreadyExistsError,
    InactiveUserError,
    InvalidCredentialsError,
    InvalidRefreshTokenError,
    UnsafeAccountLinkingError,
    get_auth_service,
    user_to_response,
)
from app.services.google_oauth_service import (
    GoogleOAuthConfigurationError,
    GoogleOAuthTokenError,
    GoogleOAuthUnverifiedEmailError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


def _refresh_cookie_path() -> str:
    """Return the cookie path scoped to auth endpoints."""
    return f"{settings.API_V1_PREFIX.rstrip('/')}/auth"


def _set_auth_cookies(
    response: Response,
    *,
    refresh_token: str,
    csrf_token: str,
    max_age: int,
) -> None:
    """Attach HttpOnly refresh token cookie and double-submit CSRF cookie."""
    secure = settings.effective_cookie_secure
    samesite = (settings.AUTH_COOKIE_SAMESITE or "lax").strip().lower()

    response.set_cookie(
        key=settings.AUTH_COOKIE_NAME,
        value=refresh_token,
        httponly=True,
        secure=secure,
        samesite=samesite,  # type: ignore[arg-type]
        path=_refresh_cookie_path(),
        max_age=max_age,
    )
    response.set_cookie(
        key=settings.AUTH_CSRF_COOKIE_NAME,
        value=csrf_token,
        httponly=False,
        secure=secure,
        samesite=samesite,  # type: ignore[arg-type]
        path="/",
        max_age=max_age,
    )


def _clear_auth_cookies(response: Response) -> None:
    """Delete refresh token and CSRF cookies on logout."""
    secure = settings.effective_cookie_secure
    samesite = (settings.AUTH_COOKIE_SAMESITE or "lax").strip().lower()
    response.delete_cookie(
        key=settings.AUTH_COOKIE_NAME,
        path=_refresh_cookie_path(),
        secure=secure,
        samesite=samesite,  # type: ignore[arg-type]
    )
    response.delete_cookie(
        key=settings.AUTH_CSRF_COOKIE_NAME,
        path="/",
        secure=secure,
        samesite=samesite,  # type: ignore[arg-type]
    )


@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register New User Account",
    dependencies=[Depends(rate_limit_auth("register"))],
    responses={
        201: {"description": "User registered and authenticated"},
        409: {"description": "Email address already registered"},
        422: {"description": "Invalid input or weak password"},
        429: {"description": "Too many registration attempts"},
    },
)
async def register(
    payload: RegisterRequest,
    response: Response,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthResponse:
    """Create a new user account with Argon2id password hashing and issue auth tokens."""
    try:
        result = auth_service.register(db, payload)
    except EmailAlreadyExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    _set_auth_cookies(
        response,
        refresh_token=result.refresh_token,
        csrf_token=result.response.csrf_token,
        max_age=result.refresh_max_age_seconds,
    )
    return result.response


@router.post(
    "/login",
    response_model=AuthResponse,
    status_code=status.HTTP_200_OK,
    summary="Authenticate with Email and Password",
    dependencies=[Depends(rate_limit_auth("login"))],
    responses={
        200: {"description": "Authentication succeeded"},
        401: {"description": "Invalid credentials"},
        403: {"description": "Inactive user account"},
        429: {"description": "Too many login attempts"},
    },
)
async def login(
    payload: LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthResponse:
    """Authenticate an existing user with email and password."""
    try:
        result = auth_service.login(db, payload)
    except InvalidCredentialsError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except InactiveUserError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc

    _set_auth_cookies(
        response,
        refresh_token=result.refresh_token,
        csrf_token=result.response.csrf_token,
        max_age=result.refresh_max_age_seconds,
    )
    return result.response


@router.get(
    "/csrf",
    response_model=CsrfTokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Get or Initialize CSRF Token",
)
async def get_csrf_token(request: Request, response: Response) -> CsrfTokenResponse:
    """Return the active CSRF token or generate a new double-submit CSRF cookie."""
    existing = request.cookies.get(settings.AUTH_CSRF_COOKIE_NAME)
    token = existing if existing and existing.strip() else generate_csrf_token()
    samesite = (settings.AUTH_COOKIE_SAMESITE or "lax").strip().lower()
    response.set_cookie(
        key=settings.AUTH_CSRF_COOKIE_NAME,
        value=token,
        httponly=False,
        secure=settings.effective_cookie_secure,
        samesite=samesite,  # type: ignore[arg-type]
        path="/",
        max_age=settings.JWT_REFRESH_REMEMBER_ME_DAYS * 24 * 60 * 60,
    )
    return CsrfTokenResponse(csrf_token=token)


@router.post(
    "/refresh",
    response_model=AuthResponse,
    status_code=status.HTTP_200_OK,
    summary="Rotate Refresh Token and Issue New Access Token",
    dependencies=[Depends(rate_limit_auth("refresh")), Depends(require_csrf_protection)],
    responses={
        200: {"description": "Tokens rotated and refreshed"},
        401: {"description": "Invalid, expired, or revoked refresh token"},
        403: {"description": "CSRF validation failed or account inactive"},
    },
)
async def refresh_tokens(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthResponse | JSONResponse:
    """Validate HttpOnly refresh cookie + CSRF header, rotate refresh token, and return new JWT."""
    raw_refresh = request.cookies.get(settings.AUTH_COOKIE_NAME)
    try:
        result = auth_service.refresh(db, raw_refresh)
    except InvalidRefreshTokenError as exc:
        err_resp = JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": str(exc)},
            headers={"WWW-Authenticate": "Bearer"},
        )
        _clear_auth_cookies(err_resp)
        return err_resp
    except InactiveUserError as exc:
        err_resp = JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": str(exc)},
        )
        _clear_auth_cookies(err_resp)
        return err_resp

    _set_auth_cookies(
        response,
        refresh_token=result.refresh_token,
        csrf_token=result.response.csrf_token,
        max_age=result.refresh_max_age_seconds,
    )
    return result.response


@router.post(
    "/logout",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Logout and Revoke Refresh Token",
    dependencies=[Depends(require_csrf_protection)],
)
async def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service),
) -> MessageResponse:
    """Revoke the active refresh token family in PostgreSQL and clear auth cookies."""
    raw_refresh = request.cookies.get(settings.AUTH_COOKIE_NAME)
    auth_service.logout(db, raw_refresh)
    _clear_auth_cookies(response)
    return MessageResponse(message="Successfully logged out.")


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Current Authenticated User",
)
async def get_me(
    current_user: User = Depends(require_active_user),
) -> UserResponse:
    """Return the currently authenticated active user's profile."""
    return user_to_response(current_user)


def _oauth_cookie_options() -> tuple[bool, str, str]:
    """Return (secure, samesite, path) for the temporary OAuth state cookie."""
    secure = settings.effective_cookie_secure
    configured_samesite = (settings.AUTH_COOKIE_SAMESITE or "lax").strip().lower()
    samesite = "none" if configured_samesite == "none" else "lax"
    path = f"{settings.API_V1_PREFIX.rstrip('/')}/auth/google"
    return secure, samesite, path


@router.get(
    "/google/login",
    summary="Initiate Google OAuth 2.0 Authorization Code Flow",
    dependencies=[Depends(rate_limit_auth("google_login"))],
    responses={
        302: {"description": "Redirect to Google OAuth consent screen"},
        200: {"description": "JSON authorization URL when mode=json"},
        503: {"description": "Google OAuth is not configured"},
    },
)
async def google_login(
    mode: Optional[str] = Query(None, description="Set to 'json' to return URL instead of 302 redirect"),
    auth_service: AuthService = Depends(get_auth_service),
):
    """Start Google OAuth 2.0 flow with signed state JWT and HttpOnly state cookie."""
    try:
        auth_url, state, nonce = auth_service.create_google_authorization()
    except GoogleOAuthConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    secure, samesite, oauth_path = _oauth_cookie_options()
    if mode == "json":
        resp = JSONResponse(
            content=GoogleAuthorizeUrlResponse(
                authorization_url=auth_url,
                state=state,
            ).model_dump()
        )
    else:
        resp = RedirectResponse(url=auth_url, status_code=status.HTTP_302_FOUND)

    resp.set_cookie(
        key=settings.AUTH_OAUTH_STATE_COOKIE_NAME,
        value=nonce,
        httponly=True,
        secure=secure,
        samesite=samesite,  # type: ignore[arg-type]
        path=oauth_path,
        max_age=600,
    )
    return resp


@router.post(
    "/google/callback",
    response_model=AuthResponse,
    status_code=status.HTTP_200_OK,
    summary="Complete Google OAuth 2.0 Authorization Code Exchange",
    dependencies=[Depends(rate_limit_auth("google_callback"))],
    responses={
        200: {"description": "Google authentication succeeded"},
        400: {"description": "Invalid OAuth state or Google token"},
        403: {"description": "Unverified Google email or inactive account"},
        409: {"description": "Unsafe account linking blocked"},
        503: {"description": "Google OAuth not configured"},
    },
)
async def google_callback(
    payload: GoogleCallbackRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service),
) -> AuthResponse:
    """Verify OAuth state, exchange code at Google, verify ID token, and issue platform tokens."""
    cookie_nonce = request.cookies.get(settings.AUTH_OAUTH_STATE_COOKIE_NAME)
    if not cookie_nonce or not cookie_nonce.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing OAuth state cookie. Please restart Google sign-in.",
        )

    try:
        result = await auth_service.complete_google_oauth(
            db,
            code=payload.code,
            state=payload.state,
            cookie_nonce=cookie_nonce,
        )
    except GoogleOAuthConfigurationError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except GoogleOAuthUnverifiedEmailError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except (InvalidCredentialsError, GoogleOAuthTokenError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except InactiveUserError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except UnsafeAccountLinkingError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    secure, samesite, oauth_path = _oauth_cookie_options()
    response.delete_cookie(
        key=settings.AUTH_OAUTH_STATE_COOKIE_NAME,
        path=oauth_path,
        secure=secure,
        samesite=samesite,  # type: ignore[arg-type]
    )
    _set_auth_cookies(
        response,
        refresh_token=result.refresh_token,
        csrf_token=result.response.csrf_token,
        max_age=result.refresh_max_age_seconds,
    )
    return result.response

