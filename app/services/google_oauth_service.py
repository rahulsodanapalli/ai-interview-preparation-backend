"""Google OAuth 2.0 Authorization Code flow and cryptographic ID token verification."""

from dataclasses import dataclass
import logging
import secrets
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urlparse

import httpx
import jwt
from jwt.exceptions import InvalidTokenError, PyJWTError

from app.core.config import is_configured_oauth_value, settings

logger = logging.getLogger(__name__)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUERS = ["https://accounts.google.com", "accounts.google.com"]


class GoogleOAuthError(Exception):
    """Base error for Google OAuth configuration, exchange, or verification failures."""


class GoogleOAuthConfigurationError(GoogleOAuthError):
    """Raised when Google OAuth environment variables are missing or invalid."""


class GoogleOAuthTokenError(GoogleOAuthError):
    """Raised when Google token exchange or ID token cryptographic validation fails."""


class GoogleOAuthUnverifiedEmailError(GoogleOAuthError):
    """Raised when the Google account's email is not verified."""


@dataclass(frozen=True)
class GoogleUserIdentity:
    """Verified Google identity extracted from a validated ID token."""

    sub: str
    email: str
    name: str
    email_verified: bool


class GoogleOAuthService:
    """Handles Google OAuth 2.0 Authorization Code exchange and JWKS RS256 ID token verification."""

    JWKS_CACHE_TTL_SECONDS = 3600

    def __init__(self) -> None:
        self._jwks_cache: Optional[List[Dict[str, Any]]] = None
        self._jwks_fetched_at: float = 0.0

    def ensure_configured(self) -> tuple[str, str, str]:
        """Validate that Google OAuth settings are present and redirect URI is valid."""
        client_id = (settings.GOOGLE_CLIENT_ID or "").strip()
        client_secret = (settings.GOOGLE_CLIENT_SECRET or "").strip()
        redirect_uri = (settings.GOOGLE_REDIRECT_URI or "").strip()

        has_id = is_configured_oauth_value(client_id)
        has_secret = is_configured_oauth_value(client_secret)
        if not has_id or not has_secret:
            if has_id != has_secret:
                logger.warning("Google OAuth is partially configured; both GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET are required.")
            raise GoogleOAuthConfigurationError(
                "Google OAuth is not configured on the server."
            )

        parsed = urlparse(redirect_uri)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.netloc
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise GoogleOAuthConfigurationError(
                "GOOGLE_REDIRECT_URI is invalid."
            )

        return client_id, client_secret, redirect_uri

    def build_authorization_url(self, *, state: str, nonce: str) -> str:
        """Build the Google OAuth 2.0 authorization redirect URL."""
        client_id, _client_secret, redirect_uri = self.ensure_configured()
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "nonce": nonce,
            "access_type": "offline",
            "prompt": "select_account",
        }
        return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"

    async def exchange_code(self, code: str) -> Dict[str, Any]:
        """Exchange an authorization code for tokens at Google's token endpoint."""
        cleaned_code = (code or "").strip()
        if not cleaned_code:
            raise GoogleOAuthTokenError("Missing Google authorization code.")

        client_id, client_secret, redirect_uri = self.ensure_configured()
        data = {
            "code": cleaned_code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    GOOGLE_TOKEN_URL,
                    data=data,
                    headers={"Accept": "application/json"},
                )
        except httpx.RequestError as exc:
            logger.error("Google OAuth token exchange network failure: %s", type(exc).__name__)
            raise GoogleOAuthTokenError("Failed to communicate with Google OAuth service.") from exc

        if response.status_code != 200:
            oauth_error = ""
            try:
                err_payload = response.json()
                if isinstance(err_payload, dict) and isinstance(err_payload.get("error"), str):
                    oauth_error = err_payload["error"].strip().lower()
            except Exception:
                oauth_error = ""

            logger.warning(
                "Google OAuth token endpoint returned HTTP %s (error=%s)",
                response.status_code,
                oauth_error or "unknown",
            )
            if oauth_error in {"invalid_client", "unauthorized_client", "redirect_uri_mismatch"}:
                raise GoogleOAuthConfigurationError(
                    "Google OAuth server configuration is invalid. Please contact support."
                )
            if oauth_error == "invalid_grant":
                raise GoogleOAuthTokenError(
                    "Google authorization code is invalid or expired. Please sign in again."
                )
            raise GoogleOAuthTokenError("Google authorization code exchange failed.")

        try:
            token_data = response.json()
        except Exception as exc:
            raise GoogleOAuthTokenError("Invalid JSON received from Google OAuth token endpoint.") from exc

        if not isinstance(token_data, dict) or not token_data.get("id_token"):
            raise GoogleOAuthTokenError("Google OAuth response did not include an id_token.")
        return token_data

    async def fetch_jwks(self, *, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Fetch and cache Google's public JWKS keys for RS256 ID token verification."""
        now = time.monotonic()
        if (
            not force_refresh
            and self._jwks_cache is not None
            and (now - self._jwks_fetched_at) < self.JWKS_CACHE_TTL_SECONDS
        ):
            return self._jwks_cache

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(GOOGLE_JWKS_URL)
                resp.raise_for_status()
                payload = resp.json()
        except Exception as exc:
            logger.error("Failed to fetch Google JWKS: %s", type(exc).__name__)
            raise GoogleOAuthTokenError("Unable to retrieve Google signing keys.") from exc

        keys = payload.get("keys") if isinstance(payload, dict) else None
        if not isinstance(keys, list) or not keys:
            raise GoogleOAuthTokenError("Google JWKS response contained no signing keys.")

        self._jwks_cache = keys
        self._jwks_fetched_at = now
        return keys

    async def verify_id_token(
        self,
        id_token: str,
        *,
        expected_nonce: Optional[str] = None,
    ) -> GoogleUserIdentity:
        """Cryptographically verify a Google ID token (signature, iss, aud, exp, email_verified)."""
        client_id, _secret, _redirect = self.ensure_configured()
        if not id_token or not isinstance(id_token, str):
            raise GoogleOAuthTokenError("Missing Google ID token.")

        try:
            unverified_header = jwt.get_unverified_header(id_token)
        except InvalidTokenError as exc:
            raise GoogleOAuthTokenError("Malformed Google ID token header.") from exc

        kid = unverified_header.get("kid")
        alg = unverified_header.get("alg")
        if alg != "RS256":
            raise GoogleOAuthTokenError("Unsupported Google ID token signing algorithm.")
        if not kid or not isinstance(kid, str):
            raise GoogleOAuthTokenError("Missing Google ID token key identifier.")

        keys = await self.fetch_jwks()
        matching_key = next((k for k in keys if k.get("kid") == kid), None)
        if matching_key is None:
            # Key rotation fallback: refresh JWKS once
            keys = await self.fetch_jwks(force_refresh=True)
            matching_key = next((k for k in keys if k.get("kid") == kid), None)

        if matching_key is None:
            raise GoogleOAuthTokenError("Unable to find matching Google signing key.")

        try:
            signing_key = jwt.PyJWK.from_dict(matching_key).key
            claims = jwt.decode(
                id_token,
                key=signing_key,
                algorithms=["RS256"],
                audience=client_id,
                issuer=GOOGLE_ISSUERS,
                options={"require": ["sub", "email", "exp", "iat", "iss", "aud"]},
            )
        except PyJWTError as exc:
            raise GoogleOAuthTokenError(f"Invalid Google ID token: {exc}") from exc

        if isinstance(claims.get("aud"), list) and claims.get("azp") != client_id:
            raise GoogleOAuthTokenError("Google ID token authorized party mismatch.")

        if expected_nonce is not None:
            token_nonce = claims.get("nonce")
            if (
                not expected_nonce
                or not isinstance(token_nonce, str)
                or not token_nonce
                or not secrets.compare_digest(token_nonce, expected_nonce)
            ):
                raise GoogleOAuthTokenError("Google ID token nonce mismatch.")

        email_verified = claims.get("email_verified")
        if email_verified is not True:
            raise GoogleOAuthUnverifiedEmailError(
                "Google account email is not verified."
            )

        sub = str(claims.get("sub") or "").strip()
        email = str(claims.get("email") or "").strip().lower()
        name = str(claims.get("name") or "").strip()
        if not sub or not email:
            raise GoogleOAuthTokenError("Google ID token missing required subject or email claim.")

        if not name:
            name = email.split("@", 1)[0][:120]

        return GoogleUserIdentity(
            sub=sub,
            email=email,
            name=name[:120],
            email_verified=True,
        )

    async def authenticate_code(
        self,
        *,
        code: str,
        expected_nonce: Optional[str] = None,
    ) -> GoogleUserIdentity:
        """Exchange authorization code and return verified Google user identity."""
        token_data = await self.exchange_code(code)
        return await self.verify_id_token(
            token_data["id_token"],
            expected_nonce=expected_nonce,
        )


_google_oauth_service = GoogleOAuthService()


def get_google_oauth_service() -> GoogleOAuthService:
    """FastAPI dependency provider for GoogleOAuthService."""
    return _google_oauth_service

