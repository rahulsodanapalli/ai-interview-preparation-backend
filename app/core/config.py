from functools import lru_cache
import json
from pathlib import Path
from typing import Any, List, Optional, Union
from urllib.parse import urlparse
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AuthConfigurationError(ValueError):
    """Raised when required authentication or security configuration is missing or invalid."""


_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE_PATH = _BACKEND_ROOT / ".env"

_INSECURE_SECRET_PLACEHOLDERS = {
    "changeme",
    "change-me",
    "secret",
    "your-secret-key",
    "your_jwt_secret_key_here",
    "replace-with-secure-random-secret",
}

_OAUTH_CONFIG_PLACEHOLDERS = {
    "changeme",
    "change-me",
    "placeholder",
    "your_google_client_id_here",
    "your-google-client-id",
    "your_google_client_secret_here",
    "your-google-client-secret",
    "replace-with-google-client-id",
    "replace-with-google-client-secret",
}


def is_configured_oauth_value(value: Optional[str]) -> bool:
    """Return True when an OAuth credential string is non-empty and not an example placeholder."""
    cleaned = (value or "").strip()
    if not cleaned:
        return False
    return cleaned.lower() not in _OAUTH_CONFIG_PLACEHOLDERS


class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables."""

    APP_NAME: str = "AI Interview Preparation Platform"
    APP_ENV: str = "development"
    API_V1_PREFIX: str = "/api/v1"

    # CORS Configuration for local frontend integration
    CORS_ORIGINS: Union[str, List[str]] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Ollama Service Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:1.5b"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_CONNECT_TIMEOUT_SECONDS: float = 10.0
    OLLAMA_TIMEOUT_SECONDS: float = 120.0
    OLLAMA_CHAT_TIMEOUT_SECONDS: float = 180.0
    OLLAMA_NUM_CTX: int = 8192
    OLLAMA_NUM_PREDICT: int = 3072
    OLLAMA_MAX_CONCURRENT_CHATS: int = 1
    OLLAMA_MAX_RETRIES: int = 1

    # Database Configuration (Neon PostgreSQL)
    DATABASE_URL: Optional[str] = None
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 300
    DB_POOL_PRE_PING: bool = True
    DB_ECHO: bool = False

    # Document Upload & Ingestion Settings
    MAX_UPLOAD_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB limit
    MAX_PDF_PAGES: int = 100  # Reasonable limit to protect against DoS

    # Text Chunking Settings (RAG Phase 3)
    # Approximation: 1 token ≈ 4 characters in standard English text.
    # Target: ~500 tokens ≈ 2,000 characters; ~10% overlap ≈ 200 characters (~50 tokens).
    CHUNK_SIZE: int = 2000
    CHUNK_OVERLAP: int = 200

    # Pinecone Vector Database Settings (RAG Phase 4)
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: str = "ai-interview-prep-v2"
    PINECONE_NAMESPACE: str = "default"
    PINECONE_TOP_K: int = 4
    PINECONE_DIMENSION: int = 768
    PINECONE_METRIC: str = "cosine"

    # Semantic Retrieval & Context Grounding Settings (RAG Phase 5)
    # Minimum cosine similarity threshold for retrieved chunks using nomic-embed-text.
    # Excerpts scoring below this threshold are filtered out as irrelevant/noise.
    RAG_MIN_RELEVANCE_SCORE: float = 0.55

    # Advanced RAG Settings (Hybrid Search, Reranking, Evaluation)
    RAG_ENABLE_HYBRID_SEARCH: bool = True
    RAG_HYBRID_ALPHA: float = 0.5  # Weight for dense vector search (1 - alpha for BM25)
    RAG_CANDIDATE_POOL_SIZE: int = 15  # Candidate pool size retrieved from Pinecone before reranking
    RAG_ENABLE_RERANKING: bool = True
    RAG_RERANKER_TYPE: str = "auto"  # 'auto', 'flashrank', 'ollama', or 'rrf'

    # GitHub MCP Integration Settings (Day 6)
    # Optional personal access token for read-only repository inspection.
    # If not provided, public repositories will still be accessible within standard GitHub rate limits.
    GITHUB_TOKEN: Optional[str] = None
    GITHUB_PERSONAL_ACCESS_TOKEN: Optional[str] = None
    GITHUB_API_BASE_URL: str = "https://api.github.com"
    GITHUB_REQUEST_TIMEOUT_SECONDS: float = 15.0
    GITHUB_MCP_TIMEOUT_SECONDS: float = 30.0
    GITHUB_MCP_SERVER_COMMAND: Optional[str] = None
    GITHUB_MAX_FILE_SIZE_BYTES: int = 500 * 1024  # 500 KB limit for reading code files

    # Project Explainer Ingestion & Analysis Limits
    PROJECT_EXPLAINER_MAX_FILES: int = 35
    PROJECT_EXPLAINER_MAX_TREE_DEPTH: int = 4
    PROJECT_EXPLAINER_MAX_DIR_LISTINGS: int = 60
    PROJECT_EXPLAINER_MAX_FILE_BYTES: int = 150 * 1024  # 150 KB per file
    PROJECT_EXPLAINER_MAX_TOTAL_CHARS: int = 180_000
    PROJECT_EXPLAINER_MAX_FILE_CHARS: int = 12_000
    PROJECT_EXPLAINER_STAGE_CONTEXT_CHARS: int = 5_500
    PROJECT_EXPLAINER_README_PROMPT_CHARS: int = 2_500
    PROJECT_EXPLAINER_BATCH_MAX_CHARS: int = 4_500
    PROJECT_EXPLAINER_GROUP_MAX_FILES: int = 5
    PROJECT_EXPLAINER_GROUP_MAX_CHARS: int = 3_500
    PROJECT_EXPLAINER_GROUP_NUM_PREDICT: int = 512
    PROJECT_EXPLAINER_SYNTHESIS_NUM_PREDICT: int = 2048

    # Authentication & JWT Security Settings (Phase 3)
    FRONTEND_URL: str = "http://localhost:5173"
    JWT_SECRET_KEY: Optional[str] = None
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    JWT_REFRESH_REMEMBER_ME_DAYS: int = 30

    # Google OAuth 2.0 Settings (Authorization Code Flow)
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None
    GOOGLE_REDIRECT_URI: str = "http://localhost:5173/auth/callback"

    # Cookie & CSRF Settings
    AUTH_COOKIE_NAME: str = "refresh_token"
    AUTH_CSRF_COOKIE_NAME: str = "csrf_token"
    AUTH_OAUTH_STATE_COOKIE_NAME: str = "oauth_state"
    AUTH_COOKIE_SECURE: Optional[bool] = None
    AUTH_COOKIE_SAMESITE: str = "lax"

    # Authentication Rate Limiting
    AUTH_RATE_LIMIT_MAX_ATTEMPTS: int = 15
    AUTH_RATE_LIMIT_WINDOW_SECONDS: int = 60


    @field_validator("CORS_ORIGINS")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            trimmed = v.strip()
            if trimmed.startswith("[") and trimmed.endswith("]"):
                try:
                    return [str(item).strip() for item in json.loads(trimmed) if str(item).strip()]
                except Exception:
                    pass
            return [item.strip() for item in trimmed.split(",") if item.strip()]
        elif isinstance(v, (list, tuple, set)):
            return [str(item).strip() for item in v if str(item).strip()]
        return ["http://localhost:5173", "http://127.0.0.1:5173"]

    @property
    def is_production(self) -> bool:
        """Return True when the application is configured to run in production."""
        return (self.APP_ENV or "").strip().lower() == "production"

    @property
    def effective_cookie_secure(self) -> bool:
        """Return True when cookies must set the Secure flag."""
        if self.AUTH_COOKIE_SECURE is not None:
            return bool(self.AUTH_COOKIE_SECURE)
        return self.is_production

    @property
    def allowed_cors_origins(self) -> List[str]:
        """Explicitly allowed frontend origins (never wildcard '*' when credentials are enabled)."""
        origins = [
            origin.rstrip("/")
            for origin in (self.CORS_ORIGINS if isinstance(self.CORS_ORIGINS, list) else [])
            if origin and origin.strip() and origin.strip() != "*"
        ]
        frontend_origin = self._extract_origin(self.FRONTEND_URL)
        if frontend_origin and frontend_origin not in origins:
            origins.append(frontend_origin)
        return origins

    @staticmethod
    def _extract_origin(url: Optional[str]) -> Optional[str]:
        if not url or not url.strip():
            return None
        parsed = urlparse(url.strip())
        if parsed.scheme in ("http", "https") and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
        return None

    @model_validator(mode="after")
    def validate_database_settings(self) -> "Settings":
        """Enforce required database configuration in production and valid pool bounds."""
        if self.DB_POOL_SIZE < 1:
            raise ValueError("DB_POOL_SIZE must be at least 1.")
        if self.DB_MAX_OVERFLOW < 0:
            raise ValueError("DB_MAX_OVERFLOW must be 0 or greater.")
        if self.DB_POOL_TIMEOUT < 1:
            raise ValueError("DB_POOL_TIMEOUT must be at least 1.")
        if self.DB_POOL_RECYCLE < 0:
            raise ValueError("DB_POOL_RECYCLE must be 0 or greater.")

        if self.JWT_ACCESS_TOKEN_EXPIRE_MINUTES < 1:
            raise ValueError("JWT_ACCESS_TOKEN_EXPIRE_MINUTES must be at least 1.")
        if self.JWT_REFRESH_TOKEN_EXPIRE_DAYS < 1:
            raise ValueError("JWT_REFRESH_TOKEN_EXPIRE_DAYS must be at least 1.")
        if self.JWT_REFRESH_REMEMBER_ME_DAYS < 1:
            raise ValueError("JWT_REFRESH_REMEMBER_ME_DAYS must be at least 1.")
        if (self.JWT_ALGORITHM or "").strip().upper() not in {"HS256", "HS384", "HS512"}:
            raise ValueError("JWT_ALGORITHM must be 'HS256', 'HS384', or 'HS512'.")
        self.JWT_ALGORITHM = self.JWT_ALGORITHM.strip().upper()
        if (self.AUTH_COOKIE_SAMESITE or "").strip().lower() not in {"lax", "strict", "none"}:
            raise ValueError("AUTH_COOKIE_SAMESITE must be 'lax', 'strict', or 'none'.")

        if self.is_production and not (self.DATABASE_URL and self.DATABASE_URL.strip()):
            raise ValueError(
                "DATABASE_URL is required when APP_ENV=production."
            )
        return self

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
    )


def validate_auth_configuration(cfg: Optional[Settings] = None) -> None:
    """Validate required authentication and OAuth settings at startup in production."""
    target = cfg or settings
    if not target.is_production:
        return

    secret = (target.JWT_SECRET_KEY or "").strip()
    if not secret:
        raise AuthConfigurationError(
            "JWT_SECRET_KEY is required when APP_ENV=production."
        )
    if len(secret) < 32 or secret.lower() in _INSECURE_SECRET_PLACEHOLDERS:
        raise AuthConfigurationError(
            "JWT_SECRET_KEY must be a strong secret of at least 32 characters in production."
        )

    frontend_origin = Settings._extract_origin(target.FRONTEND_URL)
    if not frontend_origin:
        raise AuthConfigurationError(
            "FRONTEND_URL must be a valid http(s) URL when APP_ENV=production."
        )

    if isinstance(target.CORS_ORIGINS, list) and any(o.strip() == "*" for o in target.CORS_ORIGINS):
        raise AuthConfigurationError(
            "CORS_ORIGINS cannot contain wildcard '*' when credentialed auth cookies are enabled."
        )

    has_google_id = is_configured_oauth_value(target.GOOGLE_CLIENT_ID)
    has_google_secret = is_configured_oauth_value(target.GOOGLE_CLIENT_SECRET)
    if has_google_id != has_google_secret:
        raise AuthConfigurationError(
            "Both GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET must be set together."
        )
    if has_google_id:
        parsed_redirect = urlparse((target.GOOGLE_REDIRECT_URI or "").strip())
        if (
            parsed_redirect.scheme not in ("http", "https")
            or not parsed_redirect.netloc
            or parsed_redirect.fragment
            or parsed_redirect.username
            or parsed_redirect.password
        ):
            raise AuthConfigurationError(
                "GOOGLE_REDIRECT_URI must be a valid http(s) URL without fragments or credentials when Google OAuth is enabled."
            )


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of the application settings."""
    return Settings()


settings = get_settings()
