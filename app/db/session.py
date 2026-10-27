import logging
import os
from typing import Generator, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

logger = logging.getLogger(__name__)


class DatabaseConfigurationError(Exception):
    """Raised when the database configuration or connection string is invalid or missing."""
    pass


def normalize_database_url(url: Optional[str]) -> str:
    """
    Validate and normalize the database URL to use psycopg 3 (postgresql+psycopg://)
    and ensure SSL connection requirement for Neon PostgreSQL.

    Args:
        url: Raw database connection string.

    Returns:
        Normalized connection string with psycopg driver and sslmode=require.

    Raises:
        DatabaseConfigurationError: If url is empty or None.
    """
    if not url or not url.strip():
        raise DatabaseConfigurationError(
            "DATABASE_URL is not configured. Please provide a valid Neon PostgreSQL "
            "connection string (e.g., postgresql+psycopg://USER:PASSWORD@HOST/DATABASE?sslmode=require) "
            "in your environment or .env file."
        )

    url = url.strip()

    # Normalize driver prefix to postgresql+psycopg
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://") and not url.startswith("postgresql+"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]

    # Parse and ensure sslmode=require for Neon PostgreSQL
    if "postgresql+psycopg://" in url:
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        if "sslmode" not in query_params:
            query_params["sslmode"] = ["require"]
            new_query = urlencode(query_params, doseq=True)
            url = urlunparse(parsed._replace(query=new_query))

    return url


def mask_database_url(url: Optional[str]) -> str:
    """
    Safely mask credentials in a database URL for logging and error reporting,
    ensuring passwords and secrets are never exposed.

    Args:
        url: Raw database connection string.

    Returns:
        Sanitized URL string with password masked.
    """
    if not url:
        return "<DATABASE_URL not set>"

    try:
        parsed = urlparse(url)
        if parsed.password:
            user = parsed.username or ""
            host = parsed.hostname or ""
            port_part = f":{parsed.port}" if parsed.port else ""
            netloc = f"{user}:*****@{host}{port_part}"
            return urlunparse(parsed._replace(netloc=netloc))
        return url
    except Exception:
        return "postgresql+psycopg://***:*****@***/neondb"


def is_database_configured() -> bool:
    """Check if DATABASE_URL is configured in settings or environment."""
    url = settings.DATABASE_URL or os.getenv("DATABASE_URL")
    return bool(url and url.strip())


_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker[Session]] = None


def get_engine(database_url: Optional[str] = None) -> Engine:
    """
    Create or retrieve the singleton SQLAlchemy engine.

    Args:
        database_url: Optional explicit URL override (used in testing or migrations).

    Returns:
        SQLAlchemy Engine configured for Neon PostgreSQL.
    """
    global _engine, _session_factory

    if database_url is not None:
        norm_url = normalize_database_url(database_url)
        return create_engine(
            norm_url,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=False,
        )

    if _engine is None:
        raw_url = settings.DATABASE_URL or os.getenv("DATABASE_URL")
        norm_url = normalize_database_url(raw_url)
        logger.info("Initializing database engine with target: %s", mask_database_url(norm_url))
        _engine = create_engine(
            norm_url,
            pool_pre_ping=True,  # Handle Neon serverless connection idle drops
            pool_recycle=300,
            echo=False,
        )
        _session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=_engine,
            expire_on_commit=False,
        )

    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Retrieve the configured sessionmaker."""
    global _session_factory
    if _session_factory is None:
        get_engine()
    assert _session_factory is not None
    return _session_factory


def reset_engine() -> None:
    """Dispose and reset engine and session factory. Primarily for testing."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency yielding a managed SQLAlchemy database session.
    Properly commits transactions on success, rolls back on error, and closes the session.

    Raises:
        DatabaseConfigurationError: If DATABASE_URL is not set.
    """
    if not is_database_configured():
        raise DatabaseConfigurationError(
            "DATABASE_URL is not configured. Please set DATABASE_URL in your environment or .env file."
        )

    session_factory = get_session_factory()
    db = session_factory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_optional() -> Generator[Optional[Session], None, None]:
    """
    FastAPI dependency yielding a managed session if DATABASE_URL is configured,
    or None if running without database configuration (e.g., local memory-only dev mode).
    Also supports FastAPI app.dependency_overrides[get_db].
    """
    from app.main import app  # Deferred import to support test overrides

    if get_db in app.dependency_overrides:
        override = app.dependency_overrides[get_db]
        yield from override()
        return

    if not is_database_configured():
        yield None
        return

    yield from get_db()
