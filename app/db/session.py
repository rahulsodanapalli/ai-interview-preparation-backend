import logging
import os
from typing import Any, Dict, Generator, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from fastapi import Request
from sqlalchemy import Engine, create_engine
from sqlalchemy.exc import ArgumentError, NoSuchModuleError, OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database configuration, connection, and persistence errors."""


class DatabaseConfigurationError(DatabaseError):
    """Raised when the database configuration or connection string is invalid or missing."""


class DatabaseConnectionError(DatabaseError):
    """Raised when establishing or using a database connection fails."""


class DatabasePersistenceError(DatabaseError):
    """Raised when persisting or committing database state fails."""


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


def is_production_env() -> bool:
    """Return True when the application is running in production mode."""
    env = settings.APP_ENV or os.getenv("APP_ENV") or "development"
    return env.strip().lower() == "production"


def is_database_configured() -> bool:
    """Check if DATABASE_URL is configured in settings or environment."""
    url = settings.DATABASE_URL or os.getenv("DATABASE_URL")
    return bool(url and url.strip())


def validate_database_configuration() -> None:
    """Validate database configuration at startup, failing fast in production if missing or invalid."""
    if not is_production_env():
        return

    if not is_database_configured():
        raise DatabaseConfigurationError(
            "DATABASE_URL is required when APP_ENV=production. "
            "Please configure a valid PostgreSQL connection string."
        )

    raw_url = settings.DATABASE_URL or os.getenv("DATABASE_URL")
    normalize_database_url(raw_url)


_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker[Session]] = None


def _build_engine_kwargs(norm_url: str) -> Dict[str, Any]:
    """Build SQLAlchemy engine options from application settings."""
    engine_kwargs: Dict[str, Any] = {
        "pool_pre_ping": settings.DB_POOL_PRE_PING,
        "echo": settings.DB_ECHO,
    }
    if not norm_url.startswith("sqlite"):
        engine_kwargs.update(
            {
                "pool_size": settings.DB_POOL_SIZE,
                "max_overflow": settings.DB_MAX_OVERFLOW,
                "pool_timeout": settings.DB_POOL_TIMEOUT,
                "pool_recycle": settings.DB_POOL_RECYCLE,
            }
        )
    return engine_kwargs


def _create_configured_engine(norm_url: str) -> Engine:
    """Create a SQLAlchemy Engine with configured pool settings, masking any credentials on error."""
    engine_kwargs = _build_engine_kwargs(norm_url)
    try:
        return create_engine(norm_url, **engine_kwargs)
    except (ArgumentError, NoSuchModuleError) as exc:
        logger.error(
            "Invalid database engine configuration for target %s: %s",
            mask_database_url(norm_url),
            type(exc).__name__,
        )
        raise DatabaseConfigurationError(
            f"Invalid database configuration for target {mask_database_url(norm_url)}."
        ) from exc
    except OperationalError as exc:
        logger.error(
            "Database connection failed during engine creation for target %s: %s",
            mask_database_url(norm_url),
            type(exc).__name__,
        )
        raise DatabaseConnectionError(
            "Failed to connect to the configured PostgreSQL database."
        ) from exc


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
        return _create_configured_engine(norm_url)

    if _engine is None:
        raw_url = settings.DATABASE_URL or os.getenv("DATABASE_URL")
        norm_url = normalize_database_url(raw_url)
        logger.info(
            "Initializing database engine with target: %s (pool_size=%d, max_overflow=%d, pool_timeout=%d, pool_recycle=%d)",
            mask_database_url(norm_url),
            settings.DB_POOL_SIZE,
            settings.DB_MAX_OVERFLOW,
            settings.DB_POOL_TIMEOUT,
            settings.DB_POOL_RECYCLE,
        )
        _engine = _create_configured_engine(norm_url)
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


def dispose_engine() -> None:
    """Dispose the active SQLAlchemy engine and clear session factory state."""
    global _engine, _session_factory
    if _engine is not None:
        logger.info("Disposing SQLAlchemy database engine.")
        try:
            _engine.dispose()
        finally:
            _engine = None
            _session_factory = None
    else:
        _session_factory = None


def reset_engine() -> None:
    """Dispose and reset engine and session factory. Primarily for testing."""
    dispose_engine()


def _safe_rollback(db: Session) -> None:
    """Attempt to roll back a failed database session without masking the original error."""
    try:
        db.rollback()
    except Exception as exc:
        logger.error("Failed to roll back database session: %s", type(exc).__name__)


def _safe_close(db: Session) -> None:
    """Close a database session safely."""
    try:
        db.close()
    except Exception as exc:
        logger.error("Failed to close database session: %s", type(exc).__name__)


def commit_session(db: Session) -> None:
    """Commit the active database transaction, rolling back and re-raising on failure."""
    try:
        db.commit()
    except Exception:
        _safe_rollback(db)
        raise


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
        _safe_rollback(db)
        raise
    finally:
        _safe_close(db)


def get_db_optional(request: Request = None) -> Generator[Optional[Session], None, None]:  # type: ignore[assignment]
    """
    FastAPI dependency yielding a managed session if DATABASE_URL is configured,
    or None ONLY in non-production environments when DATABASE_URL is intentionally unset.
    In production, missing DATABASE_URL or connection/commit failures raise errors.
    Also supports FastAPI app.dependency_overrides[get_db].
    """
    if request is not None:
        overrides = getattr(request.app, "dependency_overrides", {})
        if get_db in overrides:
            yield from overrides[get_db]()
            return

    if not is_database_configured():
        if is_production_env():
            raise DatabaseConfigurationError(
                "DATABASE_URL is required when APP_ENV=production."
            )
        yield None
        return

    try:
        session_factory = get_session_factory()
        db = session_factory()
    except DatabaseError:
        raise
    except SQLAlchemyError as exc:
        logger.error("Could not initialize database session: %s", type(exc).__name__)
        raise DatabaseConnectionError(
            "Could not establish a database session."
        ) from exc

    try:
        yield db
        db.commit()
    except Exception:
        _safe_rollback(db)
        raise
    finally:
        _safe_close(db)
