"""FastAPI application entrypoint, middleware registration, lifespan management, and router mounting."""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import InterfaceError, OperationalError, SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.router import api_v1_router
from app.core.config import settings, validate_auth_configuration
from app.core.logging import configure_logging
from app.db.session import (
    DatabaseConfigurationError,
    DatabaseConnectionError,
    DatabasePersistenceError,
    dispose_engine,
    get_engine,
    is_database_configured,
    is_production_env,
    validate_database_configuration,
)

configure_logging(app_env=settings.APP_ENV)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Validate database & auth configuration on startup and dispose engine on shutdown."""
    validate_database_configuration()
    validate_auth_configuration()
    if is_production_env() or is_database_configured():
        get_engine()
    else:
        logger.info("Starting in non-production mode without DATABASE_URL configured.")
    try:
        yield
    finally:
        dispose_engine()


app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Backend API for the AI Interview Preparation Platform powered by FastAPI "
        "and local Qwen 2.5 on Ollama."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS for explicitly allowed frontend origins with credentialed HttpOnly cookies
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Authorization", "Content-Type", "X-CSRF-Token"],
)


@app.exception_handler(DatabaseConfigurationError)
async def database_configuration_exception_handler(
    request: Request, exc: DatabaseConfigurationError
) -> JSONResponse:
    """Return 503 when database configuration is missing or invalid."""
    logger.error(
        "Database configuration error during %s %s: %s",
        request.method,
        request.url.path,
        type(exc).__name__,
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Database is not properly configured."},
    )


@app.exception_handler(DatabaseConnectionError)
@app.exception_handler(OperationalError)
@app.exception_handler(InterfaceError)
async def database_connection_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Return 503 when PostgreSQL is unreachable or connection fails."""
    logger.error(
        "Database connection error during %s %s: %s",
        request.method,
        request.url.path,
        type(exc).__name__,
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Database service is currently unavailable."},
    )


@app.exception_handler(DatabasePersistenceError)
@app.exception_handler(SQLAlchemyError)
async def database_persistence_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Return 500 when a database persistence or commit operation fails."""
    logger.error(
        "Database persistence error during %s %s: %s",
        request.method,
        request.url.path,
        type(exc).__name__,
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Database persistence error occurred."},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Safely catch unhandled server exceptions to return a clean 500 without leaking secrets or tracebacks."""
    if isinstance(exc, (StarletteHTTPException, RequestValidationError)):
        raise exc
    logger.error(
        "Unhandled internal error processing %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected internal server error occurred."},
    )


# Mount API v1 Routers
api_v1_prefix = settings.API_V1_PREFIX
app.include_router(api_v1_router, prefix=api_v1_prefix)


@app.get(
    "/",
    tags=["Root"],
    summary="API Root",
    description="Returns a welcome message and links to API documentation.",
)
async def root():
    """Root endpoint returning welcome message and status."""
    return {
        "message": "Welcome to the AI Interview Preparation Platform API",
        "status": "online",
        "docs": "/docs",
        "health": f"{api_v1_prefix}/health",
        "ollama_health": f"{api_v1_prefix}/health/ollama",
        "database_health": f"{api_v1_prefix}/health/db",
        "document_upload": f"{api_v1_prefix}/documents/upload",
        "github_validate": f"{api_v1_prefix}/github/validate",
    }
