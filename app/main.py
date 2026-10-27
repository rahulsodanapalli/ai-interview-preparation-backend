import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import documents, health, interview
from app.core.config import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "Backend API for the AI Interview Preparation Platform powered by FastAPI "
        "and local Qwen 2.5 on Ollama."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for local frontend development (React + TypeScript on Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
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
app.include_router(health.router, prefix=api_v1_prefix)
app.include_router(interview.router, prefix=api_v1_prefix)
app.include_router(documents.router, prefix=api_v1_prefix)


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
    }

