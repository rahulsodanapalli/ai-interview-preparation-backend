from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import settings
from app.schemas.health import HealthCheckResponse, OllamaHealthResponse
from app.services.ollama_service import (
    OllamaConnectionError,
    OllamaResponseError,
    OllamaService,
    OllamaTimeoutError,
    get_ollama_service,
)

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    response_model=HealthCheckResponse,
    summary="Backend Health Check",
    description="Returns the operational status and environment of the FastAPI backend.",
)
async def get_backend_health() -> HealthCheckResponse:
    """Return backend health and environment information."""
    return HealthCheckResponse(
        status="ok",
        app_name=settings.APP_NAME,
        environment=settings.APP_ENV,
    )


@router.get(
    "/ollama",
    response_model=OllamaHealthResponse,
    summary="Ollama LLM Service Health Check",
    description=(
        "Checks whether the Ollama server is reachable and whether the configured model is installed."
    ),
    responses={
        503: {"description": "Ollama service is unreachable or timed out"},
        502: {"description": "Ollama returned an unexpected or error response"},
    },
)
async def get_ollama_health(
    ollama_service: OllamaService = Depends(get_ollama_service),
) -> OllamaHealthResponse:
    """Verify Ollama server reachability and configured model installation."""
    try:
        health_info = await ollama_service.check_health()
    except (OllamaConnectionError, OllamaTimeoutError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except OllamaResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    status_str = "ok" if health_info["model_available"] else "warning"

    return OllamaHealthResponse(
        status=status_str,
        ollama_connected=health_info["ollama_connected"],
        configured_model=health_info["configured_model"],
        model_available=health_info["model_available"],
        available_models=health_info["available_models"],
    )
