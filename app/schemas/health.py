from typing import List
from pydantic import BaseModel, Field


class HealthCheckResponse(BaseModel):
    """Response schema for general backend health status."""

    status: str = Field(default="ok", description="Status of the backend service")
    app_name: str = Field(..., description="Application name")
    environment: str = Field(..., description="Application execution environment")


class OllamaHealthResponse(BaseModel):
    """Response schema for Ollama LLM service health status."""

    status: str = Field(..., description="Ollama integration health status ('ok' or 'warning')")
    ollama_connected: bool = Field(..., description="Whether the Ollama server is reachable")
    configured_model: str = Field(..., description="The configured Ollama model name")
    model_available: bool = Field(..., description="Whether the configured model is installed")
    available_models: List[str] = Field(
        default_factory=list, description="List of all locally installed Ollama models"
    )
