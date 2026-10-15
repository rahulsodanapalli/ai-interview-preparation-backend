from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.services.ollama_service import (
    OllamaConnectionError,
    OllamaService,
    get_ollama_service,
)

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint returns 200 OK and welcome payload."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["status"] == "online"
    assert "docs" in data


def test_backend_health_endpoint():
    """Test the general backend health endpoint returns 200 OK."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "app_name" in data
    assert "environment" in data


def test_ollama_health_endpoint_success():
    """Test Ollama health endpoint returns 200 when Ollama is reachable and model is installed."""
    class MockOllamaServiceSuccess(OllamaService):
        async def check_health(self):
            return {
                "ollama_connected": True,
                "configured_model": "qwen2.5:1.5b",
                "model_available": True,
                "available_models": ["qwen2.5:1.5b"],
            }

    app.dependency_overrides[get_ollama_service] = lambda: MockOllamaServiceSuccess()
    try:
        response = client.get("/api/v1/health/ollama")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["ollama_connected"] is True
        assert data["model_available"] is True
        assert "qwen2.5:1.5b" in data["available_models"]
    finally:
        app.dependency_overrides.clear()


def test_ollama_health_endpoint_model_not_installed():
    """Test Ollama health returns status 'warning' when Ollama is running but configured model is missing."""
    class MockOllamaServiceMissingModel(OllamaService):
        async def check_health(self):
            return {
                "ollama_connected": True,
                "configured_model": "qwen2.5:1.5b",
                "model_available": False,
                "available_models": ["llama3:latest"],
            }

    app.dependency_overrides[get_ollama_service] = lambda: MockOllamaServiceMissingModel()
    try:
        response = client.get("/api/v1/health/ollama")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "warning"
        assert data["ollama_connected"] is True
        assert data["model_available"] is False
    finally:
        app.dependency_overrides.clear()


def test_ollama_health_endpoint_service_unreachable():
    """Test Ollama health returns 503 Service Unavailable when Ollama cannot be reached."""
    class MockOllamaServiceFailure(OllamaService):
        async def check_health(self):
            raise OllamaConnectionError("Unable to connect to Ollama service.")

    app.dependency_overrides[get_ollama_service] = lambda: MockOllamaServiceFailure()
    try:
        response = client.get("/api/v1/health/ollama")
        assert response.status_code == 503
        data = response.json()
        assert "Unable to connect to Ollama service" in data["detail"]
    finally:
        app.dependency_overrides.clear()
