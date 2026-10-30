"""Automated unit and integration tests for GitHub MCP read-only operations and question generation.

Verifies:
- Repository identifier parsing and validation (valid/invalid formats, URLs, path traversal).
- Repository metadata, README, directory listing, and source file retrieval.
- Safe error handling for 404 Not Found, 403 Forbidden, 429 Rate Limits, Timeouts, and Connection Failures.
- File size limits (413 Payload Too Large).
- Question generation grounded in source code, README, or repository metadata.
- Graceful fallback when repository context is unavailable (context_grounded: False).
- Zero leaks of tokens or internal tracebacks.
"""

import base64
import json
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
import httpx
import pytest

from app.main import app
from app.schemas.github import (
    RepoFileContentResponse,
    RepoReadmeResponse,
    parse_repo_identifier,
)
from app.services.github_mcp_service import (
    GitHubConnectionError,
    GitHubFileNotFoundError,
    GitHubFileTooLargeError,
    GitHubMCPService,
    GitHubPermissionError,
    GitHubRateLimitError,
    GitHubRepoNotFoundError,
    GitHubTimeoutError,
    GitHubValidationError,
    get_github_service,
)
from app.services.ollama_service import (
    OllamaConnectionError,
    OllamaResponseError,
    OllamaService,
    OllamaTimeoutError,
    get_ollama_service,
)
from app.services.repo_question_service import (
    RepoQuestionService,
    get_repo_question_service,
)

client = TestClient(app)

SAMPLE_README_CONTENT = """# AI Interview Platform
A FastAPI-based interview prep backend with local Ollama LLM integration.
## Architecture
- FastAPI routes in `app/api`
- Service layer for LLM, vector storage, and GitHub MCP
"""

SAMPLE_CODE_CONTENT = """def calculate_score(answers: list[int]) -> float:
    \"\"\"Calculate interview score.\"\"\"
    return sum(answers) / len(answers) if answers else 0.0
"""

MOCK_REPO_DATA = {
    "name": "ai-interview-prep",
    "full_name": "test-org/ai-interview-prep",
    "owner": {"login": "test-org"},
    "description": "AI-powered technical interview preparation system.",
    "default_branch": "main",
    "language": "Python",
    "stargazers_count": 42,
    "forks_count": 5,
    "open_issues_count": 2,
    "topics": ["fastapi", "ai", "interview-prep"],
    "private": False,
    "html_url": "https://github.com/test-org/ai-interview-prep",
}

MOCK_README_DATA = {
    "name": "README.md",
    "path": "README.md",
    "size": len(SAMPLE_README_CONTENT.encode("utf-8")),
    "encoding": "base64",
    "content": base64.b64encode(SAMPLE_README_CONTENT.encode("utf-8")).decode("utf-8"),
    "html_url": "https://github.com/test-org/ai-interview-prep/blob/main/README.md",
}

MOCK_FILE_DATA = {
    "name": "score.py",
    "path": "app/score.py",
    "size": len(SAMPLE_CODE_CONTENT.encode("utf-8")),
    "encoding": "base64",
    "content": base64.b64encode(SAMPLE_CODE_CONTENT.encode("utf-8")).decode("utf-8"),
    "html_url": "https://github.com/test-org/ai-interview-prep/blob/main/app/score.py",
    "type": "file",
}

MOCK_CONTENTS_DIR_DATA = [
    {"name": "app", "path": "app", "type": "dir", "size": 0, "html_url": "https://github.com/test-org/ai-interview-prep/tree/main/app"},
    {"name": "README.md", "path": "README.md", "type": "file", "size": 150, "html_url": "https://github.com/test-org/ai-interview-prep/blob/main/README.md"},
]

MOCK_LLM_QUESTION_JSON = {
    "question": "How does your scoring algorithm in app/score.py handle zero-length answer lists to avoid division by zero?",
    "expected_answer_points": [
        "Checks list length before division",
        "Returns default float 0.0",
        "Edge case testing for empty collections",
    ],
    "follow_up_question": "How would you handle negative or out-of-bound scores if candidate input is corrupted?",
    "difficulty": "Medium",
    "topic": "System Architecture & Code Quality",
}


# ============================================================================
# Identifier Parsing & Input Validation Tests
# ============================================================================


def test_parse_repo_identifier_valid_formats():
    """Verify owner/repo format and GitHub URL parsing."""
    owner, repo = parse_repo_identifier("octocat/Hello-World")
    assert owner == "octocat"
    assert repo == "Hello-World"

    owner, repo = parse_repo_identifier("https://github.com/fastapi/fastapi")
    assert owner == "fastapi"
    assert repo == "fastapi"

    owner, repo = parse_repo_identifier("https://github.com/psf/requests.git")
    assert owner == "psf"
    assert repo == "requests"


def test_parse_repo_identifier_invalid_formats():
    """Verify malformed identifiers raise ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        parse_repo_identifier("")

    with pytest.raises(ValueError, match="Invalid GitHub URL"):
        parse_repo_identifier("https://gitlab.com/owner/repo")

    with pytest.raises(ValueError, match="Invalid GitHub owner"):
        parse_repo_identifier("-invalid-owner/repo")

    with pytest.raises(ValueError, match="Invalid GitHub repository name"):
        parse_repo_identifier("owner/invalid repo name with spaces!")


# ============================================================================
# Repository Validation Endpoint Tests (/github/validate)
# ============================================================================


def test_validate_repository_success():
    """Test successful repository validation and metadata retrieval."""
    class MockGitHub(GitHubMCPService):
        async def get_repository_metadata(self, owner_or_url, repo=None):
            return {
                "owner": "test-org",
                "repo": "ai-interview-prep",
                "full_name": "test-org/ai-interview-prep",
                "description": "AI-powered interview system.",
                "default_branch": "main",
                "language": "Python",
                "stars_count": 42,
                "forks_count": 5,
                "open_issues_count": 2,
                "topics": ["fastapi", "ai"],
                "is_private": False,
                "html_url": "https://github.com/test-org/ai-interview-prep",
                "has_readme": True,
            }

    app.dependency_overrides[get_github_service] = lambda: MockGitHub()
    try:
        response = client.post("/api/v1/github/validate", json={"repo_url_or_name": "test-org/ai-interview-prep"})
        assert response.status_code == 200
        data = response.json()
        assert data["owner"] == "test-org"
        assert data["repo"] == "ai-interview-prep"
        assert data["stars_count"] == 42
        assert data["has_readme"] is True
    finally:
        app.dependency_overrides.clear()


def test_validate_repository_not_found():
    """Test 404 returned when repository does not exist."""
    class MockGitHub(GitHubMCPService):
        async def get_repository_metadata(self, owner_or_url, repo=None):
            raise GitHubRepoNotFoundError("Repository not found.")

    app.dependency_overrides[get_github_service] = lambda: MockGitHub()
    try:
        response = client.post("/api/v1/github/validate", json={"repo_url_or_name": "unknown/nonexistent-repo"})
        assert response.status_code == 404
        assert "Repository not found" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_validate_repository_permission_denied():
    """Test 403 returned when access is forbidden."""
    class MockGitHub(GitHubMCPService):
        async def get_repository_metadata(self, owner_or_url, repo=None):
            raise GitHubPermissionError("Access denied.")

    app.dependency_overrides[get_github_service] = lambda: MockGitHub()
    try:
        response = client.post("/api/v1/github/validate", json={"repo_url_or_name": "secret/private-repo"})
        assert response.status_code == 403
        assert "Access denied" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_validate_repository_rate_limit():
    """Test 429 returned when GitHub rate limit is exceeded."""
    class MockGitHub(GitHubMCPService):
        async def get_repository_metadata(self, owner_or_url, repo=None):
            raise GitHubRateLimitError("Rate limit exceeded.")

    app.dependency_overrides[get_github_service] = lambda: MockGitHub()
    try:
        response = client.post("/api/v1/github/validate", json={"repo_url_or_name": "busy/popular-repo"})
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_validate_repository_timeout_and_connection_error():
    """Test 504 and 503 returned on timeout and network errors."""
    class MockTimeoutGitHub(GitHubMCPService):
        async def get_repository_metadata(self, owner_or_url, repo=None):
            raise GitHubTimeoutError("Timed out.")

    app.dependency_overrides[get_github_service] = lambda: MockTimeoutGitHub()
    try:
        resp = client.post("/api/v1/github/validate", json={"repo_url_or_name": "slow/repo"})
        assert resp.status_code == 504
    finally:
        app.dependency_overrides.clear()

    class MockConnGitHub(GitHubMCPService):
        async def get_repository_metadata(self, owner_or_url, repo=None):
            raise GitHubConnectionError("Connection failed.")

    app.dependency_overrides[get_github_service] = lambda: MockConnGitHub()
    try:
        resp = client.post("/api/v1/github/validate", json={"repo_url_or_name": "offline/repo"})
        assert resp.status_code == 503
    finally:
        app.dependency_overrides.clear()


# ============================================================================
# README Endpoint Tests (/github/readme)
# ============================================================================


def test_get_readme_success():
    """Test successful README retrieval and decoding."""
    class MockGitHub(GitHubMCPService):
        async def get_readme(self, owner_or_url, repo=None, ref=None):
            return {
                "owner": "test-org",
                "repo": "ai-interview-prep",
                "filename": "README.md",
                "content": SAMPLE_README_CONTENT,
                "size_bytes": len(SAMPLE_README_CONTENT),
                "html_url": "https://github.com/test-org/ai-interview-prep/blob/main/README.md",
            }

    app.dependency_overrides[get_github_service] = lambda: MockGitHub()
    try:
        response = client.get("/api/v1/github/readme?repo=test-org/ai-interview-prep")
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "README.md"
        assert "AI Interview Platform" in data["content"]
    finally:
        app.dependency_overrides.clear()


def test_get_readme_not_found():
    """Test 404 returned when README is missing in the repository."""
    class MockGitHub(GitHubMCPService):
        async def get_readme(self, owner_or_url, repo=None, ref=None):
            raise GitHubFileNotFoundError("README not found.")

    app.dependency_overrides[get_github_service] = lambda: MockGitHub()
    try:
        response = client.get("/api/v1/github/readme?repo=test-org/repo-without-readme")
        assert response.status_code == 404
        assert "No README file was found" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


# ============================================================================
# Files Listing & File Content Tests
# ============================================================================


def test_list_files_success():
    """Test listing files and folders in a repository."""
    class MockGitHub(GitHubMCPService):
        async def list_files(self, owner_or_url, repo=None, path="", ref=None):
            return {
                "owner": "test-org",
                "repo": "ai-interview-prep",
                "path": path,
                "items": [
                    {"name": "app", "path": "app", "type": "dir", "size": None, "html_url": None},
                    {"name": "README.md", "path": "README.md", "type": "file", "size": 150, "html_url": None},
                ],
            }

    app.dependency_overrides[get_github_service] = lambda: MockGitHub()
    try:
        response = client.get("/api/v1/github/files?repo=test-org/ai-interview-prep")
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2
        assert data["items"][0]["name"] == "app"
        assert data["items"][0]["type"] == "dir"
    finally:
        app.dependency_overrides.clear()


def test_get_file_content_success():
    """Test fetching and decoding source code file content."""
    class MockGitHub(GitHubMCPService):
        async def get_file_content(self, owner_or_url, file_path, repo=None, ref=None):
            return {
                "owner": "test-org",
                "repo": "ai-interview-prep",
                "file_path": file_path,
                "size_bytes": len(SAMPLE_CODE_CONTENT),
                "content": SAMPLE_CODE_CONTENT,
                "is_truncated": False,
                "html_url": "https://github.com/test-org/ai-interview-prep/blob/main/app/score.py",
            }

    app.dependency_overrides[get_github_service] = lambda: MockGitHub()
    try:
        payload = {"repo_url_or_name": "test-org/ai-interview-prep", "file_path": "app/score.py"}
        response = client.post("/api/v1/github/file-content", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["file_path"] == "app/score.py"
        assert "calculate_score" in data["content"]
    finally:
        app.dependency_overrides.clear()


def test_get_file_content_path_traversal_blocked():
    """Test that directory traversal sequences are rejected with 422/400."""
    payload = {"repo_url_or_name": "test-org/ai-interview-prep", "file_path": "../../etc/passwd"}
    response = client.post("/api/v1/github/file-content", json=payload)
    assert response.status_code in (400, 422)


def test_get_file_content_file_too_large():
    """Test that files exceeding size limits are rejected with 413."""
    class MockGitHub(GitHubMCPService):
        async def get_file_content(self, owner_or_url, file_path, repo=None, ref=None):
            raise GitHubFileTooLargeError("File exceeds 500 KB limit.")

    app.dependency_overrides[get_github_service] = lambda: MockGitHub()
    try:
        payload = {"repo_url_or_name": "test-org/ai-interview-prep", "file_path": "big_dump.csv"}
        response = client.post("/api/v1/github/file-content", json=payload)
        assert response.status_code == 413
        assert "File exceeds" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


# ============================================================================
# Question Generation Endpoint Tests (/github/question)
# ============================================================================


def test_generate_repo_question_grounded_in_source_file():
    """Test question generation successfully grounded in focus source file."""
    class MockGitHub(GitHubMCPService):
        async def get_file_content(self, owner_or_url, file_path, repo=None, ref=None):
            return RepoFileContentResponse(
                owner="test-org",
                repo="ai-interview-prep",
                file_path=file_path,
                size_bytes=len(SAMPLE_CODE_CONTENT),
                content=SAMPLE_CODE_CONTENT,
                is_truncated=False,
            )

    class MockOllama(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            return json.dumps(MOCK_LLM_QUESTION_JSON)

    mock_gh = MockGitHub()
    mock_ollama = MockOllama()
    mock_repo_qs = RepoQuestionService(github_service=mock_gh, ollama_service=mock_ollama)

    app.dependency_overrides[get_repo_question_service] = lambda: mock_repo_qs
    try:
        payload = {
            "repo_url_or_name": "test-org/ai-interview-prep",
            "focus_file_path": "app/score.py",
            "difficulty": "Medium",
        }
        response = client.post("/api/v1/github/question", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["context_grounded"] is True
        assert data["evidence"]["source_type"] == "source_file"
        assert data["evidence"]["source_path"] == "app/score.py"
        assert "division by zero" in data["question"].lower()
    finally:
        app.dependency_overrides.clear()


def test_generate_repo_question_grounded_in_readme_fallback():
    """Test question generation grounded in README when no focus file is specified."""
    class MockGitHub(GitHubMCPService):
        async def get_readme(self, owner_or_url, repo=None, ref=None):
            return RepoReadmeResponse(
                owner="test-org",
                repo="ai-interview-prep",
                filename="README.md",
                content=SAMPLE_README_CONTENT,
                size_bytes=len(SAMPLE_README_CONTENT),
            )

    class MockOllama(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            return json.dumps({
                "question": "How did you design the service layer in your FastAPI architecture?",
                "expected_answer_points": ["Separation of concerns", "Dependency injection", "Testability"],
                "follow_up_question": "What trade-offs did you consider?",
                "difficulty": "Medium",
                "topic": "Architecture",
            })

    mock_gh = MockGitHub()
    mock_ollama = MockOllama()
    mock_repo_qs = RepoQuestionService(github_service=mock_gh, ollama_service=mock_ollama)

    app.dependency_overrides[get_repo_question_service] = lambda: mock_repo_qs
    try:
        payload = {
            "repo_url_or_name": "test-org/ai-interview-prep",
            "difficulty": "Medium",
        }
        response = client.post("/api/v1/github/question", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["context_grounded"] is True
        assert data["evidence"]["source_type"] == "readme"
        assert data["evidence"]["source_path"] == "README.md"
    finally:
        app.dependency_overrides.clear()


def test_generate_repo_question_without_context_clear_reporting():
    """Test that when repository context cannot be retrieved, context_grounded is False without hallucination."""
    class MockGitHub(GitHubMCPService):
        async def get_readme(self, owner_or_url, repo=None, ref=None):
            raise GitHubFileNotFoundError("No README.")

        async def get_repository_metadata(self, owner_or_url, repo=None):
            raise GitHubRepoNotFoundError("No metadata.")

    class MockOllama(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            return json.dumps({
                "question": "Explain dependency inversion principle in modern Python architectures.",
                "expected_answer_points": ["High-level modules should not depend on low-level modules", "Abstractions"],
                "follow_up_question": "How does FastAPI Depends assist with this?",
                "difficulty": "Medium",
                "topic": "Software Engineering",
            })

    mock_gh = MockGitHub()
    mock_ollama = MockOllama()
    mock_repo_qs = RepoQuestionService(github_service=mock_gh, ollama_service=mock_ollama)

    app.dependency_overrides[get_repo_question_service] = lambda: mock_repo_qs
    try:
        payload = {
            "repo_url_or_name": "empty-user/empty-repo",
            "difficulty": "Medium",
        }
        response = client.post("/api/v1/github/question", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["context_grounded"] is False
        assert data["evidence"]["source_type"] == "none"
        assert "No readable source code" in data["grounding_note"]
    finally:
        app.dependency_overrides.clear()


def test_generate_repo_question_ollama_unavailable():
    """Test 503 returned when Ollama is unreachable during question generation."""
    class MockOllama(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            raise OllamaConnectionError("Ollama connection refused.")

    mock_repo_qs = RepoQuestionService(ollama_service=MockOllama())
    app.dependency_overrides[get_repo_question_service] = lambda: mock_repo_qs
    try:
        payload = {"repo_url_or_name": "test-org/ai-interview-prep"}
        response = client.post("/api/v1/github/question", json=payload)
        assert response.status_code == 503
        assert "Service unavailable" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_generate_repo_question_ollama_malformed_json():
    """Test 502 returned when Ollama produces malformed non-JSON output."""
    class MockOllama(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            return "Sorry, I cannot produce valid JSON right now."

    mock_repo_qs = RepoQuestionService(ollama_service=MockOllama())
    app.dependency_overrides[get_repo_question_service] = lambda: mock_repo_qs
    try:
        payload = {"repo_url_or_name": "test-org/ai-interview-prep"}
        response = client.post("/api/v1/github/question", json=payload)
        assert response.status_code == 502
        assert "AI model generation error" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


# ============================================================================
# Root and Navigation Check
# ============================================================================


def test_root_includes_github_validate_link():
    """Verify root endpoint exposes github_validate documentation link."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "github_validate" in data
    assert data["github_validate"] == "/api/v1/github/validate"


# ============================================================================
# Security, Headers, & Token Secrecy Tests
# ============================================================================


@pytest.mark.anyio
async def test_github_service_headers_omit_auth_when_token_empty():
    """Verify that Authorization header is strictly omitted when no token is configured."""
    service_no_token = GitHubMCPService(token=None)
    headers = service_no_token._get_headers()
    assert "Authorization" not in headers
    assert "User-Agent" in headers

    service_blank_token = GitHubMCPService(token="   ")
    headers_blank = service_blank_token._get_headers()
    assert "Authorization" not in headers_blank


@pytest.mark.anyio
async def test_github_service_headers_include_bearer_when_token_present():
    """Verify that Authorization header includes Bearer when token is set."""
    service_with_token = GitHubMCPService(token="ghp_testSecretToken12345")
    headers = service_with_token._get_headers()
    assert headers.get("Authorization") == "Bearer ghp_testSecretToken12345"


@pytest.mark.anyio
async def test_github_service_error_never_leaks_token():
    """Verify that exceptions and error messages never leak the GitHub token."""
    secret = "ghp_super_secret_pat_99999"
    service = GitHubMCPService(token=secret)

    # Trigger validation error
    with pytest.raises(Exception) as exc_info:
        await service.get_file_content("test-org/repo", "../secret_file.py")
    assert secret not in str(exc_info.value)


def test_generate_repo_question_fallback_to_metadata():
    """Test question generation falls back to repository metadata when README is missing."""
    class MockGitHub(GitHubMCPService):
        async def get_readme(self, owner_or_url, repo=None, ref=None):
            raise GitHubFileNotFoundError("No README.")

        async def get_repository_metadata(self, owner_or_url, repo=None):
            from app.schemas.github import RepoMetadataResponse
            return RepoMetadataResponse(
                owner="test-org",
                repo="metadata-only-repo",
                full_name="test-org/metadata-only-repo",
                description="FastAPI distributed data pipeline",
                default_branch="main",
                language="Python",
                stars_count=10,
                forks_count=1,
                open_issues_count=0,
                topics=["data-pipeline", "fastapi"],
                is_private=False,
                html_url="https://github.com/test-org/metadata-only-repo",
                has_readme=False,
            )

    class MockOllama(OllamaService):
        async def generate_chat_completion(self, messages, model=None, format=None):
            return json.dumps({
                "question": "How do you structure data pipelines in FastAPI?",
                "expected_answer_points": ["Async task workers", "Batch processing"],
                "follow_up_question": "How do you handle worker crashes?",
                "difficulty": "Medium",
                "topic": "Data Pipelines",
            })

    mock_gh = MockGitHub()
    mock_ollama = MockOllama()
    mock_repo_qs = RepoQuestionService(github_service=mock_gh, ollama_service=mock_ollama)

    app.dependency_overrides[get_repo_question_service] = lambda: mock_repo_qs
    try:
        payload = {"repo_url_or_name": "test-org/metadata-only-repo"}
        response = client.post("/api/v1/github/question", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["context_grounded"] is True
        assert data["evidence"]["source_type"] == "repo_metadata"
        assert "metadata" in data["grounding_note"].lower()
    finally:
        app.dependency_overrides.clear()


# ============================================================================
# Genuine MCP Protocol & Mocked Session Tests
# ============================================================================


class DummyTool:
    def __init__(self, name: str, description: str = "", input_schema: dict = None):
        self.name = name
        self.description = description
        self.input_schema = input_schema or {"type": "object", "properties": {}}


class DummyContent:
    def __init__(self, text: str):
        self.text = text


class DummyResource:
    def __init__(self, text: str = "", blob: str = None, uri: str = "", mime_type: str = "text/plain"):
        self.text = text
        self.blob = blob
        self.uri = uri
        self.mime_type = mime_type


class DummyEmbeddedResource:
    def __init__(self, resource: DummyResource):
        self.resource = resource


class DummyCallToolResult:
    def __init__(self, text: str = "", is_error: bool = False, content: list = None):
        if content is not None:
            self.content = content
        else:
            self.content = [DummyContent(text)]
        self.isError = is_error
        self.is_error = is_error


@pytest.mark.anyio
async def test_mcp_server_params_resolution_environment():
    """Verify that server parameters securely set tokens in environment."""
    service = GitHubMCPService(token="ghp_secure_env_pat_123")
    params = service._resolve_server_params()
    assert params.env.get("GITHUB_PERSONAL_ACCESS_TOKEN") == "ghp_secure_env_pat_123"
    assert params.env.get("GITHUB_TOKEN") == "ghp_secure_env_pat_123"


@pytest.mark.anyio
async def test_mcp_read_only_whitelist_enforcement():
    """Verify write tools are strictly rejected by the read-only security guard."""
    from contextlib import asynccontextmanager
    from app.services.github_mcp_service import READ_ONLY_TOOL_WHITELIST

    assert "search_repositories" in READ_ONLY_TOOL_WHITELIST
    assert "get_file_contents" in READ_ONLY_TOOL_WHITELIST
    assert "create_or_update_file" not in READ_ONLY_TOOL_WHITELIST
    assert "push_files" not in READ_ONLY_TOOL_WHITELIST

    mock_session = AsyncMock()

    @asynccontextmanager
    async def mock_session_factory():
        yield mock_session

    service = GitHubMCPService(session_factory=mock_session_factory)
    with pytest.raises(GitHubPermissionError, match="not in the read-only MCP whitelist"):
        await service._call_read_only_tool(
            session=mock_session,
            tool_name="create_or_update_file",
            arguments={"owner": "test", "repo": "test", "path": "x.py"},
            tools_map={},
        )


@pytest.mark.anyio
async def test_mcp_tool_discovery_and_metadata_retrieval_mocked():
    """Verify dynamic tool discovery and repository metadata retrieval via MCP search_repositories."""
    from contextlib import asynccontextmanager

    mock_session = AsyncMock()
    # Mock list_tools
    tools_result = AsyncMock()
    tools_result.tools = [
        DummyTool("search_repositories", "Search repos"),
        DummyTool("get_file_contents", "Get contents"),
    ]
    mock_session.list_tools.return_value = tools_result

    # Mock call_tool
    async def mock_call_tool(tool_name, arguments):
        if tool_name == "search_repositories":
            payload = {"items": [MOCK_REPO_DATA]}
            return DummyCallToolResult(json.dumps(payload))
        if tool_name == "get_file_contents":
            return DummyCallToolResult(json.dumps(MOCK_README_DATA))
        raise ValueError(f"Unexpected tool: {tool_name}")

    mock_session.call_tool.side_effect = mock_call_tool

    @asynccontextmanager
    async def mock_session_factory():
        yield mock_session

    service = GitHubMCPService(session_factory=mock_session_factory)
    meta = await service.get_repository_metadata("test-org/ai-interview-prep")
    assert meta.owner == "test-org"
    assert meta.repo == "ai-interview-prep"
    assert meta.stars_count == 42
    assert meta.has_readme is True


@pytest.mark.anyio
async def test_mcp_get_readme_and_safe_decoding_mocked():
    """Verify get_readme retrieves and decodes base64 content via MCP."""
    from contextlib import asynccontextmanager

    mock_session = AsyncMock()
    tools_result = AsyncMock()
    tools_result.tools = [DummyTool("get_file_contents", "Get contents")]
    mock_session.list_tools.return_value = tools_result

    async def mock_call_tool(tool_name, arguments):
        if tool_name == "get_file_contents":
            return DummyCallToolResult(json.dumps(MOCK_README_DATA))
        raise ValueError(f"Unexpected tool: {tool_name}")

    mock_session.call_tool.side_effect = mock_call_tool

    @asynccontextmanager
    async def mock_session_factory():
        yield mock_session

    service = GitHubMCPService(session_factory=mock_session_factory)
    readme = await service.get_readme("test-org/ai-interview-prep")
    assert readme.filename == "README.md"
    assert "FastAPI-based interview prep" in readme.content


@pytest.mark.anyio
async def test_mcp_list_files_mocked():
    """Verify list_files parses directory items from MCP get_file_contents."""
    from contextlib import asynccontextmanager

    mock_session = AsyncMock()
    tools_result = AsyncMock()
    tools_result.tools = [DummyTool("get_file_contents", "Get contents")]
    mock_session.list_tools.return_value = tools_result

    async def mock_call_tool(tool_name, arguments):
        if tool_name == "get_file_contents":
            return DummyCallToolResult(json.dumps(MOCK_CONTENTS_DIR_DATA))
        raise ValueError(f"Unexpected tool: {tool_name}")

    mock_session.call_tool.side_effect = mock_call_tool

    @asynccontextmanager
    async def mock_session_factory():
        yield mock_session

    service = GitHubMCPService(session_factory=mock_session_factory)
    files = await service.list_files("test-org/ai-interview-prep", path="")
    assert len(files.items) == 2
    assert files.items[0].name == "app"
    assert files.items[0].type == "dir"
    assert files.items[1].name == "README.md"
    assert files.items[1].type == "file"


@pytest.mark.anyio
async def test_mcp_error_mappings():
    """Verify MCPError translates to appropriate application domain exceptions."""
    from mcp.shared.exceptions import MCPError

    service = GitHubMCPService()

    # Rate limit
    with pytest.raises(GitHubRateLimitError):
        service._handle_mcp_error(MCPError(code=-1, message="API rate limit exceeded for user"))

    # Permission
    with pytest.raises(GitHubPermissionError):
        service._handle_mcp_error(MCPError(code=-1, message="Bad credentials (401)"))

    # Repo Not Found
    with pytest.raises(GitHubRepoNotFoundError):
        service._handle_mcp_error(MCPError(code=-1, message="Resource not found (404)"), is_file_request=False)

    # File Not Found
    with pytest.raises(GitHubFileNotFoundError):
        service._handle_mcp_error(MCPError(code=-1, message="Resource not found (404)"), is_file_request=True)


@pytest.mark.anyio
async def test_mcp_token_sanitization_in_errors():
    """Verify secret token is sanitized from all error messages and representations."""
    secret = "ghp_mcp_secret_token_abcdef"
    service = GitHubMCPService(token=secret)

    sanitized = service._sanitize(f"Failed to connect to host with token {secret}")
    assert secret not in sanitized
    assert "***" in sanitized


@pytest.mark.anyio
async def test_mcp_build_file_args_schema_adaptation():
    """Verify _build_file_args adapts ref vs branch according to discovered schema."""
    service = GitHubMCPService()

    # Tool schema with 'ref'
    tools_with_ref = {
        "get_file_contents": DummyTool(
            "get_file_contents",
            input_schema={"type": "object", "properties": {"owner": {}, "repo": {}, "path": {}, "ref": {}}},
        )
    }
    args_ref = service._build_file_args("README.md", "octocat", "Hello-World", "main", tools_with_ref)
    assert args_ref["ref"] == "main"
    assert "branch" not in args_ref

    # Tool schema with 'branch'
    tools_with_branch = {
        "get_file_contents": DummyTool(
            "get_file_contents",
            input_schema={"type": "object", "properties": {"owner": {}, "repo": {}, "path": {}, "branch": {}}},
        )
    }
    args_branch = service._build_file_args("README.md", "octocat", "Hello-World", "feature", tools_with_branch)
    assert args_branch["branch"] == "feature"
    assert "ref" not in args_branch


@pytest.mark.anyio
async def test_mcp_get_readme_resolves_readme_without_extension():
    """Verify get_readme resolves README without extension (like octocat/Hello-World)."""
    from contextlib import asynccontextmanager

    mock_session = AsyncMock()
    tools_result = AsyncMock()
    tools_result.tools = [DummyTool("get_file_contents", "Get contents")]
    mock_session.list_tools.return_value = tools_result

    # Mock directory listing with 'README' (no extension)
    mock_hello_world_listing = [
        {
            "name": "README",
            "path": "README",
            "type": "file",
            "size": 13,
            "html_url": "https://github.com/octocat/Hello-World/blob/master/README",
        }
    ]

    async def mock_call_tool(tool_name, arguments):
        if tool_name == "get_file_contents":
            path = arguments.get("path", "")
            if path in ("", "/"):
                return DummyCallToolResult(json.dumps(mock_hello_world_listing))
            if path == "README":
                return DummyCallToolResult(
                    content=[
                        DummyContent("successfully downloaded text file (SHA: 980a0d5f19a64b4b30a87d4206aade58726b60e3)"),
                        DummyEmbeddedResource(DummyResource(text="Hello World!\n")),
                    ]
                )
        raise ValueError(f"Unexpected tool/args: {tool_name} {arguments}")

    mock_session.call_tool.side_effect = mock_call_tool

    @asynccontextmanager
    async def mock_session_factory():
        yield mock_session

    service = GitHubMCPService(session_factory=mock_session_factory)
    readme = await service.get_readme("octocat/Hello-World")
    assert readme.filename == "README"
    assert readme.content == "Hello World!\n"
    assert readme.size_bytes == 13


@pytest.mark.anyio
async def test_mcp_get_readme_resolves_readme_rst():
    """Verify get_readme resolves README.rst when present."""
    from contextlib import asynccontextmanager

    mock_session = AsyncMock()
    tools_result = AsyncMock()
    tools_result.tools = [DummyTool("get_file_contents", "Get contents")]
    mock_session.list_tools.return_value = tools_result

    mock_rst_listing = [
        {"name": "setup.py", "path": "setup.py", "type": "file", "size": 200},
        {"name": "README.rst", "path": "README.rst", "type": "file", "size": 50},
    ]

    async def mock_call_tool(tool_name, arguments):
        if tool_name == "get_file_contents":
            path = arguments.get("path", "")
            if path in ("", "/"):
                return DummyCallToolResult(json.dumps(mock_rst_listing))
            if path == "README.rst":
                return DummyCallToolResult(
                    content=[
                        DummyContent("successfully downloaded text file"),
                        DummyEmbeddedResource(DummyResource(text="Documentation in reStructuredText\n============================\n")),
                    ]
                )
        raise ValueError(f"Unexpected tool/args: {tool_name} {arguments}")

    mock_session.call_tool.side_effect = mock_call_tool

    @asynccontextmanager
    async def mock_session_factory():
        yield mock_session

    service = GitHubMCPService(session_factory=mock_session_factory)
    readme = await service.get_readme("python-org/sample-repo")
    assert readme.filename == "README.rst"
    assert "reStructuredText" in readme.content


@pytest.mark.anyio
async def test_mcp_tool_error_flag_raises_file_not_found():
    """Verify that is_error=True from MCP tool properly triggers GitHubFileNotFoundError."""
    from contextlib import asynccontextmanager

    mock_session = AsyncMock()
    tools_result = AsyncMock()
    tools_result.tools = [DummyTool("get_file_contents", "Get contents")]
    mock_session.list_tools.return_value = tools_result

    # When querying a file that doesn't exist, MCP server returns is_error=True
    async def mock_call_tool(tool_name, arguments):
        if tool_name == "get_file_contents":
            return DummyCallToolResult(
                text="Failed to get file contents. The path does not point to a file or directory, or the file does not exist in the repository.",
                is_error=True,
            )
        raise ValueError(f"Unexpected: {tool_name}")

    mock_session.call_tool.side_effect = mock_call_tool

    @asynccontextmanager
    async def mock_session_factory():
        yield mock_session

    service = GitHubMCPService(session_factory=mock_session_factory)
    with pytest.raises(GitHubFileNotFoundError):
        await service.get_file_content("octocat/Hello-World", "README.md")


