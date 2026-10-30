"""API routes for GitHub MCP repository inspection and repository-grounded question generation."""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.schemas.github import (
    RepoFileContentResponse,
    RepoFileRequest,
    RepoFilesListResponse,
    RepoIdentifierRequest,
    RepoMetadataResponse,
    RepoQuestionRequest,
    RepoQuestionResponse,
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
    GitHubServiceError,
    GitHubTimeoutError,
    GitHubValidationError,
    get_github_service,
)
from app.services.ollama_service import (
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from app.services.repo_question_service import (
    RepoQuestionService,
    get_repo_question_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/github", tags=["GitHub MCP"])


@router.post(
    "/validate",
    response_model=RepoMetadataResponse,
    status_code=status.HTTP_200_OK,
    summary="Validate and Inspect GitHub Repository",
    description="Validates repository format, checks accessibility via GitHub API, and returns repository metadata.",
    responses={
        200: {"description": "Repository validated and metadata retrieved successfully"},
        400: {"description": "Invalid repository format or URL"},
        403: {"description": "Access denied or token lacking read permissions"},
        404: {"description": "Repository not found"},
        429: {"description": "GitHub API rate limit exceeded"},
        503: {"description": "GitHub API service unavailable"},
        504: {"description": "GitHub API timed out"},
    },
)
async def validate_repository(
    payload: RepoIdentifierRequest,
    github_service: GitHubMCPService = Depends(get_github_service),
) -> RepoMetadataResponse:
    """Validate repository identifier and retrieve metadata."""
    try:
        return await github_service.get_repository_metadata(payload.repo_url_or_name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except GitHubValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except GitHubRepoNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except GitHubPermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except GitHubRateLimitError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except GitHubTimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)) from exc
    except GitHubConnectionError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except GitHubServiceError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.get(
    "/readme",
    response_model=RepoReadmeResponse,
    status_code=status.HTTP_200_OK,
    summary="Fetch Decoded Repository README",
    description="Retrieves and decodes the default README file for the given repository.",
    responses={
        200: {"description": "README retrieved successfully"},
        400: {"description": "Invalid repository identifier"},
        403: {"description": "Access denied or token lacking permissions"},
        404: {"description": "README or repository not found"},
        429: {"description": "GitHub API rate limit exceeded"},
        503: {"description": "GitHub API connection failure"},
        504: {"description": "GitHub API timed out"},
    },
)
async def get_repository_readme(
    repo: str = Query(..., description="Repository identifier ('owner/repo' or URL)"),
    ref: Optional[str] = Query(None, description="Optional Git branch, tag, or commit SHA"),
    github_service: GitHubMCPService = Depends(get_github_service),
) -> RepoReadmeResponse:
    """Fetch repository README content."""
    try:
        return await github_service.get_readme(owner_or_url=repo, ref=ref)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except GitHubRepoNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except GitHubFileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No README file was found in the specified repository.",
        ) from exc
    except GitHubPermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except GitHubRateLimitError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except GitHubTimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)) from exc
    except GitHubConnectionError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except GitHubServiceError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.get(
    "/files",
    response_model=RepoFilesListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Files and Folders in Repository",
    description="Lists contents of a repository directory at the specified relative path.",
    responses={
        200: {"description": "Directory listing retrieved successfully"},
        400: {"description": "Invalid repository or path traversal sequence"},
        403: {"description": "Access denied or token lacking permissions"},
        404: {"description": "Directory or repository not found"},
        429: {"description": "GitHub API rate limit exceeded"},
        503: {"description": "GitHub API connection failure"},
        504: {"description": "GitHub API timed out"},
    },
)
async def list_repository_files(
    repo: str = Query(..., description="Repository identifier ('owner/repo' or URL)"),
    path: str = Query("", description="Relative path within the repository (default root)"),
    ref: Optional[str] = Query(None, description="Optional Git branch, tag, or commit SHA"),
    github_service: GitHubMCPService = Depends(get_github_service),
) -> RepoFilesListResponse:
    """List repository files and subdirectories."""
    try:
        return await github_service.list_files(owner_or_url=repo, path=path, ref=ref)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except GitHubValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except GitHubRepoNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except GitHubFileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except GitHubPermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except GitHubRateLimitError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except GitHubTimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)) from exc
    except GitHubConnectionError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except GitHubServiceError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post(
    "/file-content",
    response_model=RepoFileContentResponse,
    status_code=status.HTTP_200_OK,
    summary="Read Specific Repository Source File",
    description="Fetches and decodes the content of a single source file within the configured byte limit.",
    responses={
        200: {"description": "File content retrieved and decoded successfully"},
        400: {"description": "Invalid file path or path traversal attempt"},
        403: {"description": "Access denied or token lacking permissions"},
        404: {"description": "File or repository not found"},
        413: {"description": "File size exceeds configured maximum limit"},
        429: {"description": "GitHub API rate limit exceeded"},
        503: {"description": "GitHub API connection failure"},
        504: {"description": "GitHub API timed out"},
    },
)
async def get_file_content(
    payload: RepoFileRequest,
    github_service: GitHubMCPService = Depends(get_github_service),
) -> RepoFileContentResponse:
    """Fetch content of a source file."""
    try:
        return await github_service.get_file_content(
            owner_or_url=payload.repo_url_or_name,
            file_path=payload.file_path,
            ref=payload.ref,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except GitHubValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except GitHubFileTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail=str(exc)) from exc
    except GitHubRepoNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except GitHubFileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except GitHubPermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except GitHubRateLimitError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except GitHubTimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(exc)) from exc
    except GitHubConnectionError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except GitHubServiceError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post(
    "/question",
    response_model=RepoQuestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Repository-Grounded Interview Question",
    description="Retrieves repository context (README or source code) via GitHub MCP and prompts local Ollama (qwen2.5:1.5b) to generate a grounded technical question with citations.",
    responses={
        200: {"description": "Interview question generated successfully"},
        400: {"description": "Invalid repository identifier or parameter"},
        403: {"description": "GitHub permission denied"},
        404: {"description": "Repository not found"},
        429: {"description": "GitHub API rate limit exceeded"},
        502: {"description": "Local AI model returned invalid JSON structure"},
        503: {"description": "Ollama or GitHub service unavailable"},
        504: {"description": "Ollama or GitHub request timed out"},
    },
)
async def generate_repo_question(
    payload: RepoQuestionRequest,
    repo_question_service: RepoQuestionService = Depends(get_repo_question_service),
) -> RepoQuestionResponse:
    """Generate interview question grounded in repository source code or README."""
    try:
        return await repo_question_service.generate_repo_question(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except GitHubRepoNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except GitHubPermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except GitHubRateLimitError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except (OllamaConnectionError, GitHubConnectionError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unavailable: {exc}",
        ) from exc
    except (OllamaTimeoutError, GitHubTimeoutError) as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Request timed out: {exc}",
        ) from exc
    except OllamaResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI model generation error: {exc}",
        ) from exc
    except GitHubServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GitHub service error: {exc}",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error generating repository question: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal error occurred while generating the repository interview question.",
        ) from exc
