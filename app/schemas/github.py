"""Pydantic schemas for GitHub repository integration, MCP tools, and interview questions."""

import re
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


# Regular expression for GitHub owner (user/org) and repository names
OWNER_REGEX = re.compile(r"^[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$")
REPO_NAME_REGEX = re.compile(r"^[a-zA-Z0-9_.-]{1,100}$")


def parse_repo_identifier(identifier: str) -> tuple[str, str]:
    """Parse and validate GitHub repository identifier from 'owner/repo' or GitHub URL.

    Args:
        identifier: GitHub URL (https://github.com/owner/repo) or 'owner/repo' format.

    Returns:
        Tuple of (owner, repo).

    Raises:
        ValueError: If the identifier cannot be parsed or contains invalid characters.
    """
    clean = (identifier or "").strip()
    if not clean:
        raise ValueError("Repository identifier cannot be empty.")

    # Remove trailing slashes and .git suffix
    if clean.endswith("/"):
        clean = clean.rstrip("/")
    if clean.endswith(".git"):
        clean = clean[:-4]

    # Handle full GitHub URLs
    if clean.startswith("http://") or clean.startswith("https://"):
        parts = clean.split("/")
        if len(parts) >= 5 and "github.com" in parts[2].lower():
            owner, repo = parts[3], parts[4]
        else:
            raise ValueError("Invalid GitHub URL. Must be in format https://github.com/owner/repo.")
    elif "/" in clean:
        parts = clean.split("/")
        if len(parts) != 2:
            raise ValueError("Repository identifier must be in 'owner/repo' format.")
        owner, repo = parts[0].strip(), parts[1].strip()
    else:
        raise ValueError("Repository identifier must be 'owner/repo' or a full GitHub repository URL.")

    if not OWNER_REGEX.match(owner):
        raise ValueError(
            f"Invalid GitHub owner/organization name '{owner}'. "
            "Must be alphanumeric with single hyphens and up to 39 characters."
        )
    if not REPO_NAME_REGEX.match(repo):
        raise ValueError(
            f"Invalid GitHub repository name '{repo}'. "
            "Must contain only alphanumeric characters, periods, hyphens, and underscores (max 100 characters)."
        )

    return owner, repo


# ============================================================================
# Request Schemas
# ============================================================================


class RepoIdentifierRequest(BaseModel):
    """Request schema for validating or inspecting a repository."""

    repo_url_or_name: str = Field(
        ...,
        description="GitHub repository identifier, e.g. 'octocat/Hello-World' or 'https://github.com/octocat/Hello-World'.",
        min_length=3,
        max_length=256,
        examples=["torvalds/linux", "https://github.com/fastapi/fastapi"],
    )

    @field_validator("repo_url_or_name")
    @classmethod
    def validate_identifier(cls, v: str) -> str:
        parse_repo_identifier(v)
        return v.strip()


class RepoFileRequest(BaseModel):
    """Request schema for reading a specific repository file."""

    repo_url_or_name: str = Field(
        ...,
        description="GitHub repository identifier ('owner/repo' or URL).",
    )
    file_path: str = Field(
        ...,
        description="Relative path to the source file within the repository, e.g. 'src/main.py'.",
        min_length=1,
        max_length=500,
    )
    ref: Optional[str] = Field(
        default=None,
        description="Optional Git branch, tag, or commit SHA. Defaults to repository's default branch.",
    )

    @field_validator("repo_url_or_name")
    @classmethod
    def validate_repo(cls, v: str) -> str:
        parse_repo_identifier(v)
        return v.strip()

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        clean = v.strip().lstrip("/")
        if not clean:
            raise ValueError("File path cannot be empty.")
        # Prevent directory traversal attacks
        parts = clean.split("/")
        if ".." in parts or "." in parts:
            raise ValueError("Invalid file path: directory traversal sequences (..) are not allowed.")
        return clean


class RepoQuestionRequest(BaseModel):
    """Request schema for generating an interview question grounded in repository context."""

    repo_url_or_name: str = Field(
        ...,
        description="GitHub repository identifier ('owner/repo' or URL).",
    )
    role: Optional[str] = Field(
        default="Software Engineer",
        description="Target job role for the interview question.",
        max_length=100,
    )
    focus_file_path: Optional[str] = Field(
        default=None,
        description="Optional path to a specific file to probe (e.g. 'app/main.py'). If omitted, uses README and repo overview.",
        max_length=500,
    )
    topic: Optional[str] = Field(
        default="System Architecture & Code Quality",
        description="Technical focus area or domain.",
        max_length=100,
    )
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(
        default="Medium",
        description="Target difficulty level of the interview question.",
    )
    ref: Optional[str] = Field(
        default=None,
        description="Optional branch or tag name.",
    )

    @field_validator("repo_url_or_name")
    @classmethod
    def validate_repo(cls, v: str) -> str:
        parse_repo_identifier(v)
        return v.strip()


# ============================================================================
# Response Schemas
# ============================================================================


class RepoMetadataResponse(BaseModel):
    """Metadata response for a validated GitHub repository."""

    owner: str = Field(..., description="Repository owner or organization name.")
    repo: str = Field(..., description="Repository name.")
    full_name: str = Field(..., description="Full repository name ('owner/repo').")
    description: Optional[str] = Field(None, description="Repository description.")
    default_branch: str = Field("main", description="Default branch name.")
    language: Optional[str] = Field(None, description="Primary programming language.")
    stars_count: int = Field(0, description="Number of stargazers.")
    forks_count: int = Field(0, description="Number of forks.")
    open_issues_count: int = Field(0, description="Count of open issues.")
    topics: List[str] = Field(default_factory=list, description="Repository topics/tags.")
    is_private: bool = Field(False, description="Whether the repository is private.")
    html_url: str = Field(..., description="GitHub repository web URL.")
    has_readme: bool = Field(False, description="Whether a README was detected.")


class RepoReadmeResponse(BaseModel):
    """Response containing the decoded repository README text."""

    owner: str = Field(..., description="Repository owner.")
    repo: str = Field(..., description="Repository name.")
    filename: str = Field(..., description="README file name (e.g. README.md).")
    content: str = Field(..., description="Decoded README content in plain text / markdown.")
    size_bytes: int = Field(..., description="Size of the README content in bytes.")
    html_url: Optional[str] = Field(None, description="Direct web URL to the README on GitHub.")


class RepoFileItem(BaseModel):
    """Individual item (file or directory) in a repository listing."""

    name: str = Field(..., description="Name of the file or directory.")
    path: str = Field(..., description="Path relative to repository root.")
    type: Literal["file", "dir", "symlink", "submodule"] = Field(..., description="Type of item.")
    size: Optional[int] = Field(None, description="Size in bytes (files only).")
    html_url: Optional[str] = Field(None, description="Web URL on GitHub.")


class RepoFilesListResponse(BaseModel):
    """Response containing file and folder listing for a repository directory."""

    owner: str = Field(..., description="Repository owner.")
    repo: str = Field(..., description="Repository name.")
    path: str = Field("", description="Queried path within the repository.")
    items: List[RepoFileItem] = Field(default_factory=list, description="List of files and directories.")


class RepoFileContentResponse(BaseModel):
    """Response containing the decoded source content of a single file."""

    owner: str = Field(..., description="Repository owner.")
    repo: str = Field(..., description="Repository name.")
    file_path: str = Field(..., description="Path of the source file.")
    size_bytes: int = Field(..., description="File size in bytes.")
    content: str = Field(..., description="Decoded source text.")
    is_truncated: bool = Field(False, description="Whether content was truncated due to size limits.")
    html_url: Optional[str] = Field(None, description="Direct web URL on GitHub.")


class RepoEvidence(BaseModel):
    """Evidence or excerpt from the repository used to ground the interview question."""

    source_type: Literal["readme", "source_file", "repo_metadata", "none"] = Field(
        ...,
        description="Type of source used as evidence.",
    )
    source_path: str = Field(..., description="File path or identifier of the evidence source.")
    excerpt: str = Field(..., description="Extracted snippet or context that grounded the question.")


class RepoQuestionResponse(BaseModel):
    """Response schema for a repository-grounded interview question."""

    question: str = Field(..., description="Technical interview question grounded in the repository.")
    expected_answer_points: List[str] = Field(
        ...,
        description="Key architectural, design, or technical points expected in a strong answer.",
    )
    follow_up_question: str = Field(
        ...,
        description="Follow-up question to probe deeper into implementation decisions.",
    )
    difficulty: Literal["Easy", "Medium", "Hard"] = Field(
        ...,
        description="Difficulty level of the question.",
    )
    topic: str = Field(..., description="Technical focus area.")
    repository: str = Field(..., description="Full repository name ('owner/repo').")
    context_grounded: bool = Field(
        ...,
        description="True if grounded in actual repository context; False if context was unavailable.",
    )
    evidence: Optional[RepoEvidence] = Field(
        None,
        description="Specific source file or README excerpt used as grounding evidence.",
    )
    grounding_note: Optional[str] = Field(
        None,
        description="Note explaining whether and how the question was grounded.",
    )
