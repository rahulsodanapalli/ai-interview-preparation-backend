"""GitHub MCP Read-Only Service.

Provides a genuine Model Context Protocol (MCP) interface for inspecting GitHub
repositories, reading READMEs, listing directory trees, and fetching source code.
Uses the official Python MCP SDK (`mcp`), connects via standard I/O (`stdio`) transport,
dynamically discovers server tools and schema variations, enforces strict read-only security,
guards against path traversal, adheres to size limits, and guarantees zero token leakage.
"""

import asyncio
import base64
from contextlib import asynccontextmanager
import json
import logging
import os
import shutil
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple, Union

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import MCPError

from app.core.config import settings
from app.schemas.github import (
    RepoFileContentResponse,
    RepoFileItem,
    RepoFilesListResponse,
    RepoMetadataResponse,
    RepoReadmeResponse,
    parse_repo_identifier,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================


class GitHubServiceError(Exception):
    """Base exception for all GitHub MCP service errors."""


class GitHubValidationError(GitHubServiceError):
    """Raised when repository identifier or file path format is invalid or unsafe."""


class GitHubRepoNotFoundError(GitHubServiceError):
    """Raised when the specified repository does not exist or is inaccessible."""


class GitHubFileNotFoundError(GitHubServiceError):
    """Raised when the requested file or directory does not exist in the repository."""


class GitHubPermissionError(GitHubServiceError):
    """Raised when access is denied or a forbidden operation (e.g. write tool) is attempted."""


class GitHubRateLimitError(GitHubServiceError):
    """Raised when GitHub API rate limits have been exceeded."""


class GitHubTimeoutError(GitHubServiceError):
    """Raised when requests to GitHub MCP server time out."""


class GitHubConnectionError(GitHubServiceError):
    """Raised when network connectivity or subprocess connection to GitHub MCP fails."""


class GitHubFileTooLargeError(GitHubServiceError):
    """Raised when a requested file exceeds the configured maximum byte limit."""


# ============================================================================
# Read-Only Security Guard Whitelist
# ============================================================================

READ_ONLY_TOOL_WHITELIST = {
    "search_repositories",
    "get_file_contents",
    "search_code",
    "list_commits",
    "list_branches",
    "list_tags",
    "list_releases",
    "get_commit",
    "get_latest_release",
    "get_release_by_tag",
    "get_tag",
    "list_issues",
    "issue_read",
    "list_pull_requests",
    "pull_request_read",
    "search_issues",
    "search_pull_requests",
    "search_users",
    "ui_get",
    "list_repository_collaborators",
}


# ============================================================================
# GitHub MCP Service Implementation
# ============================================================================

_UNSET = object()


class GitHubMCPService:
    """Genuine MCP client service for read-only inspection of GitHub repositories.

    Attributes:
        timeout_seconds: Timeout for MCP operations in seconds.
        max_file_size_bytes: Maximum allowed size in bytes for retrieved source files.
        server_command: Executable command for the MCP server.
        server_args: Argument list for the MCP server.
    """

    def __init__(
        self,
        token: Any = _UNSET,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        max_file_size_bytes: Optional[int] = None,
        server_command: Optional[str] = None,
        server_args: Optional[List[str]] = None,
        session_factory: Optional[Callable[[], Any]] = None,
    ):
        if token is _UNSET:
            raw_token = settings.GITHUB_PERSONAL_ACCESS_TOKEN or settings.GITHUB_TOKEN
        else:
            raw_token = token
        self._token = raw_token.strip() if raw_token and isinstance(raw_token, str) and raw_token.strip() else None
        self.timeout_seconds = timeout_seconds or settings.GITHUB_MCP_TIMEOUT_SECONDS
        self.max_file_size_bytes = max_file_size_bytes or settings.GITHUB_MAX_FILE_SIZE_BYTES
        self.server_command = server_command or settings.GITHUB_MCP_SERVER_COMMAND
        self.server_args = server_args
        self._session_factory = session_factory

    def _get_headers(self) -> Dict[str, str]:
        """Construct request headers (for backward compatibility)."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AI-Interview-Preparation-Platform/1.0",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _sanitize(self, message: str) -> str:
        """Sanitize sensitive credentials from error messages or logs."""
        if self._token and self._token in message:
            return message.replace(self._token, "***")
        return message

    def _resolve_server_params(self) -> StdioServerParameters:
        """Resolve command, arguments, and environment variables for the MCP subprocess."""
        env = os.environ.copy()
        if self._token:
            env["GITHUB_PERSONAL_ACCESS_TOKEN"] = self._token
            env["GITHUB_TOKEN"] = self._token

        if self.server_command:
            command = self.server_command
            args = self.server_args or []
        elif self._token and shutil.which("github-mcp-server"):
            # Official maintained Go-based github-mcp-server when authenticated
            command = "github-mcp-server"
            args = self.server_args or ["stdio", "--read-only"]
        else:
            # Fallback to standard stdio server via npx
            command = "npx"
            args = self.server_args or ["-y", "@modelcontextprotocol/server-github"]

        return StdioServerParameters(
            command=command,
            args=args,
            env=env,
        )

    @asynccontextmanager
    async def _get_session(self) -> AsyncGenerator[Tuple[ClientSession, Dict[str, Any]], None]:
        """Establish a genuine MCP client session over stdio and discover tools dynamically."""
        if self._session_factory:
            # Test or custom session factory injection
            async with self._session_factory() as session:
                tools_map: Dict[str, Any] = {}
                try:
                    tools_result = await session.list_tools()
                    for t in getattr(tools_result, "tools", []):
                        tools_map[t.name] = t
                except Exception as exc:
                    logger.debug("Tool discovery in injected session: %s", exc)
                yield session, tools_map
            return

        server_params = self._resolve_server_params()

        try:
            async with stdio_client(server_params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    # Handshake
                    await asyncio.wait_for(session.initialize(), timeout=self.timeout_seconds)

                    # Dynamic tool discovery
                    tools_result = await asyncio.wait_for(session.list_tools(), timeout=self.timeout_seconds)
                    tools_map = {t.name: t for t in tools_result.tools}
                    logger.debug("Discovered %d MCP tools from GitHub server", len(tools_map))

                    yield session, tools_map

        except GitHubServiceError:
            raise
        except asyncio.TimeoutError as exc:
            raise GitHubTimeoutError("Connection to GitHub MCP server timed out.") from exc
        except (ConnectionError, BrokenPipeError, FileNotFoundError) as exc:
            safe_msg = self._sanitize(str(exc))
            raise GitHubConnectionError(f"Failed to connect to GitHub MCP server: {safe_msg}") from exc
        except MCPError as exc:
            self._handle_mcp_error(exc)
        except BaseException as exc:
            # Check if AnyIO TaskGroup wrapped our application domain exception
            def find_domain_error(e: BaseException) -> Optional[GitHubServiceError]:
                if isinstance(e, GitHubServiceError):
                    return e
                if hasattr(e, "exceptions"):
                    for sub in e.exceptions:
                        found = find_domain_error(sub)
                        if found:
                            return found
                if getattr(e, "__cause__", None) and isinstance(e.__cause__, BaseException):
                    found = find_domain_error(e.__cause__)
                    if found:
                        return found
                return None

            domain_err = find_domain_error(exc)
            if domain_err:
                raise domain_err from exc

            msg = str(exc).lower()
            if "timeout" in msg:
                raise GitHubTimeoutError("Operation timed out during MCP communication.") from exc
            if "authentication required" in msg or "bad credentials" in msg:
                raise GitHubPermissionError("GitHub authentication failed. Please configure a valid token.") from exc
            safe_msg = self._sanitize(str(exc))
            raise GitHubConnectionError(f"Error communicating with GitHub MCP server: {safe_msg}") from exc

    def _handle_mcp_error(self, exc: MCPError, is_file_request: bool = False) -> None:
        """Map MCPError into fine-grained application domain exceptions."""
        raw_msg = exc.message or ""
        msg = raw_msg.lower()

        # Check for rate limiting
        if "rate limit" in msg or "429" in msg:
            raise GitHubRateLimitError(
                "GitHub API rate limit exceeded. Configure a GITHUB_TOKEN in .env to increase rate limits."
            ) from exc

        # Check for authentication / permission
        if "bad credentials" in msg or "401" in msg or "403" in msg or "authentication required" in msg:
            raise GitHubPermissionError(
                "GitHub API access denied. For private repositories, verify that your token has read permissions."
            ) from exc

        # Check for 404 / resource not found
        if (
            "not found" in msg
            or "404" in msg
            or "does not exist" in msg
            or "cannot be searched" in msg
            or "does not point to a file" in msg
            or "no such file" in msg
        ):
            if is_file_request:
                raise GitHubFileNotFoundError(self._sanitize(raw_msg)) from exc
            raise GitHubRepoNotFoundError(
                "Repository not found or you lack permission to access it."
            ) from exc

        raise GitHubServiceError(self._sanitize(raw_msg)) from exc

    async def _call_read_only_tool(
        self,
        session: ClientSession,
        tool_name: str,
        arguments: Dict[str, Any],
        tools_map: Dict[str, Any],
        is_file_request: bool = False,
    ) -> Any:
        """Enforce strict read-only security whitelist and execute tool call."""
        # 1. Enforce read-only whitelist
        if tool_name not in READ_ONLY_TOOL_WHITELIST:
            raise GitHubPermissionError(
                f"Security violation: Tool '{tool_name}' is not in the read-only MCP whitelist. "
                "Mutations and write operations are strictly forbidden."
            )

        # 2. Check tool exists
        if tools_map and tool_name not in tools_map:
            raise GitHubServiceError(f"Tool '{tool_name}' is not supported by the connected GitHub MCP server.")

        # 3. Call tool with timeout
        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise GitHubTimeoutError(f"MCP tool '{tool_name}' call timed out.") from exc
        except MCPError as exc:
            self._handle_mcp_error(exc, is_file_request=is_file_request)

        # 4. Check for is_error / isError flag in tool result
        is_error = getattr(result, "is_error", False) or getattr(result, "isError", False)
        if is_error:
            err_parts = []
            if getattr(result, "content", None):
                for c in result.content:
                    if hasattr(c, "text") and c.text:
                        err_parts.append(c.text)
                    elif hasattr(c, "resource"):
                        res_obj = getattr(c, "resource", None)
                        if res_obj and hasattr(res_obj, "text") and res_obj.text:
                            err_parts.append(res_obj.text)
            err_text = " ".join(err_parts).strip()
            fake_exc = MCPError(code=-1, message=err_text or f"Tool '{tool_name}' returned error")
            self._handle_mcp_error(fake_exc, is_file_request=is_file_request)

        # 5. Parse response content
        if not getattr(result, "content", None) or len(result.content) == 0:
            return {}

        # 5a. Look for EmbeddedResource blocks (official github-mcp-server returns file contents here)
        for c in result.content:
            if hasattr(c, "resource") and c.resource is not None:
                embedded = c.resource
                text_content = ""
                if hasattr(embedded, "text") and embedded.text is not None:
                    text_content = embedded.text
                elif hasattr(embedded, "blob") and embedded.blob is not None:
                    try:
                        text_content = base64.b64decode(embedded.blob).decode("utf-8", errors="replace")
                    except Exception:
                        text_content = str(embedded.blob)

                return {
                    "content": text_content,
                    "encoding": "utf-8",
                    "uri": str(getattr(embedded, "uri", "")) if hasattr(embedded, "uri") else "",
                    "mime_type": getattr(embedded, "mime_type", None),
                    "size": len(text_content.encode("utf-8")),
                }

        # 5b. Parse TextContent blocks
        first_content = result.content[0]
        text_data = getattr(first_content, "text", "")
        if not text_data:
            return {}

        try:
            return json.loads(text_data)
        except json.JSONDecodeError:
            return text_data

    # ========================================================================
    # Read-Only MCP Tool Operations
    # ========================================================================

    async def get_repository_metadata(self, owner_or_url: str, repo: Optional[str] = None) -> RepoMetadataResponse:
        """Retrieve repository metadata using genuine MCP tool discovery and calls.

        Args:
            owner_or_url: Repository owner or full identifier ('owner/repo' or URL).
            repo: Repository name (optional if owner_or_url includes it).

        Returns:
            RepoMetadataResponse with detailed metadata.
        """
        if repo is None:
            owner, repo_name = parse_repo_identifier(owner_or_url)
        else:
            owner, repo_name = parse_repo_identifier(f"{owner_or_url}/{repo}")

        async with self._get_session() as (session, tools_map):
            # Query the search_repositories tool
            query = f"repo:{owner}/{repo_name}"
            data = await self._call_read_only_tool(
                session=session,
                tool_name="search_repositories",
                arguments={"query": query},
                tools_map=tools_map,
                is_file_request=False,
            )

            items: List[Dict[str, Any]] = []
            if isinstance(data, dict):
                items = data.get("items", [])
            elif isinstance(data, list):
                items = data

            # Filter exact match
            matched_repo = None
            target_full_name = f"{owner}/{repo_name}".lower()
            for it in items:
                if isinstance(it, dict) and it.get("full_name", "").lower() == target_full_name:
                    matched_repo = it
                    break
            if not matched_repo and items and isinstance(items[0], dict):
                # Fallback to first search result if full_name matches loosely
                if items[0].get("name", "").lower() == repo_name.lower():
                    matched_repo = items[0]

            if not matched_repo:
                raise GitHubRepoNotFoundError(
                    f"Repository '{owner}/{repo_name}' not found or inaccessible."
                )

            # Check if README exists
            has_readme = False
            try:
                root_args = self._build_file_args("", owner, repo_name, None, tools_map)
                root_data = await self._call_read_only_tool(
                    session=session,
                    tool_name="get_file_contents",
                    arguments=root_args,
                    tools_map=tools_map,
                    is_file_request=True,
                )
                if isinstance(root_data, list):
                    for it in root_data:
                        if isinstance(it, dict) and it.get("type") == "file":
                            name = it.get("name", "").lower()
                            stem, ext = os.path.splitext(name)
                            if stem == "readme" and ext in [
                                "",
                                ".md",
                                ".markdown",
                                ".mdown",
                                ".mkdn",
                                ".rst",
                                ".txt",
                                ".text",
                                ".adoc",
                                ".asciidoc",
                                ".org",
                            ]:
                                has_readme = True
                                break
                elif isinstance(root_data, dict):
                    name = root_data.get("name", "").lower()
                    stem, ext = os.path.splitext(name)
                    if stem == "readme":
                        has_readme = True
            except Exception:
                has_readme = False

            if not has_readme:
                for cand in ["README.md", "README", "readme.md", "README.rst", "README.txt"]:
                    try:
                        readme_args = self._build_file_args(cand, owner, repo_name, None, tools_map)
                        readme_data = await self._call_read_only_tool(
                            session=session,
                            tool_name="get_file_contents",
                            arguments=readme_args,
                            tools_map=tools_map,
                            is_file_request=True,
                        )
                        if readme_data:
                            has_readme = True
                            break
                    except Exception:
                        continue

            return RepoMetadataResponse(
                owner=matched_repo.get("owner", {}).get("login", owner) if isinstance(matched_repo.get("owner"), dict) else owner,
                repo=matched_repo.get("name", repo_name),
                full_name=matched_repo.get("full_name", f"{owner}/{repo_name}"),
                description=matched_repo.get("description"),
                default_branch=matched_repo.get("default_branch", "main"),
                language=matched_repo.get("language"),
                stars_count=matched_repo.get("stargazers_count", 0) or 0,
                forks_count=matched_repo.get("forks_count", 0) or 0,
                open_issues_count=matched_repo.get("open_issues_count", 0) or 0,
                topics=matched_repo.get("topics", []) or [],
                is_private=matched_repo.get("private", False) or False,
                html_url=matched_repo.get("html_url", f"https://github.com/{owner}/{repo_name}"),
                has_readme=has_readme,
            )

    def _build_file_args(
        self,
        path: str,
        owner: str,
        repo: str,
        ref: Optional[str],
        tools_map: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Dynamically adapt arguments based on the discovered MCP tool schema."""
        tool = tools_map.get("get_file_contents")
        props = {}
        if tool:
            schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
            if isinstance(schema, dict):
                props = schema.get("properties", {})

        args: Dict[str, Any] = {
            "owner": owner,
            "repo": repo,
            "path": path,
        }

        # Check ref vs branch property naming across server versions
        if ref:
            if "ref" in props:
                args["ref"] = ref
            elif "branch" in props:
                args["branch"] = ref
            else:
                args["ref"] = ref

        return args

    @staticmethod
    def _safe_decode_content(content: Any, encoding: str) -> str:
        """Safely decode file content, handling both pre-decoded MCP text and raw base64."""
        if not content or not isinstance(content, str):
            return ""
        if encoding == "base64":
            try:
                decoded_bytes = base64.b64decode(content.strip(), validate=True)
                return decoded_bytes.decode("utf-8", errors="replace")
            except Exception:
                # If base64 validation fails (e.g. incorrect padding), the content is already plain text
                return content
        return content

    async def get_readme(
        self,
        owner_or_url: str,
        repo: Optional[str] = None,
        ref: Optional[str] = None,
    ) -> RepoReadmeResponse:
        """Fetch and decode the repository README file using MCP get_file_contents.

        Discovers the exact README path from repository root rather than assuming
        a hardcoded extension, supporting README, README.md, README.rst, README.txt, etc.

        Args:
            owner_or_url: Repository owner or identifier.
            repo: Repository name (if owner_or_url is just the owner).
            ref: Optional Git branch, tag, or commit.

        Returns:
            RepoReadmeResponse with the decoded markdown or text content.
        """
        if repo is None:
            owner, repo_name = parse_repo_identifier(owner_or_url)
        else:
            owner, repo_name = parse_repo_identifier(f"{owner_or_url}/{repo}")

        # 1. Discover exact README file from root directory listing
        target_path: Optional[str] = None
        target_name: Optional[str] = None
        html_url: Optional[str] = None

        try:
            files_list = await self.list_files(owner_or_url=f"{owner}/{repo_name}", path="", ref=ref)
            readme_candidates: List[RepoFileItem] = []
            for item in files_list.items:
                if item.type == "file":
                    name_lower = item.name.lower()
                    stem, ext = os.path.splitext(name_lower)
                    if stem == "readme" and ext in [
                        "",
                        ".md",
                        ".markdown",
                        ".mdown",
                        ".mkdn",
                        ".rst",
                        ".txt",
                        ".text",
                        ".adoc",
                        ".asciidoc",
                        ".org",
                    ]:
                        readme_candidates.append(item)

            if readme_candidates:
                def rank_readme(item: RepoFileItem) -> int:
                    nl = item.name.lower()
                    if nl == "readme.md":
                        return 0
                    if nl == "readme":
                        return 1
                    if any(nl.endswith(e) for e in [".markdown", ".mdown", ".mkdn"]):
                        return 2
                    if nl.endswith(".rst"):
                        return 3
                    if nl.endswith(".txt") or nl.endswith(".text"):
                        return 4
                    return 5

                readme_candidates.sort(key=rank_readme)
                best_match = readme_candidates[0]
                target_path = best_match.path
                target_name = best_match.name
                html_url = best_match.html_url
        except Exception as exc:
            logger.debug("Root directory listing for README discovery failed: %s", exc)

        # 2. If discovered, fetch using the exact repository file path
        if target_path:
            file_resp = await self.get_file_content(
                owner_or_url=f"{owner}/{repo_name}",
                file_path=target_path,
                ref=ref,
            )
            return RepoReadmeResponse(
                owner=owner,
                repo=repo_name,
                filename=target_name or target_path,
                content=file_resp.content,
                size_bytes=file_resp.size_bytes,
                html_url=html_url or file_resp.html_url or f"https://github.com/{owner}/{repo_name}/blob/{ref or 'main'}/{target_path}",
            )

        # 3. Fallback: try common candidate paths sequentially
        fallback_candidates = [
            "README.md",
            "README",
            "readme.md",
            "README.rst",
            "README.txt",
            "README.markdown",
            "docs/README.md",
            ".github/README.md",
        ]
        last_exception: Optional[Exception] = None
        for cand in fallback_candidates:
            try:
                file_resp = await self.get_file_content(
                    owner_or_url=f"{owner}/{repo_name}",
                    file_path=cand,
                    ref=ref,
                )
                return RepoReadmeResponse(
                    owner=owner,
                    repo=repo_name,
                    filename=cand.split("/")[-1],
                    content=file_resp.content,
                    size_bytes=file_resp.size_bytes,
                    html_url=file_resp.html_url or f"https://github.com/{owner}/{repo_name}/blob/{ref or 'main'}/{cand}",
                )
            except GitHubFileNotFoundError as exc:
                last_exception = exc
                continue
            except (GitHubRateLimitError, GitHubPermissionError, GitHubTimeoutError, GitHubConnectionError) as exc:
                raise exc
            except Exception as exc:
                last_exception = exc
                continue

        if last_exception and isinstance(last_exception, (GitHubRateLimitError, GitHubPermissionError)):
            raise last_exception

        raise GitHubFileNotFoundError(
            f"No README file found in repository '{owner}/{repo_name}'."
        )

    async def list_files(
        self,
        owner_or_url: str,
        repo: Optional[str] = None,
        path: str = "",
        ref: Optional[str] = None,
    ) -> RepoFilesListResponse:
        """List files and subdirectories at the given path within the repository.

        Args:
            owner_or_url: Repository identifier or owner.
            repo: Repository name (optional if included in identifier).
            path: Relative directory path inside repository.
            ref: Optional Git branch, tag, or commit.

        Returns:
            RepoFilesListResponse containing list of files and directories.
        """
        if repo is None:
            owner, repo_name = parse_repo_identifier(owner_or_url)
        else:
            owner, repo_name = parse_repo_identifier(f"{owner_or_url}/{repo}")

        clean_path = path.strip().lstrip("/")
        if clean_path:
            parts = clean_path.split("/")
            if ".." in parts or "." in parts:
                raise GitHubValidationError("Path traversal sequences are not allowed.")

        async with self._get_session() as (session, tools_map):
            args = self._build_file_args(clean_path, owner, repo_name, ref, tools_map)
            data = await self._call_read_only_tool(
                session=session,
                tool_name="get_file_contents",
                arguments=args,
                tools_map=tools_map,
                is_file_request=True,
            )

            if isinstance(data, dict):
                # Single file
                items = [
                    RepoFileItem(
                        name=data.get("name", clean_path.split("/")[-1] if clean_path else ""),
                        path=data.get("path", clean_path),
                        type="file" if data.get("type") == "file" else "dir",
                        size=data.get("size"),
                        html_url=data.get("html_url"),
                    )
                ]
            elif isinstance(data, list):
                items = [
                    RepoFileItem(
                        name=item.get("name", ""),
                        path=item.get("path", ""),
                        type=item.get("type", "file"),
                        size=item.get("size"),
                        html_url=item.get("html_url"),
                    )
                    for item in data
                    if isinstance(item, dict)
                ]
            else:
                items = []

            return RepoFilesListResponse(
                owner=owner,
                repo=repo_name,
                path=clean_path,
                items=items,
            )

    async def get_file_content(
        self,
        owner_or_url: str,
        file_path: str,
        repo: Optional[str] = None,
        ref: Optional[str] = None,
    ) -> RepoFileContentResponse:
        """Fetch and safely decode the contents of a specific source code file via MCP.

        Args:
            owner_or_url: Repository identifier or owner.
            file_path: Relative path to the file within the repository.
            repo: Repository name (optional if included in identifier).
            ref: Optional Git branch, tag, or commit.

        Returns:
            RepoFileContentResponse with decoded text.

        Raises:
            GitHubValidationError: If file path contains invalid or traversal characters.
            GitHubFileTooLargeError: If file exceeds maximum allowed size.
            GitHubFileNotFoundError: If the file does not exist.
        """
        if repo is None:
            owner, repo_name = parse_repo_identifier(owner_or_url)
        else:
            owner, repo_name = parse_repo_identifier(f"{owner_or_url}/{repo}")

        clean_path = file_path.strip().lstrip("/")
        if not clean_path:
            raise GitHubValidationError("File path cannot be empty.")

        parts = clean_path.split("/")
        if ".." in parts or "." in parts:
            raise GitHubValidationError("Path traversal sequences are not allowed.")

        async with self._get_session() as (session, tools_map):
            args = self._build_file_args(clean_path, owner, repo_name, ref, tools_map)
            data = await self._call_read_only_tool(
                session=session,
                tool_name="get_file_contents",
                arguments=args,
                tools_map=tools_map,
                is_file_request=True,
            )

            if isinstance(data, list):
                raise GitHubValidationError(f"Path '{clean_path}' is a directory, not a file.")

            if isinstance(data, str):
                decoded_text = data
                file_size = len(decoded_text.encode("utf-8"))
                html_url = f"https://github.com/{owner}/{repo_name}/blob/{ref or 'main'}/{clean_path}"
            else:
                encoded_content = data.get("content", "")
                encoding = data.get("encoding", "base64")
                decoded_text = self._safe_decode_content(encoded_content, encoding)
                file_size = data.get("size", len(decoded_text.encode("utf-8")))
                html_url = data.get("html_url") or f"https://github.com/{owner}/{repo_name}/blob/{ref or 'main'}/{clean_path}"

            if file_size > self.max_file_size_bytes:
                raise GitHubFileTooLargeError(
                    f"File '{clean_path}' size ({file_size} bytes) exceeds maximum allowed limit "
                    f"of {self.max_file_size_bytes} bytes."
                )

            return RepoFileContentResponse(
                owner=owner,
                repo=repo_name,
                file_path=clean_path,
                size_bytes=file_size,
                content=decoded_text,
                is_truncated=False,
                html_url=html_url,
            )


def get_github_service() -> GitHubMCPService:
    """FastAPI dependency provider for GitHubMCPService."""
    return GitHubMCPService()
