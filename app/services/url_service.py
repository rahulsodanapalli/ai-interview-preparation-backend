"""URL Ingestion Service with SSRF protection, redirect revalidation, and GitHub MCP integration."""

import ipaddress
import logging
import socket
from typing import Optional, Tuple
from urllib.parse import urljoin, urlparse
import uuid

from fastapi import Depends
import httpx

from app.schemas.document import (
    SourceProcessingStatus,
    SourceType,
    SourceUploadResponse,
)
from app.services.chunking_service import TextChunkingService, get_chunking_service
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.github_mcp_service import GitHubMCPService, get_github_service
from app.services.pinecone_service import PineconeService, get_pinecone_service
from app.services.source_service import SourceService, get_source_service
from app.utils.html_parser import HTMLTextExtractor

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================


class URLServiceError(Exception):
    """Base exception for URL ingestion errors."""
    pass


class InvalidURLError(URLServiceError):
    """Raised when the URL format or scheme is invalid."""
    pass


class SSRFSecurityError(URLServiceError):
    """Raised when a URL targets a private, loopback, link-local, or unsafe IP destination."""
    pass


class URLFetchError(URLServiceError):
    """Raised when fetching the URL fails due to network, HTTP, or timeout issues."""
    pass


class ProtectedURLError(URLServiceError):
    """Raised when the URL returns 401 Unauthorized or 403 Forbidden (paywall / login)."""
    pass


class ContentTooLargeError(URLServiceError):
    """Raised when the response body exceeds configured size limits."""
    pass


class UnsupportedContentTypeError(URLServiceError):
    """Raised when the response is not readable HTML or text."""
    pass


class EmptyContentError(URLServiceError):
    """Raised when the fetched page contains no extractable or readable text."""
    pass


# HTMLTextExtractor is imported from app.utils.html_parser and re-exported here for compatibility.


# ============================================================================
# URL Ingestion Service Implementation
# ============================================================================


class URLService:
    """Service responsible for validating URLs, preventing SSRF, fetching content, and indexing."""

    MAX_RESPONSE_SIZE_BYTES: int = 5 * 1024 * 1024  # 5 MB limit
    REQUEST_TIMEOUT_SECONDS: float = 15.0
    MAX_REDIRECTS: int = 5

    def __init__(
        self,
        github_service: Optional[GitHubMCPService] = None,
        chunking_service: Optional[TextChunkingService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        pinecone_service: Optional[PineconeService] = None,
        source_service: Optional[SourceService] = None,
    ):
        self.github_service = github_service or get_github_service()
        self.chunking_service = chunking_service or get_chunking_service()
        self.embedding_service = embedding_service or get_embedding_service()
        self.pinecone_service = pinecone_service or get_pinecone_service()
        self.source_service = source_service or get_source_service()

    @staticmethod
    def is_safe_ip(ip_str: str) -> bool:
        """
        Verify that an IP address is not private, loopback, link-local, multicast,
        or reserved. Supports both IPv4 and IPv6.
        """
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            return False

        # If IPv6 is an IPv4-mapped address (::ffff:192.168.1.1), check the mapped IPv4
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped

        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False

        # Check carrier-grade NAT (100.64.0.0/10) and link-local cloud metadata (169.254.0.0/16)
        if isinstance(ip, ipaddress.IPv4Address):
            if ip in ipaddress.ip_network("100.64.0.0/10"):
                return False
            if ip in ipaddress.ip_network("169.254.0.0/16"):
                return False
            if ip in ipaddress.ip_network("192.0.0.0/24"):
                return False

        # Check unique-local IPv6 (fc00::/7) and link-local IPv6 (fe80::/10)
        if isinstance(ip, ipaddress.IPv6Address):
            if ip in ipaddress.ip_network("fc00::/7"):
                return False
            if ip in ipaddress.ip_network("fe80::/10"):
                return False

        return True

    @classmethod
    def validate_ssrf_url(cls, raw_url: str) -> str:
        """
        Validates URL scheme, resolves hostname DNS, and verifies that all destination IPs
        are public, preventing SSRF attacks.
        """
        if not raw_url or not raw_url.strip():
            raise InvalidURLError("URL must not be empty.")

        url_clean = raw_url.strip()
        parsed = urlparse(url_clean)

        scheme = (parsed.scheme or "").lower()
        if scheme not in ("http", "https"):
            raise InvalidURLError(f"Invalid URL scheme '{scheme}'. Only HTTP and HTTPS are permitted.")

        hostname = (parsed.hostname or "").strip().lower()
        if not hostname:
            raise InvalidURLError("URL does not contain a valid hostname.")

        # Block known dangerous local hostnames immediately
        if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "metadata.google.internal"):
            raise SSRFSecurityError(f"URL destination '{hostname}' is not permitted (localhost/private address).")

        # Resolve hostname DNS to check IP addresses
        port = parsed.port or (443 if scheme == "https" else 80)
        try:
            addr_info = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise URLFetchError(f"DNS resolution failed for hostname '{hostname}': {exc}") from exc

        if not addr_info:
            raise URLFetchError(f"Could not resolve any IP addresses for hostname '{hostname}'.")

        # Check all resolved IP addresses
        resolved_ips = set()
        for item in addr_info:
            sockaddr = item[4]
            ip_str = sockaddr[0]
            resolved_ips.add(ip_str)
            if not cls.is_safe_ip(ip_str):
                raise SSRFSecurityError(
                    f"URL destination '{hostname}' resolved to unsafe private or local address ({ip_str})."
                )

        return url_clean

    @staticmethod
    def _parse_github_repo(url: str) -> Optional[Tuple[str, str]]:
        """
        Detects if a URL points to a GitHub repository, returning (owner, repo) if matched.
        Examples:
            https://github.com/fastapi/fastapi -> ('fastapi', 'fastapi')
            https://github.com/octocat/Hello-World.git -> ('octocat', 'Hello-World')
        """
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower().split(":")[0]
        if host not in ("github.com", "www.github.com"):
            return None

        path_parts = [p for p in parsed.path.strip("/").split("/") if p]
        if len(path_parts) >= 2:
            owner = path_parts[0]
            repo = path_parts[1].removesuffix(".git")
            # If path points to tree/blob, ignore extra path elements for repo identification
            return owner, repo
        return None

    async def _fetch_webpage_content(self, initial_url: str) -> Tuple[str, str, str]:
        """
        Fetches webpage content with explicit redirect re-validation, timeout,
        and response size bounds. Returns (canonical_url, title, readable_text).
        """
        current_url = self.validate_ssrf_url(initial_url)
        redirect_count = 0

        headers = {
            "User-Agent": "AIInterviewPrepPlatform/1.0 (Public Documentation Bot; +http://localhost)",
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
        }

        async with httpx.AsyncClient(
            timeout=self.REQUEST_TIMEOUT_SECONDS,
            follow_redirects=False,
            verify=True,
        ) as client:
            while True:
                try:
                    response = await client.get(current_url, headers=headers)
                except httpx.TimeoutException as exc:
                    raise URLFetchError(f"Request to '{current_url}' timed out: {exc}") from exc
                except httpx.RequestError as exc:
                    raise URLFetchError(f"Network request to '{current_url}' failed: {exc}") from exc

                # Handle HTTP redirects explicitly with full SSRF re-validation per hop
                if response.status_code in (301, 302, 303, 307, 308):
                    redirect_count += 1
                    if redirect_count > self.MAX_REDIRECTS:
                        raise URLFetchError(f"Exceeded maximum permitted redirects ({self.MAX_REDIRECTS}).")

                    location = response.headers.get("Location")
                    if not location:
                        raise URLFetchError("Redirect response did not include a Location header.")

                    next_url = urljoin(current_url, location)
                    # Revalidate destination for SSRF
                    current_url = self.validate_ssrf_url(next_url)
                    continue

                # Handle HTTP error statuses
                if response.status_code in (401, 403):
                    raise ProtectedURLError(
                        f"Access to '{current_url}' is protected (HTTP {response.status_code}). "
                        "Bypassing paywalls or login restrictions is strictly prohibited."
                    )
                if response.status_code == 404:
                    raise URLFetchError(f"Webpage was not found at '{current_url}' (HTTP 404).")
                if response.status_code >= 400:
                    raise URLFetchError(f"Webpage returned error status HTTP {response.status_code}.")

                # Check Content-Type header
                content_type = response.headers.get("Content-Type", "").lower()
                if any(bad in content_type for bad in ["image/", "video/", "audio/", "application/zip", "application/octet-stream", "application/pdf"]):
                    raise UnsupportedContentTypeError(
                        f"Unsupported content type '{content_type}'. Supported formats are HTML and plain text web pages."
                    )

                # Check Content-Length if present
                content_length = response.headers.get("Content-Length")
                if content_length and content_length.isdigit():
                    if int(content_length) > self.MAX_RESPONSE_SIZE_BYTES:
                        max_mb = self.MAX_RESPONSE_SIZE_BYTES / (1024 * 1024)
                        raise ContentTooLargeError(f"Webpage content exceeds maximum allowed size limit of {max_mb:.1f} MB.")

                raw_bytes = response.content
                if len(raw_bytes) > self.MAX_RESPONSE_SIZE_BYTES:
                    max_mb = self.MAX_RESPONSE_SIZE_BYTES / (1024 * 1024)
                    raise ContentTooLargeError(f"Webpage content exceeds maximum allowed size limit of {max_mb:.1f} MB.")

                break

        # Decode HTML / text
        encoding = response.encoding or "utf-8"
        try:
            html_text = raw_bytes.decode(encoding, errors="replace")
        except Exception:
            html_text = raw_bytes.decode("utf-8", errors="replace")

        # Parse text and title using HTMLTextExtractor
        extractor = HTMLTextExtractor()
        extractor.feed(html_text)
        title = extractor.get_extracted_title()
        readable_text = extractor.get_extracted_text()

        # Fallback title if title tag was generic or absent
        if not title or title == "Webpage Documentation":
            parsed_curr = urlparse(current_url)
            title = f"{parsed_curr.netloc}{parsed_curr.path}" or current_url

        return current_url, title, readable_text

    KEY_REPO_FILENAMES = {
        "package.json",
        "pyproject.toml",
        "requirements.txt",
        "cargo.toml",
        "go.mod",
        "dockerfile",
        "docker-compose.yml",
        "main.py",
        "app.py",
        "index.ts",
        "index.js",
        "src/main.py",
        "src/index.ts",
        "app/main.py",
    }

    async def _extract_github_repo_content(self, owner: str, repo: str) -> Tuple[str, str, str]:
        """Fetch README and key root source files from a GitHub repository via GitHub MCP."""
        repo_id = f"{owner}/{repo}"
        title = f"{repo_id} Repository"
        canonical_url = f"https://github.com/{owner}/{repo}"

        logger.info("Ingesting GitHub repository '%s' via GitHub MCP Service.", repo_id)
        readme_resp = await self.github_service.get_readme(owner_or_url=repo_id)
        readable_text = readme_resp.content

        if not readable_text or not readable_text.strip():
            raise EmptyContentError(f"README for GitHub repository '{repo_id}' is empty.")

        additional_context = []
        try:
            files_res = await self.github_service.list_files(owner_or_url=repo_id, path="")
            for item in getattr(files_res, "items", []):
                if (
                    getattr(item, "type", "") == "file"
                    and getattr(item, "name", "").lower() in self.KEY_REPO_FILENAMES
                ):
                    try:
                        file_content_resp = await self.github_service.get_file_content(
                            owner_or_url=repo_id,
                            file_path=item.path,
                        )
                        if getattr(file_content_resp, "content", None):
                            safe_file_text = file_content_resp.content[:3000]
                            additional_context.append(
                                f"\n\n--- Source File: {item.path} ---\n{safe_file_text}"
                            )
                    except Exception as exc:
                        logger.debug(
                            "Could not fetch file '%s' for repository context: %s",
                            getattr(item, "path", ""),
                            exc,
                        )
        except Exception as exc:
            logger.debug("Could not list repository files for '%s': %s", repo_id, exc)

        if additional_context:
            readable_text += "".join(additional_context)

        return canonical_url, title, readable_text

    async def ingest_url(
        self,
        url: str,
        db=None,
    ) -> SourceUploadResponse:
        """
        Validates URL, checks SSRF, extracts content (reusing GitHub MCP if GitHub repo),
        chunks text, embeds via Ollama, stores in Pinecone, and persists source metadata.
        """
        clean_url = self.validate_ssrf_url(url)
        source_id = f"src_{uuid.uuid4().hex[:16]}"

        github_repo = self._parse_github_repo(clean_url)
        if github_repo is not None:
            owner, repo = github_repo
            source_type = SourceType.GITHUB
            canonical_url, title, readable_text = await self._extract_github_repo_content(owner, repo)
        else:
            source_type = SourceType.URL
            canonical_url, title, readable_text = await self._fetch_webpage_content(clean_url)

        char_count = len(readable_text)
        if char_count < 50:
            raise EmptyContentError("Extracted webpage content is too short or contains no readable text.")

        preview = " ".join(readable_text.split())[:500]

        # Register source in store / database
        self.source_service.register_source(
            source_id=source_id,
            source_type=source_type,
            title=title,
            url=canonical_url,
            status=SourceProcessingStatus.PROCESSING,
            character_count=char_count,
            db=db,
        )

        try:
            # Chunk the extracted text
            chunks = self.chunking_service.create_chunks(readable_text)
            chunk_count = len(chunks)

            # Generate embeddings via Ollama
            chunk_texts = [c.text for c in chunks]
            vectors = await self.embedding_service.embed_chunks(chunk_texts)

            # Store in Pinecone with source isolation metadata
            vector_count = await self.pinecone_service.upsert_document_chunks(
                document_id=source_id,
                chunks=chunks,
                vectors=vectors,
                filename=title,
            )

            # Update status to completed
            self.source_service.update_source(
                source_id=source_id,
                status=SourceProcessingStatus.COMPLETED,
                character_count=char_count,
                chunk_count=chunk_count,
                vector_count=vector_count,
                db=db,
            )

            logger.info(
                "Source '%s' (%s) ingested successfully: %d chars, %d chunks, %d vectors.",
                source_id,
                canonical_url,
                char_count,
                chunk_count,
                vector_count,
            )

            return SourceUploadResponse(
                success=True,
                source_id=source_id,
                source_type=source_type,
                status=SourceProcessingStatus.COMPLETED,
                url=canonical_url,
                title=title,
                character_count=char_count,
                chunk_count=chunk_count,
                vector_count=vector_count,
                preview=preview,
                message=f"Successfully extracted {char_count} characters and stored {vector_count} vector(s) in Pinecone.",
            )

        except Exception as exc:
            self.source_service.update_source(
                source_id=source_id,
                status=SourceProcessingStatus.FAILED,
                error_message=str(exc),
                db=db,
            )
            raise


def get_url_service(
    github_service: GitHubMCPService = Depends(get_github_service),
    chunking_service: TextChunkingService = Depends(get_chunking_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    pinecone_service: PineconeService = Depends(get_pinecone_service),
    source_service: SourceService = Depends(get_source_service),
) -> URLService:
    """FastAPI dependency yielding URLService with properly injected dependencies."""
    return URLService(
        github_service=github_service,
        chunking_service=chunking_service,
        embedding_service=embedding_service,
        pinecone_service=pinecone_service,
        source_service=source_service,
    )
