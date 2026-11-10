"""Low-level GitHub MCP protocol constants, whitelist, and payload decoders."""

import base64
import json
import os
from typing import Any, Dict

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

README_EXTENSIONS = {
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
}

FALLBACK_README_CANDIDATES = [
    "README.md",
    "README",
    "readme.md",
    "README.rst",
    "README.txt",
    "README.markdown",
    "docs/README.md",
    ".github/README.md",
]


def is_readme_filename(filename: str) -> bool:
    """Return True if a filename matches a standard README stem and supported extension."""
    stem, ext = os.path.splitext(filename.lower())
    return stem == "readme" and ext in README_EXTENSIONS


def rank_readme_filename(filename: str) -> int:
    """Rank README filenames by preference order (README.md highest)."""
    nl = filename.lower()
    if nl == "readme.md":
        return 0
    if nl == "readme":
        return 1
    if any(nl.endswith(ext) for ext in (".markdown", ".mdown", ".mkdn")):
        return 2
    if nl.endswith(".rst"):
        return 3
    if nl.endswith(".txt") or nl.endswith(".text"):
        return 4
    return 5


def decode_base64_or_text(content: Any, encoding: str) -> str:
    """Safely decode file content, handling both pre-decoded MCP text and raw base64."""
    if not content or not isinstance(content, str):
        return ""
    if encoding == "base64":
        try:
            decoded_bytes = base64.b64decode(content.strip(), validate=True)
            return decoded_bytes.decode("utf-8", errors="replace")
        except Exception:
            return content
    return content


def extract_mcp_error_text(result: Any, tool_name: str) -> str:
    """Extract readable error text from an MCP CallToolResult flagged as error."""
    err_parts = []
    if getattr(result, "content", None):
        for item in result.content:
            if hasattr(item, "text") and item.text:
                err_parts.append(item.text)
            elif hasattr(item, "resource"):
                res_obj = getattr(item, "resource", None)
                if res_obj and hasattr(res_obj, "text") and res_obj.text:
                    err_parts.append(res_obj.text)
    return " ".join(err_parts).strip() or f"Tool '{tool_name}' returned error"


def parse_mcp_tool_result_content(result: Any) -> Any:
    """Parse EmbeddedResource or TextContent blocks from an MCP CallToolResult."""
    if not getattr(result, "content", None) or len(result.content) == 0:
        return {}

    for item in result.content:
        if hasattr(item, "resource") and item.resource is not None:
            embedded = item.resource
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

    first_content = result.content[0]
    text_data = getattr(first_content, "text", "")
    if not text_data:
        return {}

    try:
        return json.loads(text_data)
    except json.JSONDecodeError:
        return text_data

