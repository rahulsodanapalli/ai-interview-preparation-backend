"""Repository Ingestion & Evidence Extraction Service for Project Explainer.

Safely traverses a GitHub repository via GitHubMCPService, excludes secrets,
binaries, lockfiles, and build directories, enforces file/size limits, extracts
deterministic technology evidence from manifest/config files, and indexes
file-path-tagged chunks into Pinecone via TextChunkingService, EmbeddingService,
and PineconeService.
"""

from collections import Counter, deque
from contextlib import nullcontext
from dataclasses import dataclass, field
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.schemas.github import RepoMetadataResponse
from app.schemas.project_explainer import (
    ProjectIngestionStats,
    TechnologyEvidenceItem,
)
from app.services.chunking_service import TextChunk, TextChunkingService
from app.services.github_mcp_service import (
    GitHubFileNotFoundError,
    GitHubFileTooLargeError,
    GitHubMCPService,
)

logger = logging.getLogger(__name__)


class EmptyRepositoryError(Exception):
    """Raised when a repository contains no readable README or supported source/config files."""


IGNORED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".github",
    "node_modules",
    "bower_components",
    "jspm_packages",
    "venv",
    ".venv",
    "env",
    "virtualenv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "dist",
    "build",
    "out",
    "target",
    "bin",
    "obj",
    "vendor",
    "third_party",
    ".next",
    ".nuxt",
    ".svelte-kit",
    ".angular",
    ".cache",
    "coverage",
    ".nyc_output",
    "htmlcov",
    ".terraform",
    ".idea",
    ".vscode",
    "logs",
    "tmp",
    "temp",
}

LOCK_OR_GENERATED_FILENAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "pipfile.lock",
    "cargo.lock",
    "composer.lock",
    "gemfile.lock",
    "go.sum",
    "bun.lockb",
}

SENSITIVE_EXACT_FILENAMES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.staging",
    ".env.test",
    ".env.example",
    ".env.sample",
    ".env.template",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "credentials.json",
    "secrets.json",
    "secrets.yml",
    "secrets.yaml",
    ".npmrc",
    ".pypirc",
    ".netrc",
    ".htpasswd",
    ".pgpass",
    "kubeconfig",
}

SENSITIVE_EXTENSIONS = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".jks",
    ".keystore",
    ".crt",
    ".cer",
    ".der",
    ".ppk",
    ".ovpn",
    ".kdbx",
    ".env",
}

BINARY_OR_UNSUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".bmp",
    ".tiff",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".wasm",
    ".pyc",
    ".pyo",
    ".class",
    ".jar",
    ".war",
    ".o",
    ".a",
    ".lib",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".otf",
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mov",
    ".webm",
    ".iso",
    ".dmg",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".map",
}

SUPPORTED_EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".swift": "swift",
    ".scala": "scala",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".ps1": "powershell",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".vue": "vue",
    ".svelte": "svelte",
    ".astro": "astro",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".proto": "protobuf",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "config",
    ".xml": "xml",
    ".gradle": "gradle",
    ".properties": "properties",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "restructuredtext",
    ".txt": "text",
}

SUPPORTED_SPECIAL_FILENAMES: Dict[str, str] = {
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "procfile": "procfile",
    "jenkinsfile": "groovy",
    "gemfile": "ruby",
    "pipfile": "toml",
    "rakefile": "ruby",
}

MANIFEST_FILENAMES = {
    "package.json",
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "pipfile",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "go.mod",
    "cargo.toml",
    "composer.json",
    "gemfile",
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "tsconfig.json",
    "vite.config.ts",
    "vite.config.js",
    "next.config.js",
    "next.config.mjs",
    "next.config.ts",
    "alembic.ini",
    "makefile",
}

_SECRET_REDACTION_PATTERNS = [
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{10,}|github_pat_[A-Za-z0-9_]{10,}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"), "[REDACTED_GOOGLE_KEY]"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"), "[REDACTED_PRIVATE_KEY]"),
]


@dataclass
class IngestedRepoFile:
    """Single fetched and sanitized repository file."""

    path: str
    language: str
    category: str
    content: str
    size_bytes: int
    is_truncated: bool = False


@dataclass
class RepoIngestionBundle:
    """Complete repository snapshot collected safely via GitHubMCPService."""

    owner: str
    repo: str
    full_name: str
    repo_url: str
    ref: Optional[str]
    metadata: RepoMetadataResponse
    readme_filename: Optional[str]
    readme_content: Optional[str]
    tree_paths: List[str]
    files: List[IngestedRepoFile]
    stats: ProjectIngestionStats
    detected_technologies: Dict[str, List[TechnologyEvidenceItem]] = field(default_factory=dict)
    unavailable_information: List[str] = field(default_factory=list)


def sanitize_repo_text(text: str) -> str:
    """Redact accidental secret patterns from repository text before chunking or LLM prompts."""
    cleaned = text or ""
    for pattern, replacement in _SECRET_REDACTION_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def is_sensitive_filename(path_or_name: str) -> bool:
    """Return True if a path or filename represents a secret, key, credential, or .env file."""
    lower_path = (path_or_name or "").strip().lower()
    base = os.path.basename(lower_path)
    if not base:
        return False
    if base in SENSITIVE_EXACT_FILENAMES or base.startswith(".env.") or base.endswith(".env"):
        return True
    _, ext = os.path.splitext(base)
    if ext in SENSITIVE_EXTENSIONS:
        return True
    if "service_account" in base or "service-account" in base or "private_key" in base or "privatekey" in base:
        return True
    return False


def detect_language_and_support(path_or_name: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """Determine if file is supported for text ingestion, returning (supported, language, skip_reason)."""
    lower_path = (path_or_name or "").strip().lower()
    base = os.path.basename(lower_path)
    if not base:
        return False, None, "unsupported_extension"

    if is_sensitive_filename(lower_path):
        return False, None, "secret_file"

    if base in LOCK_OR_GENERATED_FILENAMES or base.endswith(".min.js") or base.endswith(".min.css") or base.endswith(".bundle.js"):
        return False, None, "generated_or_lockfile"

    if base in SUPPORTED_SPECIAL_FILENAMES:
        return True, SUPPORTED_SPECIAL_FILENAMES[base], None

    _, ext = os.path.splitext(base)
    if ext in BINARY_OR_UNSUPPORTED_EXTENSIONS:
        return False, None, "unsupported_extension"

    if ext in SUPPORTED_EXTENSION_TO_LANGUAGE:
        return True, SUPPORTED_EXTENSION_TO_LANGUAGE[ext], None

    return False, None, "unsupported_extension"


def classify_file_category(file_path: str) -> str:
    """Classify file into a semantic category for prioritization and metadata."""
    lower = file_path.lower()
    base = os.path.basename(lower)
    if base.startswith("readme"):
        return "readme"
    if base in MANIFEST_FILENAMES or base.endswith((".yml", ".yaml", ".toml", ".ini", ".json", ".xml", ".gradle")):
        return "config"
    if any(tok in lower for tok in ("main.", "app.", "index.", "server.", "manage.py", "wsgi.", "asgi.")):
        return "entrypoint"
    if any(tok in lower for tok in ("model", "schema", "migration", "alembic", "prisma", "database", "/db/")):
        return "database"
    if any(tok in lower for tok in ("auth", "security", "oauth", "jwt", "session")):
        return "auth"
    if any(tok in lower for tok in ("route", "controller", "endpoint", "api/", "service", "repository", "middleware")):
        return "backend"
    if any(tok in lower for tok in ("component", "page", "context", "hook", "store", "src/", "frontend/")):
        return "frontend"
    return "source"


def score_file_priority(file_path: str) -> int:
    """Rank repository files so manifests, entry points, and architecture files are fetched first."""
    lower = file_path.lower()
    base = os.path.basename(lower)
    depth = lower.count("/")

    if base in MANIFEST_FILENAMES:
        return 100 - depth * 2
    if "test" in lower or "spec" in lower or "__tests__" in lower:
        return 10 - depth

    score = 30 - depth * 3
    high_signal_tokens = (
        "main.",
        "app.",
        "index.",
        "server.",
        "router",
        "routes",
        "controller",
        "service",
        "model",
        "schema",
        "repository",
        "auth",
        "config",
        "db",
        "database",
        "api",
        "context",
        "pages",
        "components",
    )
    for token in high_signal_tokens:
        if token in lower:
            score += 20
    return score


def _add_tech(
    bucket: Dict[str, Dict[str, TechnologyEvidenceItem]],
    *,
    group: str,
    name: str,
    category: str,
    evidence_file: str,
    note: str,
) -> None:
    group_map = bucket.setdefault(group, {})
    key = name.lower()
    if key in group_map:
        existing = group_map[key]
        if evidence_file not in existing.evidence_files:
            existing.evidence_files.append(evidence_file)
        return
    group_map[key] = TechnologyEvidenceItem(
        name=name,
        category=category,  # type: ignore[arg-type]
        evidence_files=[evidence_file],
        evidence_notes=note,
        confidence="verified",
    )


def detect_technology_evidence(
    files: List[IngestedRepoFile],
    metadata: RepoMetadataResponse,
) -> Dict[str, List[TechnologyEvidenceItem]]:
    """Deterministically detect languages, frameworks, DBs, libraries, and infra from inspected files."""
    bucket: Dict[str, Dict[str, TechnologyEvidenceItem]] = {
        "programming_languages": {},
        "frontend_frameworks": {},
        "backend_frameworks": {},
        "databases": {},
        "libraries": {},
        "infrastructure_and_deployment": {},
        "external_services": {},
    }

    lang_labels = {
        "python": "Python",
        "typescript": "TypeScript",
        "javascript": "JavaScript",
        "go": "Go",
        "rust": "Rust",
        "java": "Java",
        "kotlin": "Kotlin",
        "ruby": "Ruby",
        "php": "PHP",
        "csharp": "C#",
        "cpp": "C++",
        "c": "C",
        "swift": "Swift",
        "sql": "SQL",
    }

    for f in files:
        if f.language in lang_labels:
            _add_tech(
                bucket,
                group="programming_languages",
                name=lang_labels[f.language],
                category="language",
                evidence_file=f.path,
                note=f"Source file detected ({f.language})",
            )

        base = os.path.basename(f.path).lower()
        content_lower = f.content.lower()

        if base == "tsconfig.json":
            _add_tech(
                bucket,
                group="programming_languages",
                name="TypeScript",
                category="language",
                evidence_file=f.path,
                note="Configured via tsconfig.json",
            )

        if base == "dockerfile" or base.startswith("docker-compose"):
            _add_tech(
                bucket,
                group="infrastructure_and_deployment",
                name="Docker Compose" if base.startswith("docker-compose") else "Docker",
                category="infrastructure",
                evidence_file=f.path,
                note=f"Container configuration defined in {f.path}",
            )
            if "postgres" in content_lower:
                _add_tech(bucket, group="databases", name="PostgreSQL", category="database", evidence_file=f.path, note="Referenced in container configuration")
            if "redis" in content_lower:
                _add_tech(bucket, group="databases", name="Redis", category="database", evidence_file=f.path, note="Referenced in container configuration")
            if "mongo" in content_lower:
                _add_tech(bucket, group="databases", name="MongoDB", category="database", evidence_file=f.path, note="Referenced in container configuration")

        if base == "package.json":
            try:
                pkg = json.loads(f.content)
            except Exception:
                pkg = {}
            deps: Dict[str, Any] = {}
            if isinstance(pkg, dict):
                for section in ("dependencies", "devDependencies", "peerDependencies"):
                    sec_val = pkg.get(section)
                    if isinstance(sec_val, dict):
                        deps.update(sec_val)

            pkg_rules = [
                ("react", "frontend_frameworks", "React", "frontend_framework"),
                ("next", "frontend_frameworks", "Next.js", "frontend_framework"),
                ("vue", "frontend_frameworks", "Vue.js", "frontend_framework"),
                ("svelte", "frontend_frameworks", "Svelte", "frontend_framework"),
                ("@angular/core", "frontend_frameworks", "Angular", "frontend_framework"),
                ("express", "backend_frameworks", "Express.js", "backend_framework"),
                ("@nestjs/core", "backend_frameworks", "NestJS", "backend_framework"),
                ("fastify", "backend_frameworks", "Fastify", "backend_framework"),
                ("typescript", "programming_languages", "TypeScript", "language"),
                ("vite", "infrastructure_and_deployment", "Vite", "infrastructure"),
                ("tailwindcss", "libraries", "Tailwind CSS", "library"),
                ("axios", "libraries", "Axios", "library"),
                ("prisma", "databases", "Prisma ORM", "database"),
                ("@prisma/client", "databases", "Prisma ORM", "database"),
                ("mongoose", "databases", "MongoDB (Mongoose)", "database"),
                ("pg", "databases", "PostgreSQL (pg)", "database"),
                ("redis", "databases", "Redis", "database"),
                ("stripe", "external_services", "Stripe", "external_service"),
            ]
            for dep_key, grp, label, cat in pkg_rules:
                if dep_key in deps:
                    _add_tech(
                        bucket,
                        group=grp,
                        name=label,
                        category=cat,
                        evidence_file=f.path,
                        note=f"Listed in {f.path} ({dep_key}: {deps[dep_key]})",
                    )

        if base in ("requirements.txt", "requirements-dev.txt", "pyproject.toml", "pipfile"):
            py_rules = [
                ("fastapi", "backend_frameworks", "FastAPI", "backend_framework"),
                ("django", "backend_frameworks", "Django", "backend_framework"),
                ("flask", "backend_frameworks", "Flask", "backend_framework"),
                ("uvicorn", "infrastructure_and_deployment", "Uvicorn", "infrastructure"),
                ("sqlalchemy", "libraries", "SQLAlchemy", "library"),
                ("alembic", "databases", "Alembic", "database"),
                ("psycopg", "databases", "PostgreSQL (psycopg)", "database"),
                ("asyncpg", "databases", "PostgreSQL (asyncpg)", "database"),
                ("pymongo", "databases", "MongoDB (PyMongo)", "database"),
                ("redis", "databases", "Redis", "database"),
                ("pinecone", "databases", "Pinecone", "database"),
                ("pydantic", "libraries", "Pydantic", "library"),
                ("httpx", "libraries", "HTTPX", "library"),
                ("pytest", "libraries", "pytest", "library"),
                ("celery", "libraries", "Celery", "library"),
                ("openai", "external_services", "OpenAI API", "external_service"),
                ("anthropic", "external_services", "Anthropic API", "external_service"),
                ("stripe", "external_services", "Stripe", "external_service"),
            ]
            for dep_token, grp, label, cat in py_rules:
                if re.search(rf"(?<![a-z0-9_-]){re.escape(dep_token)}(?![a-z0-9_-])", content_lower):
                    _add_tech(
                        bucket,
                        group=grp,
                        name=label,
                        category=cat,
                        evidence_file=f.path,
                        note=f"Declared in {f.path}",
                    )

        if base == "go.mod":
            _add_tech(bucket, group="programming_languages", name="Go", category="language", evidence_file=f.path, note="Declared in go.mod")
            if "gin-gonic/gin" in content_lower:
                _add_tech(bucket, group="backend_frameworks", name="Gin", category="backend_framework", evidence_file=f.path, note="Declared in go.mod")
            if "labstack/echo" in content_lower:
                _add_tech(bucket, group="backend_frameworks", name="Echo", category="backend_framework", evidence_file=f.path, note="Declared in go.mod")
            if "gofiber/fiber" in content_lower:
                _add_tech(bucket, group="backend_frameworks", name="Fiber", category="backend_framework", evidence_file=f.path, note="Declared in go.mod")
            if "gorm.io" in content_lower:
                _add_tech(bucket, group="libraries", name="GORM", category="library", evidence_file=f.path, note="Declared in go.mod")

        if base == "cargo.toml":
            _add_tech(bucket, group="programming_languages", name="Rust", category="language", evidence_file=f.path, note="Declared in Cargo.toml")
            if "axum" in content_lower:
                _add_tech(bucket, group="backend_frameworks", name="Axum", category="backend_framework", evidence_file=f.path, note="Declared in Cargo.toml")
            if "actix-web" in content_lower:
                _add_tech(bucket, group="backend_frameworks", name="Actix Web", category="backend_framework", evidence_file=f.path, note="Declared in Cargo.toml")
            if "tokio" in content_lower:
                _add_tech(bucket, group="libraries", name="Tokio", category="library", evidence_file=f.path, note="Declared in Cargo.toml")

        if base in ("pom.xml", "build.gradle", "build.gradle.kts"):
            if "spring-boot" in content_lower:
                _add_tech(bucket, group="backend_frameworks", name="Spring Boot", category="backend_framework", evidence_file=f.path, note=f"Declared in {f.path}")
            if "postgresql" in content_lower:
                _add_tech(bucket, group="databases", name="PostgreSQL", category="database", evidence_file=f.path, note=f"Declared in {f.path}")

    if not bucket["programming_languages"] and metadata.language:
        _add_tech(
            bucket,
            group="programming_languages",
            name=metadata.language,
            category="language",
            evidence_file="github_metadata",
            note="Primary language reported by GitHub repository metadata",
        )

    return {grp: list(items.values()) for grp, items in bucket.items()}


class RepoIngestionService:
    """Safely collects repository structure, README, configs, and source files via GitHubMCPService."""

    def __init__(
        self,
        github_service: GitHubMCPService,
        chunking_service: TextChunkingService,
        *,
        max_files: Optional[int] = None,
        max_tree_depth: Optional[int] = None,
        max_dir_listings: Optional[int] = None,
        max_file_bytes: Optional[int] = None,
        max_total_chars: Optional[int] = None,
        max_file_chars: Optional[int] = None,
    ) -> None:
        self.github_service = github_service
        self.chunking_service = chunking_service
        self.max_files = max_files if max_files is not None else settings.PROJECT_EXPLAINER_MAX_FILES
        self.max_tree_depth = max_tree_depth if max_tree_depth is not None else settings.PROJECT_EXPLAINER_MAX_TREE_DEPTH
        self.max_dir_listings = max_dir_listings if max_dir_listings is not None else settings.PROJECT_EXPLAINER_MAX_DIR_LISTINGS
        self.max_file_bytes = max_file_bytes if max_file_bytes is not None else settings.PROJECT_EXPLAINER_MAX_FILE_BYTES
        self.max_total_chars = max_total_chars if max_total_chars is not None else settings.PROJECT_EXPLAINER_MAX_TOTAL_CHARS
        self.max_file_chars = max_file_chars if max_file_chars is not None else settings.PROJECT_EXPLAINER_MAX_FILE_CHARS

    async def collect_repository(
        self,
        *,
        owner: str,
        repo: str,
        repo_url: str,
        ref: Optional[str] = None,
        metadata: Optional[RepoMetadataResponse] = None,
    ) -> RepoIngestionBundle:
        """Traverse repository, select high-signal files, exclude secrets/binaries, and extract evidence."""
        scope = (
            self.github_service.session_scope()
            if hasattr(self.github_service, "session_scope")
            else nullcontext()
        )
        async with scope:
            return await self._collect_repository_inner(
                owner=owner,
                repo=repo,
                repo_url=repo_url,
                ref=ref,
                metadata=metadata,
            )

    async def _collect_repository_inner(
        self,
        *,
        owner: str,
        repo: str,
        repo_url: str,
        ref: Optional[str] = None,
        metadata: Optional[RepoMetadataResponse] = None,
    ) -> RepoIngestionBundle:
        """Inner repository collection executed within a single reusable MCP session when supported."""
        full_name = f"{owner}/{repo}"
        repo_meta = metadata or await self.github_service.get_repository_metadata(full_name)
        unavailable_info: List[str] = []

        # 1. Fetch README (optional — missing README is handled gracefully)
        readme_filename: Optional[str] = None
        readme_content: Optional[str] = None
        try:
            readme_resp = await self.github_service.get_readme(owner_or_url=full_name, ref=ref)
            if readme_resp.content and readme_resp.content.strip():
                readme_filename = readme_resp.filename
                readme_content = sanitize_repo_text(readme_resp.content.strip())[: self.max_file_chars]
            else:
                unavailable_info.append("Repository README exists but is empty.")
        except GitHubFileNotFoundError:
            unavailable_info.append("No README file was found in the repository.")
        except Exception as exc:
            logger.warning("README retrieval failed for %s: %s", full_name, type(exc).__name__)
            unavailable_info.append("Repository README could not be retrieved.")

        # 2. Traverse directory tree via BFS up to configured depth & listing budget
        queue: deque[Tuple[str, int]] = deque([("", 0)])
        visited_dirs: set[str] = set()
        discovered_files: List[Tuple[str, Optional[int]]] = []
        tree_paths: List[str] = []
        skipped_reasons: Counter[str] = Counter()
        listings_count = 0
        truncated_due_to_limits = False

        while queue and listings_count < self.max_dir_listings:
            dir_path, depth = queue.popleft()
            if dir_path in visited_dirs:
                continue
            visited_dirs.add(dir_path)
            listings_count += 1

            try:
                listing = await self.github_service.list_files(owner_or_url=full_name, path=dir_path, ref=ref)
            except GitHubFileNotFoundError:
                continue

            for item in listing.items:
                item_path = (item.path or item.name or "").strip().lstrip("/")
                if not item_path:
                    continue
                item_name = os.path.basename(item_path).lower()

                if item.type == "dir":
                    if item_name in IGNORED_DIRECTORIES:
                        skipped_reasons["ignored_directory"] += 1
                        continue
                    tree_paths.append(f"{item_path}/")
                    if depth + 1 < self.max_tree_depth:
                        queue.append((item_path, depth + 1))
                    else:
                        truncated_due_to_limits = True
                elif item.type == "file":
                    tree_paths.append(item_path)
                    if readme_filename and item_path.lower() == readme_filename.lower():
                        continue
                    discovered_files.append((item_path, item.size))

        if queue:
            truncated_due_to_limits = True

        # 3. Filter and prioritize discovered files
        candidate_files: List[Tuple[int, str, str]] = []
        for f_path, f_size in discovered_files:
            supported, lang, skip_reason = detect_language_and_support(f_path)
            if not supported or not lang:
                skipped_reasons[skip_reason or "unsupported_extension"] += 1
                continue
            if f_size is not None and f_size > self.max_file_bytes:
                skipped_reasons["file_too_large"] += 1
                truncated_due_to_limits = True
                continue
            candidate_files.append((score_file_priority(f_path), f_path, lang))

        candidate_files.sort(key=lambda entry: (-entry[0], entry[1]))
        if len(candidate_files) > self.max_files:
            skipped_reasons["limit_reached"] += len(candidate_files) - self.max_files
            candidate_files = candidate_files[: self.max_files]
            truncated_due_to_limits = True

        # 4. Fetch prioritized file contents within total character budget
        ingested_files: List[IngestedRepoFile] = []
        total_chars = len(readme_content or "")

        for _score, f_path, lang in candidate_files:
            if total_chars >= self.max_total_chars:
                skipped_reasons["limit_reached"] += 1
                truncated_due_to_limits = True
                continue
            try:
                file_resp = await self.github_service.get_file_content(
                    owner_or_url=full_name,
                    file_path=f_path,
                    ref=ref,
                )
            except GitHubFileTooLargeError:
                skipped_reasons["file_too_large"] += 1
                truncated_due_to_limits = True
                continue
            except GitHubFileNotFoundError:
                skipped_reasons["fetch_error"] += 1
                continue
            except Exception as exc:
                logger.warning("Skipping file '%s' in %s due to %s", f_path, full_name, type(exc).__name__)
                skipped_reasons["fetch_error"] += 1
                continue

            if file_resp.size_bytes > self.max_file_bytes:
                skipped_reasons["file_too_large"] += 1
                truncated_due_to_limits = True
                continue

            clean_content = sanitize_repo_text(file_resp.content or "").strip()
            if not clean_content:
                skipped_reasons["empty_file"] += 1
                continue

            remaining_budget = max(0, self.max_total_chars - total_chars)
            cap = min(self.max_file_chars, remaining_budget)
            if cap < 50:
                skipped_reasons["limit_reached"] += 1
                truncated_due_to_limits = True
                continue

            was_truncated = len(clean_content) > cap
            if was_truncated:
                clean_content = clean_content[:cap]
                truncated_due_to_limits = True

            total_chars += len(clean_content)
            ingested_files.append(
                IngestedRepoFile(
                    path=f_path,
                    language=lang,
                    category=classify_file_category(f_path),
                    content=clean_content,
                    size_bytes=file_resp.size_bytes,
                    is_truncated=was_truncated,
                )
            )

        if not readme_content and not ingested_files:
            raise EmptyRepositoryError(
                f"Repository '{full_name}' contains no readable README or supported source/configuration files."
            )

        detected_tech = detect_technology_evidence(ingested_files, repo_meta)
        analyzed_paths = ([readme_filename] if readme_filename and readme_content else []) + [f.path for f in ingested_files]

        stats = ProjectIngestionStats(
            files_discovered=len(discovered_files) + (1 if readme_filename else 0),
            files_analyzed=len(analyzed_paths),
            files_skipped=sum(skipped_reasons.values()),
            chunks_indexed=0,
            vectors_stored=0,
            total_characters=total_chars,
            truncated_due_to_limits=truncated_due_to_limits,
            skipped_reasons=dict(skipped_reasons),
            analyzed_file_paths=analyzed_paths,
        )

        return RepoIngestionBundle(
            owner=owner,
            repo=repo,
            full_name=full_name,
            repo_url=repo_url,
            ref=ref,
            metadata=repo_meta,
            readme_filename=readme_filename,
            readme_content=readme_content,
            tree_paths=sorted(set(tree_paths))[:200],
            files=ingested_files,
            stats=stats,
            detected_technologies=detected_tech,
            unavailable_information=unavailable_info,
        )

    def build_repository_chunks(
        self,
        bundle: RepoIngestionBundle,
        *,
        analysis_id: str,
    ) -> Tuple[List[TextChunk], List[Dict[str, Any]]]:
        """Split README and repository files into path- and language-tagged chunks for Pinecone."""
        all_chunks: List[TextChunk] = []
        metadata_list: List[Dict[str, Any]] = []
        global_idx = 0
        repo_key = bundle.full_name.lower()

        def _append_file_chunks(file_path: str, language: str, category: str, text: str) -> None:
            nonlocal global_idx
            raw_chunks = self.chunking_service.create_chunks(text)
            for chunk in raw_chunks:
                prefixed_text = f"File: {file_path} ({language})\n{chunk.text}"
                all_chunks.append(
                    TextChunk(
                        index=global_idx,
                        text=prefixed_text,
                        character_count=len(prefixed_text),
                        token_estimate=self.chunking_service.estimate_tokens(prefixed_text),
                    )
                )
                metadata_list.append(
                    {
                        "file_path": file_path,
                        "language": language,
                        "repository": repo_key,
                        "analysis_id": analysis_id,
                        "source_type": "project_explainer",
                        "file_category": category,
                    }
                )
                global_idx += 1

        if bundle.readme_content and bundle.readme_filename:
            _append_file_chunks(bundle.readme_filename, "markdown", "readme", bundle.readme_content)

        for repo_file in bundle.files:
            _append_file_chunks(repo_file.path, repo_file.language, repo_file.category, repo_file.content)

        return all_chunks, metadata_list

