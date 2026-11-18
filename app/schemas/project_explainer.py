"""Pydantic request and response schemas for the GitHub Project Explainer feature."""

from enum import Enum
from typing import Dict, List, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

from app.schemas.github import RepoMetadataResponse, parse_repo_identifier


def parse_github_repo_url(raw_url: str) -> tuple[str, str, str]:
    """Strictly validate a GitHub repository HTTP/HTTPS URL and return (owner, repo, canonical_url)."""
    cleaned = (raw_url or "").strip()
    if not cleaned:
        raise ValueError("GitHub repository URL must not be empty.")

    parsed = urlparse(cleaned)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ValueError("GitHub repository URL must use http or https scheme.")

    if parsed.username or parsed.password:
        raise ValueError("GitHub repository URL must not contain embedded credentials.")

    host = (parsed.hostname or "").lower()
    if host not in ("github.com", "www.github.com"):
        raise ValueError("URL host must be github.com (e.g., https://github.com/owner/repo).")

    if parsed.query or parsed.fragment:
        raise ValueError("GitHub repository URL must not include query parameters or fragments.")

    path_parts = [p for p in parsed.path.split("/") if p]
    if len(path_parts) != 2:
        raise ValueError(
            "GitHub repository URL must point directly to a repository: https://github.com/owner/repo."
        )

    owner, repo = parse_repo_identifier(f"https://github.com/{path_parts[0]}/{path_parts[1]}")
    canonical_url = f"https://github.com/{owner}/{repo}"
    return owner, repo, canonical_url


class ProjectAnalysisStatus(str, Enum):
    """Lifecycle status for a Project Explainer analysis job."""

    PENDING = "pending"
    INGESTING = "ingesting"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProjectAnalyzeRequest(BaseModel):
    """Request payload to start or retrieve a GitHub project analysis."""

    repo_url: str = Field(
        ...,
        min_length=19,
        max_length=512,
        description="Full GitHub repository URL (e.g. https://github.com/fastapi/fastapi).",
        examples=["https://github.com/fastapi/fastapi"],
    )
    ref: Optional[str] = Field(
        default=None,
        max_length=120,
        description="Optional Git branch, tag, or commit SHA. Defaults to the repository's default branch.",
    )
    force_reanalyze: bool = Field(
        default=False,
        description="If True, purges prior repository vectors and re-ingests/re-analyzes from scratch.",
    )
    async_mode: bool = Field(
        default=True,
        description="If True, queues analysis asynchronously (202 Accepted). If False, runs inline and returns completed result.",
    )

    @field_validator("repo_url")
    @classmethod
    def validate_repo_url(cls, v: str) -> str:
        _, _, canonical = parse_github_repo_url(v)
        return canonical

    @field_validator("ref")
    @classmethod
    def validate_ref(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        cleaned = v.strip()
        if not cleaned:
            return None
        if any(ch in cleaned for ch in ("..", " ", "\n", "\r", "\t")):
            raise ValueError("Invalid Git ref format.")
        return cleaned


class ProjectExplainQueryRequest(BaseModel):
    """Request payload for asking a repository-specific explanation question via RAG."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Specific question about the analyzed repository's code, architecture, or behavior.",
        examples=["How does authentication middleware validate incoming requests?"],
    )
    file_path: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional relative file path to narrow RAG retrieval within this repository.",
    )
    top_k: int = Field(
        default=6,
        ge=1,
        le=15,
        description="Maximum number of repository code chunks to retrieve from Pinecone.",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        cleaned = (v or "").strip()
        if len(cleaned) < 3:
            raise ValueError("Question must be at least 3 characters long.")
        return cleaned

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        cleaned = v.strip().lstrip("/")
        if not cleaned:
            return None
        parts = cleaned.split("/")
        if ".." in parts or "." in parts:
            raise ValueError("Invalid file path: directory traversal sequences are not allowed.")
        return cleaned


# ============================================================================
# Structured 9-Section Analysis Output Schemas
# ============================================================================


class ProjectOverviewSection(BaseModel):
    """Section 1: High-level overview grounded in README and repository metadata."""

    what_it_does: str
    main_purpose: str
    target_users: List[str] = Field(default_factory=list)
    major_features: List[str] = Field(default_factory=list)
    evidence_paths: List[str] = Field(default_factory=list)


class TechnologyEvidenceItem(BaseModel):
    """Detected technology paired with concrete repository evidence."""

    name: str
    category: Literal[
        "language",
        "frontend_framework",
        "backend_framework",
        "database",
        "library",
        "infrastructure",
        "external_service",
    ]
    evidence_files: List[str] = Field(default_factory=list)
    evidence_notes: str = ""
    confidence: Literal["verified", "inferred"] = "verified"


class TechnologyStackSection(BaseModel):
    """Section 2: Categorized technology stack with file-level evidence."""

    programming_languages: List[TechnologyEvidenceItem] = Field(default_factory=list)
    frontend_frameworks: List[TechnologyEvidenceItem] = Field(default_factory=list)
    backend_frameworks: List[TechnologyEvidenceItem] = Field(default_factory=list)
    databases: List[TechnologyEvidenceItem] = Field(default_factory=list)
    libraries: List[TechnologyEvidenceItem] = Field(default_factory=list)
    infrastructure_and_deployment: List[TechnologyEvidenceItem] = Field(default_factory=list)
    external_services: List[TechnologyEvidenceItem] = Field(default_factory=list)


class ArchitectureComponentItem(BaseModel):
    """Important architectural component or service with its responsibility and evidence."""

    name: str
    responsibility: str
    evidence_paths: List[str] = Field(default_factory=list)


class ProjectArchitectureSection(BaseModel):
    """Section 3: System architecture across frontend, backend, data, and external integrations."""

    overview: str
    frontend_architecture: str
    backend_architecture: str
    database_architecture: str
    api_communication: str
    external_integrations: str
    important_components: List[ArchitectureComponentItem] = Field(default_factory=list)
    unavailable_details: List[str] = Field(default_factory=list)


class FolderExplanationItem(BaseModel):
    """Description of an important directory in the repository."""

    path: str
    purpose: str
    key_files: List[str] = Field(default_factory=list)


class FileRoleItem(BaseModel):
    """Description of an important source or configuration file."""

    path: str
    language: str
    responsibility: str


class FolderStructureSection(BaseModel):
    """Section 4: Repository directory layout and module responsibilities."""

    summary: str
    important_folders: List[FolderExplanationItem] = Field(default_factory=list)
    important_files: List[FileRoleItem] = Field(default_factory=list)


class ApplicationWorkflowSection(BaseModel):
    """Section 5: End-to-end execution, API, database, and business flows."""

    end_to_end_summary: str
    frontend_to_backend_communication: str
    api_calls: List[str] = Field(default_factory=list)
    database_interactions: List[str] = Field(default_factory=list)
    business_flows: List[str] = Field(default_factory=list)
    evidence_paths: List[str] = Field(default_factory=list)


class ImportantCodeItem(BaseModel):
    """Section 6 item: Walkthrough of an important file, module, or function."""

    file_path: str
    symbol_or_section: str
    purpose: str
    module_interactions: str
    evidence_notes: str = ""


class ImprovementSuggestionItem(BaseModel):
    """Section 7 item: Actionable engineering improvement grounded in inspected code."""

    category: Literal[
        "performance",
        "security",
        "code_quality",
        "scalability",
        "architecture",
        "best_practices",
    ]
    title: str
    description: str
    priority: Literal["High", "Medium", "Low"] = "Medium"
    evidence_paths: List[str] = Field(default_factory=list)
    classification: Literal["recommendation"] = "recommendation"


class ImprovementSuggestionsSection(BaseModel):
    """Section 7: Categorized engineering improvement suggestions."""

    performance: List[ImprovementSuggestionItem] = Field(default_factory=list)
    security: List[ImprovementSuggestionItem] = Field(default_factory=list)
    code_quality: List[ImprovementSuggestionItem] = Field(default_factory=list)
    scalability: List[ImprovementSuggestionItem] = Field(default_factory=list)
    architecture: List[ImprovementSuggestionItem] = Field(default_factory=list)
    missing_best_practices: List[ImprovementSuggestionItem] = Field(default_factory=list)


class FeatureIdeaItem(BaseModel):
    """Section 8 item: Proposed feature tailored to this repository."""

    title: str
    description: str
    value_proposition: str
    complexity: Literal["Low", "Medium", "High"] = "Medium"
    impacted_modules: List[str] = Field(default_factory=list)


class RepoInterviewQuestionItem(BaseModel):
    """Section 9 item: Repository-grounded interview question."""

    question: str
    level: Literal["beginner", "intermediate", "advanced", "architecture_and_scenario"]
    expected_answer_points: List[str] = Field(default_factory=list)
    follow_up_question: str
    evidence_paths: List[str] = Field(default_factory=list)


class InterviewPreparationSection(BaseModel):
    """Section 9: Categorized interview questions grounded in repository evidence."""

    beginner: List[RepoInterviewQuestionItem] = Field(default_factory=list)
    intermediate: List[RepoInterviewQuestionItem] = Field(default_factory=list)
    advanced: List[RepoInterviewQuestionItem] = Field(default_factory=list)
    architecture_and_scenario: List[RepoInterviewQuestionItem] = Field(default_factory=list)


class EvidenceReport(BaseModel):
    """Traceability summary separating verified repository facts from assumptions/gaps."""

    analyzed_files: List[str] = Field(default_factory=list)
    readme_present: bool = False
    unavailable_information: List[str] = Field(default_factory=list)
    facts_vs_assumptions_note: str = (
        "All verified technologies and behaviors cite inspected repository files; "
        "items marked 'inferred' or 'recommendation' represent engineering assessments."
    )


class ProjectAnalysisReport(BaseModel):
    """Complete 9-section structured project explanation."""

    project_overview: ProjectOverviewSection
    technology_stack: TechnologyStackSection
    project_architecture: ProjectArchitectureSection
    folder_structure: FolderStructureSection
    application_workflow: ApplicationWorkflowSection
    important_code_explanation: List[ImportantCodeItem] = Field(default_factory=list)
    improvement_suggestions: ImprovementSuggestionsSection
    new_feature_ideas: List[FeatureIdeaItem] = Field(default_factory=list)
    interview_preparation: InterviewPreparationSection
    evidence_report: EvidenceReport


# ============================================================================
# API Endpoint Response Schemas
# ============================================================================


class ProjectIngestionStats(BaseModel):
    """Metrics from repository traversal, filtering, chunking, and Pinecone indexing."""

    files_discovered: int = 0
    files_analyzed: int = 0
    files_skipped: int = 0
    chunks_indexed: int = 0
    vectors_stored: int = 0
    total_characters: int = 0
    truncated_due_to_limits: bool = False
    skipped_reasons: Dict[str, int] = Field(default_factory=dict)
    analyzed_file_paths: List[str] = Field(default_factory=list)


class ProjectAnalysisStartResponse(BaseModel):
    """Response returned when initiating or reusing a project analysis."""

    analysis_id: str
    repository: str
    repo_url: str
    source_id: str
    status: ProjectAnalysisStatus
    progress_percent: int = Field(0, ge=0, le=100)
    current_step: str
    reused_existing: bool = False
    message: str


class ProjectAnalysisStatusResponse(BaseModel):
    """Progress and status response for tracking an ongoing or finished analysis."""

    analysis_id: str
    repository: str
    repo_url: str
    ref: Optional[str] = None
    source_id: str
    status: ProjectAnalysisStatus
    progress_percent: int = Field(0, ge=0, le=100)
    current_step: str
    ingestion_stats: Optional[ProjectIngestionStats] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


class ProjectAnalysisDetailResponse(BaseModel):
    """Full analysis response including metadata, ingestion metrics, and 9-section report."""

    analysis_id: str
    repository: str
    repo_url: str
    ref: Optional[str] = None
    source_id: str
    status: ProjectAnalysisStatus
    progress_percent: int = Field(0, ge=0, le=100)
    current_step: str
    metadata: Optional[RepoMetadataResponse] = None
    ingestion_stats: Optional[ProjectIngestionStats] = None
    analysis: Optional[ProjectAnalysisReport] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


class ProjectCodeCitation(BaseModel):
    """Code chunk citation returned by repository RAG explanation queries."""

    file_path: str
    language: str
    chunk_index: int
    score: float
    excerpt: str


class ProjectExplainQueryResponse(BaseModel):
    """Response to a repository-specific explanation query grounded via Pinecone RAG."""

    analysis_id: str
    repository: str
    question: str
    answer: str
    context_grounded: bool
    citations: List[ProjectCodeCitation] = Field(default_factory=list)
    unavailable_details: List[str] = Field(default_factory=list)


class ProjectSuggestionsResponse(BaseModel):
    """Focused response returning improvement suggestions and new feature ideas."""

    analysis_id: str
    repository: str
    improvement_suggestions: ImprovementSuggestionsSection
    new_feature_ideas: List[FeatureIdeaItem] = Field(default_factory=list)


class ProjectInterviewQuestionsResponse(BaseModel):
    """Focused response returning categorized repository-grounded interview questions."""

    analysis_id: str
    repository: str
    total_questions: int
    interview_preparation: InterviewPreparationSection

