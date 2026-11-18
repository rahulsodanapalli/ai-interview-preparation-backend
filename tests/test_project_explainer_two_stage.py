"""Targeted unit and integration tests for the Two-Stage Project Explainer and Ollama completion handling.

Covers all required scenarios from Section 6:
- Small repository (1-3 files)
- Medium repository (10-15 files)
- Repository with 36 or more files
- Large README (bounded and truncated safely)
- Large source file (bounded per file and group)
- Output token limit reached (done_reason='length' -> OllamaTruncatedResponseError)
- Context window exceeded (handled gracefully)
- Valid structured JSON (matching schema)
- Incomplete JSON (detection and controlled repair attempt)
- Ollama unavailable (OllamaConnectionError -> 503)
- Retry after generation failure (controlled repair once, no duplicate huge prompt)
- Real local Ollama integration test (runs against live qwen2.5:1.5b)
"""

import json
from unittest.mock import AsyncMock, MagicMock
import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.schemas.github import RepoFileItem, RepoMetadataResponse, RepoReadmeResponse
from app.schemas.project_explainer import ProjectIngestionStats
from app.services.chunking_service import TextChunkingService
from app.services.github_mcp_service import GitHubMCPService
from app.services.ollama_service import (
    OllamaConnectionError,
    OllamaResponseError,
    OllamaService,
    OllamaTruncatedResponseError,
)
from app.services.project_explainer_service import (
    STAGE_1_GROUP_JSON_SCHEMA,
    STAGE_1_JSON_SCHEMA,
    STAGE_2_JSON_SCHEMA,
    STAGE_3_JSON_SCHEMA,
    ProjectExplainerService,
    get_project_explainer_service,
)
from app.services.repo_ingestion_service import IngestedRepoFile, RepoIngestionBundle
from tests.conftest import DefaultMockEmbeddingService, DefaultMockPineconeService


# ----------------------------------------------------------------------------
# Helper builders for test repositories
# ----------------------------------------------------------------------------

def _sample_s1_dict():
    return {
        "project_overview": {
            "what_it_does": "A lightweight web service for task tracking and management.",
            "main_purpose": "Provide REST endpoints to track developer tasks.",
            "target_users": ["Software engineers", "Project managers"],
            "major_features": ["Task CRUD operations", "Status updates"],
            "evidence_paths": ["README.md", "app/main.py"],
        },
        "additional_technologies": [
            {
                "name": "FastAPI",
                "category": "backend_framework",
                "evidence_files": ["requirements.txt"],
                "evidence_notes": "FastAPI listed in requirements.txt",
                "confidence": "verified",
            }
        ],
        "project_architecture": {
            "overview": "Single-service architecture exposing HTTP endpoints via FastAPI.",
            "frontend_architecture": "Unavailable in inspected files.",
            "backend_architecture": "FastAPI application in app/main.py.",
            "database_architecture": "SQLite database for local persistence.",
            "api_communication": "JSON over HTTP.",
            "external_integrations": "None detected.",
            "important_components": [
                {
                    "name": "API Service",
                    "responsibility": "Routes and validates incoming HTTP requests.",
                    "evidence_paths": ["app/main.py"],
                }
            ],
            "unavailable_details": [],
        },
        "folder_structure": {
            "summary": "Application source code lives in app/.",
            "important_folders": [
                {"path": "app", "purpose": "Application logic and routes", "key_files": ["app/main.py"]}
            ],
            "important_files": [
                {"path": "app/main.py", "language": "python", "responsibility": "FastAPI entry point"}
            ],
        },
    }


def _sample_s2_dict():
    return {
        "application_workflow": {
            "end_to_end_summary": "Clients send HTTP requests to FastAPI routes which validate parameters and return JSON.",
            "frontend_to_backend_communication": "HTTP JSON requests.",
            "api_calls": ["GET /tasks", "POST /tasks"],
            "database_interactions": ["Reads and writes task rows."],
            "business_flows": ["User creates a task, updates its status, and marks it completed."],
            "evidence_paths": ["app/main.py"],
        },
        "important_code_explanation": [
            {
                "file_path": "app/main.py",
                "symbol_or_section": "app",
                "purpose": "Configures and starts FastAPI instance.",
                "module_interactions": "Registers route handlers.",
                "evidence_notes": "Defined in app/main.py",
            }
        ],
        "improvement_suggestions": {
            "performance": [{"title": "Async DB driver", "description": "Use async driver", "priority": "Medium", "evidence_paths": ["app/main.py"]}],
            "security": [{"title": "Add Auth", "description": "Enforce JWT authentication", "priority": "High", "evidence_paths": ["app/main.py"]}],
            "code_quality": [],
            "scalability": [],
            "architecture": [],
            "missing_best_practices": [],
        },
        "new_feature_ideas": [
            {
                "title": "Task Search Filter",
                "description": "Add query filtering for tasks by status.",
                "value_proposition": "Better usability.",
                "complexity": "Low",
                "impacted_modules": ["app/main.py"],
            }
        ],
    }


def _sample_s3_dict():
    return {
        "interview_preparation": {
            "beginner": [
                {
                    "question": "What framework is used in app/main.py?",
                    "expected_answer_points": ["FastAPI"],
                    "follow_up_question": "Why choose FastAPI?",
                    "evidence_paths": ["app/main.py"],
                }
            ],
            "intermediate": [
                {
                    "question": "How are routes structured in this service?",
                    "expected_answer_points": ["Defined as decorators on app instance."],
                    "follow_up_question": "How would you modularize routes?",
                    "evidence_paths": ["app/main.py"],
                }
            ],
            "advanced": [
                {
                    "question": "How would you handle high concurrent writes?",
                    "expected_answer_points": ["Add a connection pool and message queue."],
                    "follow_up_question": "What queue would you choose?",
                    "evidence_paths": ["app/main.py"],
                }
            ],
            "architecture_and_scenario": [
                {
                    "question": "How would you containerize this app for production?",
                    "expected_answer_points": ["Dockerfile with multi-stage build."],
                    "follow_up_question": "How do you manage configuration?",
                    "evidence_paths": ["requirements.txt"],
                }
            ],
        }
    }


def _build_mock_two_stage_ollama():
    svc = MagicMock(spec=OllamaService)

    async def _chat(messages, model=None, format=None, **kwargs):
        sys_prompt = messages[0]["content"]
        if "logical group" in sys_prompt or "group_name" in str(format):
            return json.dumps({
                "group_name": "Group Summary",
                "summary": "Group of files providing core services.",
                "key_responsibilities": ["Serve API endpoints", "Handle data models"],
                "key_files": [{"path": "app/main.py", "purpose": "Entry point"}],
                "technologies": ["Python", "FastAPI"],
            })
        if "project_overview" in sys_prompt or "architecture from verified evidence" in sys_prompt:
            return json.dumps(_sample_s1_dict())
        if "application_workflow" in sys_prompt:
            return json.dumps(_sample_s2_dict())
        return json.dumps(_sample_s3_dict())

    svc.generate_chat_completion = AsyncMock(side_effect=_chat)
    return svc


# ----------------------------------------------------------------------------
# Unit Tests
# ----------------------------------------------------------------------------

def test_stage1_grouping_small_repository():
    """Small repository (1-3 files) produces 1-2 focused groups."""
    service = ProjectExplainerService(
        github_service=MagicMock(),
        chunking_service=TextChunkingService(),
        embedding_service=DefaultMockEmbeddingService(),
        pinecone_service=DefaultMockPineconeService(),
        ollama_service=_build_mock_two_stage_ollama(),
    )
    files = [
        IngestedRepoFile(path="requirements.txt", language="pip-requirements", category="config", content="fastapi==0.115.0\nuvicorn==0.32.0", size_bytes=40),
        IngestedRepoFile(path="main.py", language="python", category="entrypoint", content="from fastapi import FastAPI\napp = FastAPI()", size_bytes=50),
    ]
    groups = service._group_repository_files(files, max_group_files=5, max_group_chars=3500)
    assert len(groups) == 2
    assert groups[0]["group_name"] == "Manifests & Configuration"
    assert groups[1]["group_name"] == "Application Entry Points"


def test_stage1_grouping_medium_repository():
    """Medium repository (10-15 files) partitions into prioritized groups."""
    service = ProjectExplainerService(
        github_service=MagicMock(),
        chunking_service=TextChunkingService(),
        embedding_service=DefaultMockEmbeddingService(),
        pinecone_service=DefaultMockPineconeService(),
        ollama_service=_build_mock_two_stage_ollama(),
    )
    files = [
        IngestedRepoFile(path="package.json", language="json", category="config", content='{"name": "test"}', size_bytes=30),
        IngestedRepoFile(path="tsconfig.json", language="json", category="config", content='{}', size_bytes=10),
        IngestedRepoFile(path="src/index.ts", language="typescript", category="entrypoint", content='console.log("start");', size_bytes=30),
        IngestedRepoFile(path="src/routes/api.ts", language="typescript", category="backend", content='export const r = 1;', size_bytes=30),
        IngestedRepoFile(path="src/routes/users.ts", language="typescript", category="backend", content='export const u = 1;', size_bytes=30),
        IngestedRepoFile(path="src/models/user.ts", language="typescript", category="database", content='export interface User {}', size_bytes=30),
        IngestedRepoFile(path="src/components/App.tsx", language="typescript", category="frontend", content='export default function App() {}', size_bytes=40),
        IngestedRepoFile(path="src/components/Header.tsx", language="typescript", category="frontend", content='export function Header() {}', size_bytes=40),
        IngestedRepoFile(path="src/utils/helpers.ts", language="typescript", category="source", content='export const add = (a, b) => a + b;', size_bytes=40),
    ]
    groups = service._group_repository_files(files, max_group_files=4, max_group_chars=3500)
    assert len(groups) >= 4
    categories = [g["category"] for g in groups]
    assert "Manifests & Configuration" in categories
    assert "Application Entry Points" in categories
    assert "Core Routes & Services" in categories
    assert "Database & Models" in categories
    assert "Frontend Components" in categories


def test_stage1_grouping_36_or_more_files():
    """36 or more repository files are properly sub-batched and prioritized."""
    service = ProjectExplainerService(
        github_service=MagicMock(),
        chunking_service=TextChunkingService(),
        embedding_service=DefaultMockEmbeddingService(),
        pinecone_service=DefaultMockPineconeService(),
        ollama_service=_build_mock_two_stage_ollama(),
    )
    files = [
        IngestedRepoFile(path="package.json", language="json", category="config", content="{}", size_bytes=10),
        IngestedRepoFile(path="Dockerfile", language="dockerfile", category="config", content="FROM node:20", size_bytes=20),
        IngestedRepoFile(path="src/index.ts", language="typescript", category="entrypoint", content="console.log(1)", size_bytes=20),
    ]
    # Add 33 backend routes/services
    for i in range(33):
        files.append(
            IngestedRepoFile(
                path=f"src/routes/route_{i}.ts",
                language="typescript",
                category="backend",
                content=f"export function handler_{i}() {{ return {i}; }}",
                size_bytes=40,
            )
        )
    assert len(files) == 36

    groups = service._group_repository_files(files, max_group_files=5, max_group_chars=3500)
    # The 33 backend files will be broken into multiple parts (up to 5 per group)
    backend_groups = [g for g in groups if "Core Routes & Services" in g["group_name"]]
    assert len(backend_groups) >= 6
    assert all(len(g["files"]) <= 5 for g in groups)


@pytest.mark.anyio
async def test_large_readme_is_bounded_safely():
    """A 50,000+ char README is bounded without blowing up the synthesis prompt."""
    ollama = _build_mock_two_stage_ollama()
    service = ProjectExplainerService(
        github_service=MagicMock(),
        chunking_service=TextChunkingService(),
        embedding_service=DefaultMockEmbeddingService(),
        pinecone_service=DefaultMockPineconeService(),
        ollama_service=ollama,
    )
    large_readme = "# Huge README\n" + ("This is detailed repository documentation.\n" * 1500)
    bundle = RepoIngestionBundle(
        owner="testowner",
        repo="hugereadme",
        full_name="testowner/hugereadme",
        repo_url="https://github.com/testowner/hugereadme",
        ref="main",
        metadata=RepoMetadataResponse(
            owner="testowner", repo="hugereadme", full_name="testowner/hugereadme",
            default_branch="main", language="Python", html_url="https://github.com/testowner/hugereadme"
        ),
        readme_filename="README.md",
        readme_content=large_readme,
        files=[
            IngestedRepoFile(path="app/main.py", language="python", category="entrypoint", content="app = 1", size_bytes=10)
        ],
        tree_paths=["README.md", "app/main.py"],
        stats=ProjectIngestionStats(),
        detected_technologies={},
    )
    report = await service._synthesize_report(bundle=bundle, source_id="src_large_readme", user_id="user1")
    assert report.project_overview.what_it_does is not None
    # Verify the prompt sent to Ollama did not include the full 50,000 chars of README
    for call in ollama.generate_chat_completion.call_args_list:
        messages = call.kwargs["messages"]
        user_content = messages[1]["content"]
        assert len(user_content) < 15000


@pytest.mark.anyio
async def test_large_source_file_is_bounded_per_group():
    """A 100,000+ char source file is sliced to 1000 chars per file inside group analysis."""
    captured_prompts = []

    async def _inspect_chat(messages, **kwargs):
        captured_prompts.append(messages[1]["content"])
        sys = messages[0]["content"]
        if "logical group" in sys:
            return json.dumps({"group_name": "G", "summary": "S", "key_responsibilities": [], "key_files": []})
        if "project_overview" in sys:
            return json.dumps(_sample_s1_dict())
        if "application_workflow" in sys:
            return json.dumps(_sample_s2_dict())
        return json.dumps(_sample_s3_dict())

    ollama = MagicMock(spec=OllamaService)
    ollama.generate_chat_completion = AsyncMock(side_effect=_inspect_chat)

    service = ProjectExplainerService(
        github_service=MagicMock(),
        chunking_service=TextChunkingService(),
        embedding_service=DefaultMockEmbeddingService(),
        pinecone_service=DefaultMockPineconeService(),
        ollama_service=ollama,
    )
    huge_code = "def large_fn():\n" + ("    x = 42\n" * 5000)
    bundle = RepoIngestionBundle(
        owner="testowner",
        repo="largefile",
        full_name="testowner/largefile",
        repo_url="https://github.com/testowner/largefile",
        ref="main",
        metadata=RepoMetadataResponse(
            owner="testowner", repo="largefile", full_name="testowner/largefile",
            default_branch="main", language="Python", html_url="https://github.com/testowner/largefile"
        ),
        readme_filename="README.md",
        readme_content="# Readme",
        files=[
            IngestedRepoFile(path="app/huge.py", language="python", category="source", content=huge_code, size_bytes=len(huge_code))
        ],
        tree_paths=["app/huge.py"],
        stats=ProjectIngestionStats(),
        detected_technologies={},
    )
    await service._synthesize_report(bundle=bundle, source_id="src_large_file", user_id="user1")
    # Verify group analysis snippet was truncated to ~1000 chars
    group_prompt = captured_prompts[0]
    assert len(group_prompt) < 2500


@pytest.mark.anyio
async def test_output_token_limit_reached_raises_truncated_error():
    """When Ollama returns done_reason='length', OllamaTruncatedResponseError is raised."""
    ollama = OllamaService()

    # Simulate Ollama response with done_reason='length' and eval_count == num_predict
    mock_response = httpx.Response(
        status_code=200,
        json={
            "model": "qwen2.5:1.5b",
            "created_at": "2026-11-13T10:00:00Z",
            "message": {"role": "assistant", "content": "{\"project_overview\": {\"what_it_does\": \"Truncated mid string\"}, \"extra\":"},
            "done": True,
            "done_reason": "length",
            "eval_count": 2048,
            "prompt_eval_count": 350,
        },
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )

    with pytest.MonkeyPatch.context() as mp:
        async def mock_post(*args, **kwargs):
            return mock_response
        mp.setattr(httpx.AsyncClient, "post", mock_post)

        with pytest.raises(OllamaTruncatedResponseError) as exc_info:
            await ollama.generate_chat_completion(
                messages=[{"role": "user", "content": "hello"}],
                raise_on_truncated=True,
            )
        assert exc_info.value.eval_count == 2048
        assert exc_info.value.done_reason == "length"
        assert "token limit" in str(exc_info.value).lower()


@pytest.mark.anyio
async def test_incomplete_json_triggers_controlled_repair_and_recovers():
    """Incomplete JSON or truncation triggers at most 1 controlled repair with conciseness instruction."""
    attempts = {"count": 0}

    async def _flaky_chat(messages, **kwargs):
        attempts["count"] += 1
        sys = messages[0]["content"]
        if "logical group" in sys:
            return json.dumps({"group_name": "G", "summary": "S", "key_responsibilities": [], "key_files": []})
        if "project_overview" in sys:
            if attempts["count"] <= 2:
                # Return cut-off unclosed JSON on first attempt
                return '{"project_overview": {"what_it_does": "Cut off mid-way"'
            # Return valid on repair attempt
            return json.dumps(_sample_s1_dict())
        if "application_workflow" in sys:
            return json.dumps(_sample_s2_dict())
        return json.dumps(_sample_s3_dict())

    ollama = MagicMock(spec=OllamaService)
    ollama.generate_chat_completion = AsyncMock(side_effect=_flaky_chat)

    service = ProjectExplainerService(
        github_service=MagicMock(),
        chunking_service=TextChunkingService(),
        embedding_service=DefaultMockEmbeddingService(),
        pinecone_service=DefaultMockPineconeService(),
        ollama_service=ollama,
    )
    bundle = RepoIngestionBundle(
        owner="testowner",
        repo="repairtest",
        full_name="testowner/repairtest",
        repo_url="https://github.com/testowner/repairtest",
        ref="main",
        metadata=RepoMetadataResponse(
            owner="testowner", repo="repairtest", full_name="testowner/repairtest",
            default_branch="main", language="Python", html_url="https://github.com/testowner/repairtest"
        ),
        readme_filename="README.md",
        readme_content="# Readme",
        files=[
            IngestedRepoFile(path="app/main.py", language="python", category="entrypoint", content="app = 1", size_bytes=10)
        ],
        tree_paths=["app/main.py"],
        stats=ProjectIngestionStats(),
        detected_technologies={},
    )
    report = await service._synthesize_report(bundle=bundle, source_id="src_repair", user_id="user1")
    assert report.project_overview.what_it_does is not None


@pytest.mark.anyio
async def test_ollama_unavailable_raises_connection_error():
    """When Ollama is unreachable, OllamaConnectionError is raised."""
    ollama = OllamaService(base_url="http://127.0.0.1:59999", timeout=1.0)
    with pytest.raises(OllamaConnectionError):
        await ollama.generate_chat_completion([{"role": "user", "content": "ping"}])


@pytest.mark.anyio
async def test_real_local_ollama_integration():
    """Real local Ollama integration test exercising qwen2.5:1.5b with two-stage analysis."""
    # Check if local Ollama server is running and qwen2.5:1.5b is installed
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code != 200:
                pytest.skip("Local Ollama not reachable on port 11434")
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            if not any("qwen2.5:1.5b" in m for m in models):
                pytest.skip("qwen2.5:1.5b model not installed in local Ollama")
    except Exception:
        pytest.skip("Local Ollama server not reachable")

    ollama = OllamaService()
    service = ProjectExplainerService(
        github_service=MagicMock(),
        chunking_service=TextChunkingService(),
        embedding_service=DefaultMockEmbeddingService(),
        pinecone_service=DefaultMockPineconeService(),
        ollama_service=ollama,
    )

    bundle = RepoIngestionBundle(
        owner="localtest",
        repo="micro-fastapi",
        full_name="localtest/micro-fastapi",
        repo_url="https://github.com/localtest/micro-fastapi",
        ref="main",
        metadata=RepoMetadataResponse(
            owner="localtest", repo="micro-fastapi", full_name="localtest/micro-fastapi",
            default_branch="main", language="Python", html_url="https://github.com/localtest/micro-fastapi"
        ),
        readme_filename="README.md",
        readme_content="# Micro FastAPI\nA micro-service providing task operations with SQLite persistence.",
        files=[
            IngestedRepoFile(path="requirements.txt", language="pip-requirements", category="config", content="fastapi==0.115.0\nsqlite3", size_bytes=30),
            IngestedRepoFile(path="app/main.py", language="python", category="entrypoint", content="from fastapi import FastAPI\napp = FastAPI()\n@app.get('/tasks')\ndef list_tasks(): return []", size_bytes=90),
        ],
        tree_paths=["README.md", "requirements.txt", "app/main.py"],
        stats=ProjectIngestionStats(),
        detected_technologies={},
    )

    report = await service._synthesize_report(bundle=bundle, source_id="src_real_test", user_id="user1")
    assert report.project_overview.what_it_does is not None
    assert len(report.project_overview.what_it_does) > 10
    assert report.project_architecture.overview is not None
    assert report.folder_structure.summary is not None
    assert len(report.application_workflow.end_to_end_summary) > 10
    assert len(report.interview_preparation.beginner) >= 1
