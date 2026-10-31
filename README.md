# AI Interview Preparation Platform - Backend (Day 6: Frontend API Integration Readiness)

A robust, enterprise-ready Python backend for an AI-powered technical interview preparation platform. Built with **FastAPI**, **SQLAlchemy**, **Alembic**, **PostgreSQL on Neon**, **HTTPX**, and a local **Ollama** runtime hosting **Qwen 2.5 (`qwen2.5:1.5b`)**.

---

## 🏗️ Architecture Overview

```mermaid
graph LR
    Frontend[React + Vite Frontend :5173] -->|HTTP / JSON (CORS Allowed)| FastAPI[FastAPI Backend :8000]
    FastAPI -->|SQLAlchemy ORM / Psycopg 3| NeonDB[(Neon PostgreSQL Cloud)]
    FastAPI -->|HTTPX Async Client| Ollama[Ollama Local Runtime :11434]
    Ollama -->|Inference| Qwen[Qwen 2.5 Model 1.5B]
```

### Key Highlights
- **Frontend Integration Ready (Day 6)**: Configured CORS middleware supporting `http://localhost:5173` and `http://127.0.0.1:5173` for Vite/React development.
- **Strict API Contracts & Error Handling**: Comprehensive Pydantic V2 request/response schemas, returning standard HTTP `422` on validation errors, `404` on missing records, and safe `500` error masks that never expose secrets or stack traces.
- **Direct Local AI Inference**: Native communication with Ollama `/api/chat` and `/api/tags` using async HTTPX. No third-party AI framework overhead.
- **Relational Persistence**: PostgreSQL on Neon storing session configurations, generated questions, and answer evaluations via SQLAlchemy 2.0 with Alembic version-controlled migrations.
- **Dual Runtime Support**: Operates seamlessly in cloud-connected database mode (Neon) or offline fallback mode with in-memory caching.
- **Automated Test Suite**: 93 automated tests (92 passing, 1 skipped live integration test) using `pytest`, isolated SQLite in-memory databases, and mocked AI inference.

---

## 📋 Prerequisites

Before running the backend, make sure you have:
1. **Python**: Python 3.10+ (Tested on Python 3.14.6).
2. **Ollama**: Installed and running locally ([ollama.com](https://ollama.com)).
   - Verify: `ollama --version`
   - Model: Ensure `qwen2.5:1.5b` is downloaded (`ollama pull qwen2.5:1.5b`).
3. **PostgreSQL / Neon Account**: A Neon serverless PostgreSQL connection string (or local PostgreSQL instance).

---

## 📁 Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application factory, CORS, exception handlers
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py               # Pydantic Settings, CORS origins, environment loader
│   ├── db/
│   │   ├── __init__.py             # Database session and connection exports
│   │   ├── base.py                 # SQLAlchemy DeclarativeBase
│   │   └── session.py              # Managed sessionmaker, URL normalization, masking
│   ├── models/
│   │   ├── __init__.py             # SQLAlchemy models index
│   │   └── interview.py            # InterviewSession, InterviewQuestion, AnswerEvaluation
│   ├── schemas/
│   │   ├── __init__.py             # Central schema exports
│   │   ├── health.py               # Health check response schemas
│   │   ├── interview.py            # Question generation & evaluation schemas
│   │   └── session.py              # Session management, questions, summary schemas
│   ├── services/
│   │   ├── __init__.py             # Service exports
│   │   ├── ollama_service.py       # Async HTTPX client for Ollama LLM integration
│   │   └── session_service.py      # Stateful interview lifecycle & persistence manager
│   └── api/
│       ├── __init__.py
│       └── routes/
│           ├── __init__.py
│           ├── health.py           # Backend, Ollama, and Database health checks
│           └── interview.py        # Question, evaluation, session CRUD and listing
├── alembic/
│   ├── env.py                      # Alembic migration runner & metadata binder
│   ├── script.py.mako              # Migration script template
│   └── versions/
│       └── 20261016_0001_initial_interview_tables.py  # Initial PostgreSQL schema
├── tests/
│   ├── __init__.py
│   ├── test_cors.py                # CORS, contract validation, and safe 500 error tests
│   ├── test_database.py            # SQLAlchemy models, SQLite isolated DB persistence tests
│   ├── test_health.py              # Health check endpoint unit tests (mocked)
│   ├── test_interview.py           # Standalone question & evaluation unit tests (mocked)
│   └── test_session.py             # Session lifecycle and listing unit tests (mocked)
├── alembic.ini                     # Alembic configuration
├── .env.example                    # Safe environment template (no real secrets)
├── .env                            # Local environment configuration (git-ignored)
├── .gitignore                      # Git ignore file
├── requirements.txt                # Pinned Python dependencies
└── README.md                       # Comprehensive documentation
```

---

## 🚀 Quickstart Guide

### 1. Activate the Virtual Environment

Always work inside the Python virtual environment:

```bash
# Navigate to the backend directory
cd /Users/rahul/Documents/Ai-Interview-Platform/BackEnd

# Create a virtual environment (if not already created)
python3 -m venv .venv

# Activate the virtual environment:
# On macOS / Linux:
source .venv/bin/activate
# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

Install all pinned project dependencies:

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables (`.env`)

Copy the template `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` to configure your settings (**never commit real credentials to git**):

```env
# Application Configuration
APP_NAME="AI Interview Preparation Platform"
APP_ENV=development
API_V1_PREFIX=/api/v1

# CORS Allowed Origins (comma-separated for frontend integration)
CORS_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"

# Ollama LLM Service Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:1.5b
OLLAMA_TIMEOUT_SECONDS=60.0

# Neon PostgreSQL Database Configuration
# Format: postgresql+psycopg://USER:PASSWORD@HOST.neon.tech/neondb?sslmode=require
DATABASE_URL="postgresql+psycopg://USER:PASSWORD@ep-sample-host.us-east-2.aws.neon.tech/neondb?sslmode=require"
```

> [!TIP]
> If `DATABASE_URL` is omitted, the backend runs in **offline in-memory mode** for lightweight local testing without a database connection.

### 4. Run Database Migrations

Apply database schema migrations to your PostgreSQL database:

```bash
# Check current migration version
alembic current

# Run migrations to the latest revision
alembic upgrade head
```

### 5. Ensure Ollama is Running

Verify Ollama is active and hosting the configured model:

```bash
ollama list
```

If Ollama is stopped, start it in a separate terminal:
```bash
ollama serve
```

### 6. Start the FastAPI Server

Launch the development server with live reload:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

The backend is now live at:
- **Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative ReDoc UI**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🌐 Frontend Integration & CORS Configuration

The backend is pre-configured with FastAPI's `CORSMiddleware` to integrate smoothly with a **React + TypeScript (Vite)** frontend:

| Setting | Value | Description |
| :--- | :--- | :--- |
| **Allowed Origins** | `http://localhost:5173`<br>`http://127.0.0.1:5173` | Configured via `CORS_ORIGINS` in `.env` |
| **Allowed Methods** | `GET, POST, PUT, DELETE, OPTIONS, PATCH` | All required standard REST verbs |
| **Allowed Headers** | `*` (`Content-Type`, `Authorization`, etc.) | Custom frontend headers permitted |
| **Allow Credentials**| `False` | Cookies/credentials disabled until authentication is introduced |

---

## 📡 API Reference & JSON Contracts

All endpoints are prefixed with `/api/v1` (configurable via `API_V1_PREFIX`).

### 1. Health Endpoints

#### `GET /api/v1/health`
Check core FastAPI backend operational status.

**Response (`200 OK`):**
```json
{
  "status": "ok",
  "app_name": "AI Interview Preparation Platform",
  "environment": "development"
}
```

#### `GET /api/v1/health/ollama`
Verify Ollama server reachability and model availability.

**Response (`200 OK`):**
```json
{
  "status": "ok",
  "ollama_connected": true,
  "configured_model": "qwen2.5:1.5b",
  "model_available": true,
  "available_models": ["qwen2.5:1.5b"]
}
```

#### `GET /api/v1/health/db`
Verify PostgreSQL database connectivity (credentials masked).

**Response (`200 OK`):**
```json
{
  "status": "ok",
  "database": "postgresql",
  "database_connected": true,
  "database_configured": true
}
```

---

### 2. Standalone AI Generation & Evaluation

#### `POST /api/v1/interview/question`
Generate an isolated interview question using Qwen 2.5 without session persistence.

**Request Body:**
```json
{
  "role": "Frontend Developer",
  "experience_level": "Junior",
  "difficulty": "Medium",
  "topic": "React Hooks",
  "question_type": "Technical"
}
```

**Response (`200 OK`):**
```json
{
  "question": "Can you explain the differences between useEffect and useLayoutEffect in React?",
  "expected_answer_points": [
    "useEffect runs asynchronously after DOM paint",
    "useLayoutEffect runs synchronously immediately after DOM mutations before paint",
    "useLayoutEffect is typically used for measuring layout or synchronous DOM updates"
  ],
  "follow_up_question": "When might using useLayoutEffect degrade application performance?",
  "difficulty": "Medium",
  "topic": "React Hooks"
}
```

#### `POST /api/v1/interview/evaluate`
Evaluate an isolated candidate answer against expected criteria.

**Request Body:**
```json
{
  "question": "Explain the difference between useEffect and useLayoutEffect.",
  "expected_answer_points": [
    "useEffect runs asynchronously after paint",
    "useLayoutEffect runs synchronously before paint"
  ],
  "candidate_answer": "useEffect fires asynchronously after the browser paints the screen. useLayoutEffect runs synchronously before painting, which is helpful to prevent UI flicker when calculating dimensions."
}
```

**Response (`200 OK`):**
```json
{
  "overall_score": 90,
  "rating": "Excellent",
  "correct_points": [
    "useEffect runs asynchronously after paint",
    "useLayoutEffect runs synchronously before paint"
  ],
  "missing_points": [],
  "strengths": [
    "Accurately differentiated asynchronous vs synchronous execution timings",
    "Identified use case of preventing visual flicker"
  ],
  "improvements": [
    "Mention performance implications when layout computations take too long"
  ],
  "sample_answer": "useEffect fires asynchronously after DOM paint is complete. useLayoutEffect fires synchronously after layout mutations but before browser paint, making it suitable for DOM measurements."
}
```

---

### 3. Interview Session Management

#### `POST /api/v1/interview/sessions`
Initialize a new interview session and persist configuration to PostgreSQL.

**Request Body:**
```json
{
  "role": "Frontend Developer",
  "experience_level": "Junior",
  "difficulty": "Medium",
  "topic": "React Hooks",
  "question_type": "Technical",
  "total_questions": 3
}
```

**Response (`201 Created`):**
```json
{
  "session_id": "e058e49c-d69a-4ffb-b949-ded55736acca",
  "role": "Frontend Developer",
  "experience_level": "Junior",
  "difficulty": "Medium",
  "topic": "React Hooks",
  "question_type": "Technical",
  "total_questions": 3,
  "completed_questions": 0,
  "current_question_number": 0,
  "status": "in_progress",
  "questions": [],
  "created_at": "2026-10-26T06:00:00+00:00",
  "updated_at": "2026-10-26T06:00:00+00:00"
}
```

#### `GET /api/v1/interview/sessions`
List interview sessions ordered by most recent, with optional pagination.

**Query Parameters:**
- `limit` (int, default `50`): Maximum sessions to retrieve.
- `skip` (int, default `0`): Number of sessions to skip.

**Response (`200 OK`):**
```json
[
  {
    "session_id": "e058e49c-d69a-4ffb-b949-ded55736acca",
    "role": "Frontend Developer",
    "experience_level": "Junior",
    "difficulty": "Medium",
    "topic": "React Hooks",
    "question_type": "Technical",
    "total_questions": 3,
    "completed_questions": 1,
    "current_question_number": 1,
    "status": "in_progress",
    "questions": [...],
    "created_at": "2026-10-26T06:00:00+00:00",
    "updated_at": "2026-10-26T06:05:00+00:00"
  }
]
```

#### `GET /api/v1/interview/sessions/{session_id}`
Retrieve complete session details, question history, and evaluations.

**Response (`200 OK`):**
```json
{
  "session_id": "e058e49c-d69a-4ffb-b949-ded55736acca",
  "role": "Frontend Developer",
  "experience_level": "Junior",
  "difficulty": "Medium",
  "topic": "React Hooks",
  "question_type": "Technical",
  "total_questions": 3,
  "completed_questions": 1,
  "current_question_number": 1,
  "status": "in_progress",
  "questions": [
    {
      "question_number": 1,
      "question": "Can you explain the differences between useEffect and useLayoutEffect in React?",
      "expected_answer_points": [
        "useEffect runs asynchronously after DOM paint",
        "useLayoutEffect runs synchronously immediately after DOM mutations before paint"
      ],
      "follow_up_question": "When might using useLayoutEffect degrade application performance?",
      "difficulty": "Medium",
      "topic": "React Hooks",
      "candidate_answer": "useEffect runs asynchronously after paint...",
      "evaluation": {
        "overall_score": 90,
        "rating": "Excellent",
        "correct_points": ["useEffect runs asynchronously after paint"],
        "missing_points": [],
        "strengths": ["Accurately differentiated asynchronous vs synchronous execution"],
        "improvements": [],
        "sample_answer": "..."
      }
    }
  ],
  "created_at": "2026-10-26T06:00:00+00:00",
  "updated_at": "2026-10-26T06:05:00+00:00"
}
```

#### `POST /api/v1/interview/sessions/{session_id}/next-question`
Generate the next tailored question and persist it to PostgreSQL.

**Response (`200 OK`):**
```json
{
  "session_id": "e058e49c-d69a-4ffb-b949-ded55736acca",
  "question_number": 1,
  "total_questions": 3,
  "question": "Can you explain the differences between useEffect and useLayoutEffect in React?",
  "expected_answer_points": [
    "useEffect runs asynchronously after DOM paint",
    "useLayoutEffect runs synchronously immediately after DOM mutations before paint"
  ],
  "follow_up_question": "When might using useLayoutEffect degrade application performance?",
  "difficulty": "Medium",
  "topic": "React Hooks"
}
```

#### `POST /api/v1/interview/sessions/{session_id}/answers`
Submit candidate answer and record rubric-aligned AI evaluation.

**Request Body:**
```json
{
  "candidate_answer": "useEffect fires asynchronously after the browser paints the screen. useLayoutEffect runs synchronously before painting to prevent flickering."
}
```

**Response (`200 OK`):**
```json
{
  "session_id": "e058e49c-d69a-4ffb-b949-ded55736acca",
  "question_number": 1,
  "completed_questions": 1,
  "total_questions": 3,
  "status": "in_progress",
  "evaluation": {
    "overall_score": 90,
    "rating": "Excellent",
    "correct_points": ["useEffect runs asynchronously after paint"],
    "missing_points": [],
    "strengths": ["Accurately differentiated execution timings"],
    "improvements": [],
    "sample_answer": "..."
  }
}
```

#### `POST /api/v1/interview/sessions/{session_id}/complete`
Mark session completed and calculate aggregate statistics and strengths.

**Response (`200 OK`):**
```json
{
  "session_id": "e058e49c-d69a-4ffb-b949-ded55736acca",
  "role": "Frontend Developer",
  "topic": "React Hooks",
  "total_questions": 3,
  "questions_answered": 1,
  "average_score": 90.0,
  "overall_rating": "Excellent",
  "aggregated_strengths": ["Accurately differentiated execution timings"],
  "aggregated_improvement_areas": [],
  "session_status": "completed"
}
```

---

### 4. Standard Error Response Contract

The backend adheres strictly to consistent JSON error responses:

| HTTP Status | Condition | Response Structure |
| :--- | :--- | :--- |
| **`422 Unprocessable Entity`** | Invalid or missing request body fields | `{"detail": [{"loc": ["body", "role"], "msg": "Field required", "type": "missing"}]}` |
| **`404 Not Found`** | Requested session ID does not exist | `{"detail": "Interview session '<id>' not found."}` |
| **`400 Bad Request`** | Invalid state transition (e.g. question already answered) | `{"detail": "Current question #1 has not been answered yet..."}` |
| **`500 Internal Server Error`** | Unhandled internal exception (secrets masked) | `{"detail": "An unexpected internal server error occurred."}` |
| **`502 Bad Gateway`** | Ollama LLM returned malformed JSON | `{"detail": "The AI model returned an invalid JSON response."}` |
| **`503 Service Unavailable`** | Ollama LLM or Neon Database unreachable | `{"detail": "AI inference service is currently unavailable..."}` |

---

## 🧪 Running Automated Tests

The automated test suite runs 100% offline without requiring a live Ollama instance or Neon database connection. Tests use isolated SQLite in-memory databases and dependency-injected mocks:

```bash
# Activate your virtual environment
source .venv/bin/activate

# Run all tests
pytest -v

# Or run via the virtual environment binary directly:
.venv/bin/pytest -v
```

### Test Suite Structure

| Test File | Scope | Test Count |
| :--- | :--- | :--- |
| `tests/test_cors.py` | CORS preflight/actual requests, 422 validations, 404 missing sessions, safe 500 handler | 15 tests |
| `tests/test_database.py` | SQLAlchemy models, URL normalization & masking, isolated SQLite persistence | 20 tests (1 live skipped) |
| `tests/test_health.py` | Root status, backend health, Ollama health, DB connectivity | 5 tests |
| `tests/test_interview.py` | Standalone question generation & evaluation, Ollama mocking, error handling | 23 tests |
| `tests/test_session.py` | Multi-question session lifecycle, answer recording, session listing, summaries | 30 tests |
| **Total** | | **93 tests (92 passed, 1 skipped)** |

---

## 💡 Frontend Integration Guide (React + TypeScript)

For frontend developers connecting a React / Next.js client to this backend:

1. **Base API URL**: Set your API client base URL to `http://127.0.0.1:8000/api/v1`.
2. **CORS Headers**: Ensure requests originate from `http://localhost:5173` or `http://127.0.0.1:5173` with standard `Content-Type: application/json`.
3. **Session Flow**:
   - `POST /interview/sessions` -> Store `session_id` in React state.
   - `POST /interview/sessions/{session_id}/next-question` -> Fetch question for candidate.
   - `POST /interview/sessions/{session_id}/answers` -> Submit candidate answer and render evaluation rubric.
   - `POST /interview/sessions/{session_id}/complete` -> Render final interview score report and aggregated feedback.
