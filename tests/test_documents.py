"""Tests for secure PDF document upload, validation, and text extraction."""

import io
from fastapi.testclient import TestClient
import pypdf
import pytest

from app.main import app
from app.schemas.document import DocumentUploadResponse
from app.services.pdf_service import PDFService, get_pdf_service

client = TestClient(app)


# ============================================================================
# PDF Generation Test Helpers
# ============================================================================


def create_test_pdf(text: str = "Senior Software Engineer Resume", pages: int = 1) -> bytes:
    """Generates a valid, readable in-memory PDF containing extractable text across specified pages."""
    writer = pypdf.PdfWriter()
    for i in range(pages):
        page_stream = (
            f"BT\n/F1 12 Tf\n100 700 Td\n({text} - Page {i+1}) Tj\nET"
        ).encode("latin-1")
        raw_pdf = (
            f"%PDF-1.4\n"
            f"1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n"
            f"2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> endobj\n"
            f"3 0 obj <</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources <</Font <</F1 5 0 R>>>>>> endobj\n"
            f"4 0 obj <</Length {len(page_stream)}>> stream\n"
            f"{page_stream.decode('latin-1')}\n"
            f"endstream endobj\n"
            f"5 0 obj <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>> endobj\n"
            f"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000056 00000 n \n0000000111 00000 n \n0000000216 00000 n \n0000000300 00000 n \n"
            f"trailer <</Size 6 /Root 1 0 R>>\nstartxref\n380\n%%EOF"
        ).encode("latin-1")

        reader = pypdf.PdfReader(io.BytesIO(raw_pdf))
        writer.add_page(reader.pages[0])

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def create_blank_pdf(pages: int = 1) -> bytes:
    """Generates a valid PDF containing blank pages with zero extractable text."""
    writer = pypdf.PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=100, height=100)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def create_encrypted_pdf(text: str = "Confidential resume", password: str = "secret123") -> bytes:
    """Generates a password-protected encrypted PDF document."""
    pdf_bytes = create_test_pdf(text=text, pages=1)
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    writer = pypdf.PdfWriter()
    writer.append(reader)
    writer.encrypt(password)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


# ============================================================================
# Test Cases
# ============================================================================


def test_upload_valid_single_page_pdf():
    """Test successful upload and text extraction from a valid single-page PDF."""
    pdf_bytes = create_test_pdf("Python & React Architect", pages=1)
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("candidate_resume.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["filename"] == "candidate_resume.pdf"
    assert data["page_count"] == 1
    assert data["character_count"] > 0
    assert "Python & React Architect" in data["preview"]
    assert len(data["preview"]) <= 500
    assert "Successfully extracted" in data["message"]

    # Validate against Pydantic schema
    validated = DocumentUploadResponse(**data)
    assert validated.success is True


def test_upload_valid_multipage_pdf():
    """Test successful upload and multi-page text aggregation from a 3-page PDF."""
    pdf_bytes = create_test_pdf("Senior DevOps Engineer", pages=3)
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("multi_page_resume.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["page_count"] == 3
    assert data["character_count"] > 0
    assert "Senior DevOps Engineer" in data["preview"]


def test_upload_pdf_preview_capped_at_500_chars():
    """Verify preview is strictly capped at 500 characters for long documents."""
    long_text = "Experienced Backend Engineer specializing in distributed architectures and FastAPI. " * 20
    pdf_bytes = create_test_pdf(long_text, pages=2)
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("long_document.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["character_count"] > 500
    assert len(data["preview"]) <= 500


def test_upload_empty_file_rejected():
    """Test 400 rejection when uploading a 0-byte file."""
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "empty" in data["detail"].lower()


def test_upload_non_pdf_extension_rejected():
    """Test 400 rejection when uploading a non-PDF file extension."""
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("resume.txt", b"Some plain text resume content", "text/plain")},
    )

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "only pdf files" in data["detail"].lower()


def test_upload_invalid_pdf_content_rejected():
    """Test 400 rejection when file has .pdf extension but lacks PDF magic bytes."""
    fake_pdf = b"NOT A REAL PDF FILE HEADER OR CONTENT"
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("fake_header.pdf", fake_pdf, "application/pdf")},
    )

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "not a valid pdf document" in data["detail"].lower()


def test_upload_corrupted_pdf_rejected():
    """Test 400 rejection when file has %PDF- header but malformed internal structure."""
    corrupted_pdf = b"%PDF-1.4\n\x00\xff\xfe corrupted bytes with no valid xref or trailer"
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("corrupted.pdf", corrupted_pdf, "application/pdf")},
    )

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "corrupted" in data["detail"].lower() or "malformed" in data["detail"].lower()


def test_upload_encrypted_pdf_rejected():
    """Test 400 rejection when uploading a password-protected/encrypted PDF."""
    encrypted_bytes = create_encrypted_pdf("Confidential Candidate Data", "supersecret")
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("encrypted_resume.pdf", encrypted_bytes, "application/pdf")},
    )

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "encrypted" in data["detail"].lower() or "password" in data["detail"].lower()


def test_upload_file_exceeds_size_limit():
    """Test 413 rejection when file exceeds configured MAX_UPLOAD_SIZE_BYTES limit."""
    pdf_bytes = create_test_pdf("Size limit test", pages=1)

    # Configure a custom PDFService with a strict 200-byte upload limit
    strict_service = PDFService(max_upload_size_bytes=200, max_pdf_pages=100)
    app.dependency_overrides[get_pdf_service] = lambda: strict_service
    try:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("too_large.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 413
        data = response.json()
        assert "detail" in data
        assert "exceeds the maximum allowed limit" in data["detail"].lower()
    finally:
        app.dependency_overrides.clear()


def test_upload_pdf_with_no_extractable_text_rejected():
    """Test 400 rejection when PDF has pages but contains no extractable text (e.g., scanned/blank)."""
    blank_bytes = create_blank_pdf(pages=2)
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("scanned_blank.pdf", blank_bytes, "application/pdf")},
    )

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "no extractable text" in data["detail"].lower()


def test_upload_page_count_limit_exceeded():
    """Test 400 rejection when PDF page count exceeds configured limit."""
    pdf_bytes = create_test_pdf("Multi page limit test", pages=3)

    # Configure a custom PDFService with max_pdf_pages=2
    limited_service = PDFService(max_upload_size_bytes=10 * 1024 * 1024, max_pdf_pages=2)
    app.dependency_overrides[get_pdf_service] = lambda: limited_service
    try:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("too_many_pages.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "maximum allowed limit of 2 pages" in data["detail"]
    finally:
        app.dependency_overrides.clear()


def test_filename_sanitization():
    """Test filename sanitization against path traversal, control characters, and hidden file prefixes."""
    pdf_bytes = create_test_pdf("Sanitization test", pages=1)

    # Path traversal attack representation
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("../../etc/passwd.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "passwd.pdf"
    assert ".." not in data["filename"]
    assert "/" not in data["filename"]
    assert "\\" not in data["filename"]

    # Leading dots / hidden file name
    response2 = client.post(
        "/api/v1/documents/upload",
        files={"file": ("...hidden.pdf", pdf_bytes, "application/pdf")},
    )
    assert response2.status_code == 200
    assert response2.json()["filename"] == "hidden.pdf"


def test_root_endpoint_includes_document_upload():
    """Verify root endpoint lists the document upload endpoint URL."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "document_upload" in data
    assert data["document_upload"] == "/api/v1/documents/upload"
