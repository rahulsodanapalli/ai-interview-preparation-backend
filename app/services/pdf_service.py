"""PDF Ingestion Service for secure PDF upload, validation, and text extraction."""

import io
import logging
import os
import re
from typing import List, Optional
import unicodedata
from fastapi import UploadFile
import pypdf
import pypdf.errors

from app.core.config import settings
from app.schemas.document import DocumentUploadResponse

logger = logging.getLogger(__name__)


# ============================================================================
# PDF Processing Exceptions
# ============================================================================


class PDFProcessingError(Exception):
    """Base exception for PDF ingestion and processing errors."""

    pass


class PDFSizeLimitError(PDFProcessingError):
    """Raised when the uploaded file exceeds the maximum allowed size limit."""

    pass


class PDFEmptyFileError(PDFProcessingError):
    """Raised when the uploaded file is zero bytes or has zero pages."""

    pass


class PDFInvalidError(PDFProcessingError):
    """Raised when the file is not a valid or well-formed PDF document."""

    pass


class PDFEncryptedError(PDFProcessingError):
    """Raised when the PDF document is encrypted or password-protected."""

    pass


class PDFPageLimitError(PDFProcessingError):
    """Raised when the PDF document exceeds the maximum permitted page count."""

    pass


class PDFNoTextError(PDFProcessingError):
    """Raised when the PDF document contains pages but no extractable text."""

    pass


# ============================================================================
# PDF Ingestion Service Implementation
# ============================================================================


class PDFService:
    """Service responsible for validating PDF uploads and extracting document text."""

    def __init__(
        self,
        max_upload_size_bytes: int = settings.MAX_UPLOAD_SIZE_BYTES,
        max_pdf_pages: int = settings.MAX_PDF_PAGES,
    ):
        self.max_upload_size_bytes = max_upload_size_bytes
        self.max_pdf_pages = max_pdf_pages

    @staticmethod
    def sanitize_filename(filename: Optional[str]) -> str:
        """Sanitizes user-provided filename to prevent path traversal and injection.

        Security Note: The returned filename should never be directly used as
        a local filesystem path.
        """
        if not filename:
            return "unnamed_document.pdf"

        # Remove null bytes and non-printable control characters
        cleaned = "".join(ch for ch in filename if ch.isprintable() and ch != "\0")

        # Normalize unicode characters
        cleaned = unicodedata.normalize("NFKD", cleaned)

        # Extract only the base name (strip directory paths e.g. ../, /, \\)
        cleaned = os.path.basename(cleaned)
        cleaned = re.sub(r"[/\\]+", "", cleaned)

        # Retain only safe alphanumeric characters, dots, underscores, hyphens, and spaces
        cleaned = re.sub(r"[^a-zA-Z0-9._\- ]", "_", cleaned).strip()

        # Remove leading dots to prevent hidden file representation
        cleaned = cleaned.lstrip(".")
        if not cleaned:
            return "unnamed_document.pdf"

        # Cap length to 255 characters
        if len(cleaned) > 255:
            base, ext = os.path.splitext(cleaned)
            cleaned = base[: 255 - len(ext)] + ext

        return cleaned

    async def read_and_validate_stream(self, file: UploadFile) -> bytes:
        """Reads the uploaded file stream in chunks, enforcing the size limit before loading into memory."""
        total_size = 0
        chunks: List[bytes] = []
        chunk_size = 65536  # 64 KB

        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > self.max_upload_size_bytes:
                max_mb = self.max_upload_size_bytes / (1024 * 1024)
                raise PDFSizeLimitError(
                    f"File size exceeds the maximum allowed limit of {max_mb:.1f} MB."
                )
            chunks.append(chunk)

        if total_size == 0:
            raise PDFEmptyFileError("Uploaded file is empty (0 bytes).")

        return b"".join(chunks)

    def extract_full_text_and_metadata(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> tuple[str, int, int, str]:
        """Validates PDF content structure, extracts text using pypdf, and returns metadata.

        Returns:
            Tuple of (full_text, page_count, character_count, preview).
        """
        # 1. Content validation: Check for standard PDF magic header (%PDF-)
        if b"%PDF-" not in file_bytes[:1024]:
            raise PDFInvalidError(
                "Uploaded file is not a valid PDF document (missing PDF magic header)."
            )

        # 2. Parse PDF structure using pypdf
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes), strict=False)
        except pypdf.errors.EmptyFileError as exc:
            raise PDFEmptyFileError("Uploaded PDF file is empty.") from exc
        except (pypdf.errors.PdfStreamError, pypdf.errors.PdfReadError, Exception) as exc:
            raise PDFInvalidError(f"Corrupted or malformed PDF file: {exc}") from exc

        # 3. Check for encryption / password protection
        if reader.is_encrypted:
            try:
                decrypted = reader.decrypt("")
                if decrypted == 0:
                    raise PDFEncryptedError(
                        "Password-protected or encrypted PDFs are not supported."
                    )
            except Exception as exc:
                raise PDFEncryptedError(
                    "Password-protected or encrypted PDFs are not supported."
                ) from exc

        # 4. Check page count
        try:
            page_count = len(reader.pages)
        except Exception as exc:
            raise PDFInvalidError(f"Unable to read PDF page structure: {exc}") from exc

        if page_count == 0:
            raise PDFEmptyFileError("The uploaded PDF contains zero pages.")

        if page_count > self.max_pdf_pages:
            raise PDFPageLimitError(
                f"PDF exceeds the maximum allowed limit of {self.max_pdf_pages} pages (received {page_count} pages)."
            )

        # 5. Extract text page-by-page safely
        extracted_chunks: List[str] = []
        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    extracted_chunks.append(page_text.strip())
            except Exception as exc:
                logger.warning(
                    "Failed to extract text from page %d of '%s': %s",
                    page_idx,
                    filename,
                    exc,
                )
                continue

        full_text = "\n\n".join(extracted_chunks).strip()
        character_count = len(full_text)

        if character_count == 0:
            raise PDFNoTextError(
                "The uploaded PDF contains no extractable text. Scanned or image-only documents are not supported."
            )

        preview = full_text[:500]

        logger.info(
            "Successfully extracted text from document '%s': %d pages, %d characters.",
            filename,
            page_count,
            character_count,
        )

        return full_text, page_count, character_count, preview

    def extract_text_from_bytes(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> DocumentUploadResponse:
        """Validates PDF content structure and extracts text returning DocumentUploadResponse."""
        full_text, page_count, character_count, preview = self.extract_full_text_and_metadata(
            file_bytes, filename
        )
        return DocumentUploadResponse(
            success=True,
            filename=filename,
            page_count=page_count,
            character_count=character_count,
            preview=preview,
            message=f"Successfully extracted {character_count} characters across {page_count} page(s).",
        )

    async def extract_text_and_metadata(
        self,
        file: UploadFile,
    ) -> tuple[str, str, int, int, str]:
        """Reads upload stream and returns (filename, full_text, page_count, character_count, preview)."""
        sanitized_name = self.sanitize_filename(file.filename)
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise PDFInvalidError("Only PDF files (.pdf) are supported.")

        file_bytes = await self.read_and_validate_stream(file)
        full_text, page_count, character_count, preview = self.extract_full_text_and_metadata(
            file_bytes, sanitized_name
        )
        return sanitized_name, full_text, page_count, character_count, preview

    async def process_upload(self, file: UploadFile) -> DocumentUploadResponse:
        """Coordinates reading stream, validating format, and extracting text."""
        sanitized_name, _, page_count, character_count, preview = await self.extract_text_and_metadata(file)
        return DocumentUploadResponse(
            success=True,
            filename=sanitized_name,
            page_count=page_count,
            character_count=character_count,
            preview=preview,
            message=f"Successfully extracted {character_count} characters across {page_count} page(s).",
        )


def get_pdf_service() -> PDFService:
    """FastAPI dependency provider for PDFService."""
    return PDFService()
