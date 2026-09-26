"""
document_loader.py
-------------------
Handles PDF upload validation and text extraction.
Each extracted page becomes a dict with the page text plus
metadata (source document name + page number) so answers can
later be traced back to where they came from.
"""

from pypdf import PdfReader

MAX_FILE_SIZE_MB = 25
ALLOWED_EXTENSION = ".pdf"


def validate_file(uploaded_file):
    """
    Validate an uploaded file's type and size.
    Returns (is_valid: bool, error_message: str or None)
    """
    name = uploaded_file.name.lower()
    if not name.endswith(ALLOWED_EXTENSION):
        return False, f"'{uploaded_file.name}' is not a PDF file."

    # Streamlit's UploadedFile exposes .size in bytes
    size_mb = getattr(uploaded_file, "size", 0) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"'{uploaded_file.name}' is {size_mb:.1f} MB, over the {MAX_FILE_SIZE_MB} MB limit."

    return True, None


def extract_pages(uploaded_file):
    """
    Extract text from every page of a single uploaded PDF.

    Returns a list of dicts:
        {
            "text": "<page text>",
            "source": "<file name>",
            "page": <1-indexed page number>
        }
    Empty pages are skipped safely.
    """
    pages = []
    reader = PdfReader(uploaded_file)

    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            # Corrupt or unreadable page - skip rather than crash the app
            text = ""

        text = text.strip()
        if not text:
            # Empty / image-only page. OCR could be added here as an
            # optional extension for scanned documents.
            continue

        pages.append({
            "text": text,
            "source": uploaded_file.name,
            "page": i,
        })

    return pages


def extract_all(uploaded_files):
    """
    Run extract_pages over a list of uploaded files and combine results.
    Returns a single flat list of page dicts across all documents.
    """
    all_pages = []
    for f in uploaded_files:
        all_pages.extend(extract_pages(f))
    return all_pages
