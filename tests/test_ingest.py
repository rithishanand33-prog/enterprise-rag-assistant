
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src import ingest


class FakeUploadedFile:
    def __init__(
        self,
        name,
        content=b"%PDF-1.4\nfake pdf content"
    ):
        self.name = name
        self.content = content

    def getvalue(self):
        return self.content


class FakePDFDocument:
    """Mock PyMuPDF document context manager."""

    def __init__(self):
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True

    def load_page(self, page_index):
        return object()


# --------------------------------------------------
# Existing PDF upload tests
# --------------------------------------------------

def test_load_uploaded_pdf_returns_all_pages(monkeypatch):
    expected_documents = [
        Document(page_content="Page one"),
        Document(page_content="Page two"),
    ]

    class FakePDFLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return expected_documents

    monkeypatch.setattr(ingest, "PyPDFLoader", FakePDFLoader)

    uploaded_file = FakeUploadedFile("handbook.pdf")
    documents = ingest.load_uploaded_pdf(uploaded_file)

    assert len(documents) == 2
    assert documents[0].page_content == "Page one"
    assert documents[1].page_content == "Page two"


def test_uploaded_pdf_source_metadata_is_updated(monkeypatch):
    expected_documents = [
        Document(
            page_content="Employee leave policy",
            metadata={"source": "temporary.pdf"}
        )
    ]

    class FakePDFLoader:
        def __init__(self, path):
            pass

        def load(self):
            return expected_documents

    monkeypatch.setattr(ingest, "PyPDFLoader", FakePDFLoader)

    uploaded_file = FakeUploadedFile("company_handbook.pdf")
    documents = ingest.load_uploaded_pdf(uploaded_file)

    assert documents[0].metadata["source"] == "company_handbook.pdf"


def test_temporary_file_is_deleted_after_success(monkeypatch):
    captured_paths = []

    class FakePDFLoader:
        def __init__(self, path):
            captured_paths.append(path)

        def load(self):
            return [Document(page_content="Test content")]

    monkeypatch.setattr(ingest, "PyPDFLoader", FakePDFLoader)

    uploaded_file = FakeUploadedFile("handbook.pdf")
    ingest.load_uploaded_pdf(uploaded_file)

    assert len(captured_paths) == 1
    assert not Path(captured_paths[0]).exists()


def test_temporary_file_is_deleted_after_failure(monkeypatch):
    captured_paths = []

    class FakePDFLoader:
        def __init__(self, path):
            captured_paths.append(path)

        def load(self):
            raise ValueError("Invalid PDF document")

    monkeypatch.setattr(ingest, "PyPDFLoader", FakePDFLoader)

    uploaded_file = FakeUploadedFile("invalid.pdf")

    with pytest.raises(ValueError, match="Invalid PDF document"):
        ingest.load_uploaded_pdf(uploaded_file)

    assert len(captured_paths) == 1
    assert not Path(captured_paths[0]).exists()


def test_load_pdf_uses_provided_path(monkeypatch):
    expected_documents = [
        Document(page_content="Sample PDF page")
    ]

    captured_paths = []

    class FakePDFLoader:
        def __init__(self, path):
            captured_paths.append(path)

        def load(self):
            return expected_documents

    monkeypatch.setattr(ingest, "PyPDFLoader", FakePDFLoader)

    documents = ingest.load_pdf("custom_document.pdf")

    assert captured_paths == ["custom_document.pdf"]
    assert documents == expected_documents


def test_rejects_non_pdf_filename():
    uploaded_file = FakeUploadedFile("document.txt")

    with pytest.raises(ValueError, match="Only PDF files"):
        ingest.load_uploaded_pdf(uploaded_file)


def test_rejects_empty_file():
    uploaded_file = FakeUploadedFile("empty.pdf", content=b"")

    with pytest.raises(ValueError, match="uploaded file is empty"):
        ingest.load_uploaded_pdf(uploaded_file)


def test_rejects_file_without_pdf_signature():
    uploaded_file = FakeUploadedFile(
        "invalid.pdf",
        content=b"This is not a PDF"
    )

    with pytest.raises(ValueError, match="Invalid PDF file"):
        ingest.load_uploaded_pdf(uploaded_file)


def test_rejects_file_exceeding_size_limit():
    uploaded_file = FakeUploadedFile(
        "large.pdf",
        content=b"%PDF-" + b"A" * ingest.MAX_FILE_SIZE
    )

    with pytest.raises(ValueError, match="10 MB upload limit"):
        ingest.load_uploaded_pdf(uploaded_file)


# --------------------------------------------------
# New OCR tests
# --------------------------------------------------

def test_extract_page_text_with_ocr(monkeypatch):
    """Verify that a rendered PDF page is passed to Tesseract."""

    class FakePixmap:
        def tobytes(self, image_format):
            assert image_format == "png"
            return b"fake image bytes"

    class FakePage:
        def get_pixmap(self, matrix, alpha):
            assert alpha is False
            return FakePixmap()

    class FakePDFDocument:
        def load_page(self, page_index):
            assert page_index == 0
            return FakePage()

    fake_image = object()

    monkeypatch.setattr(
        ingest.Image,
        "open",
        lambda image_bytes: fake_image
    )

    def fake_ocr(image, lang):
        assert image is fake_image
        assert lang == "eng"
        return "  Extracted employee policy  "

    monkeypatch.setattr(
        ingest.pytesseract,
        "image_to_string",
        fake_ocr
    )

    result = ingest.extract_page_text_with_ocr(
        FakePDFDocument(),
        0
    )

    assert result == "Extracted employee policy"


def test_scanned_pdf_page_is_detected_and_processed(monkeypatch):
    """Verify that blank PDF pages trigger OCR."""

    expected_documents = [
        Document(page_content="")
    ]

    fake_pdf_document = FakePDFDocument()

    class FakePDFLoader:
        def __init__(self, path):
            pass

        def load(self):
            return expected_documents

    monkeypatch.setattr(ingest, "PyPDFLoader", FakePDFLoader)
    monkeypatch.setattr(
        ingest.pymupdf,
        "open",
        lambda path: fake_pdf_document
    )

    ocr_calls = []

    def fake_ocr(pdf_document, page_index):
        ocr_calls.append(page_index)
        return "Scanned employee leave policy"

    monkeypatch.setattr(
        ingest,
        "extract_page_text_with_ocr",
        fake_ocr
    )

    documents = ingest.load_pdf("scanned.pdf")

    assert documents[0].page_content == (
        "Scanned employee leave policy"
    )
    assert ocr_calls == [0]
    assert fake_pdf_document.closed is True


def test_normal_pdf_does_not_trigger_ocr(monkeypatch):
    """Verify that text-based PDFs bypass OCR."""

    expected_documents = [
        Document(page_content="Normal employee handbook text")
    ]

    class FakePDFLoader:
        def __init__(self, path):
            pass

        def load(self):
            return expected_documents

    def unexpected_ocr(*args, **kwargs):
        pytest.fail("OCR should not run for normal text PDFs")

    def unexpected_pdf_open(*args, **kwargs):
        pytest.fail("PyMuPDF should not open normal text PDFs")

    monkeypatch.setattr(ingest, "PyPDFLoader", FakePDFLoader)
    monkeypatch.setattr(
        ingest,
        "extract_page_text_with_ocr",
        unexpected_ocr
    )
    monkeypatch.setattr(
        ingest.pymupdf,
        "open",
        unexpected_pdf_open
    )

    documents = ingest.load_pdf("normal.pdf")

    assert len(documents) == 1
    assert documents[0].page_content == (
        "Normal employee handbook text"
    )


def test_mixed_pdf_processes_only_scanned_pages(monkeypatch):
    """Verify that OCR processes blank pages but preserves normal text."""

    expected_documents = [
        Document(page_content="Normal text on page one"),
        Document(page_content="")
    ]

    fake_pdf_document = FakePDFDocument()

    class FakePDFLoader:
        def __init__(self, path):
            pass

        def load(self):
            return expected_documents

    monkeypatch.setattr(ingest, "PyPDFLoader", FakePDFLoader)
    monkeypatch.setattr(
        ingest.pymupdf,
        "open",
        lambda path: fake_pdf_document
    )

    ocr_calls = []

    def fake_ocr(pdf_document, page_index):
        ocr_calls.append(page_index)
        return "OCR text from scanned page two"

    monkeypatch.setattr(
        ingest,
        "extract_page_text_with_ocr",
        fake_ocr
    )

    documents = ingest.load_pdf("mixed.pdf")

    assert documents[0].page_content == (
        "Normal text on page one"
    )
    assert documents[1].page_content == (
        "OCR text from scanned page two"
    )

    # Only the second page should have triggered OCR.
    assert ocr_calls == [1]
    assert fake_pdf_document.closed is True
