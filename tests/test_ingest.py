
from pathlib import Path

import pytest

from langchain_core.documents import Document

from src import ingest


class FakeUploadedFile:
    def __init__(self, name, content=b"fake pdf content"):
        self.name = name
        self.content = content

    def getvalue(self):
        return self.content


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

    monkeypatch.setattr(
        ingest,
        "PyPDFLoader",
        FakePDFLoader
    )

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

    monkeypatch.setattr(
        ingest,
        "PyPDFLoader",
        FakePDFLoader
    )

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

    monkeypatch.setattr(
        ingest,
        "PyPDFLoader",
        FakePDFLoader
    )

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

    monkeypatch.setattr(
        ingest,
        "PyPDFLoader",
        FakePDFLoader
    )

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

    monkeypatch.setattr(
        ingest,
        "PyPDFLoader",
        FakePDFLoader
    )

    documents = ingest.load_pdf("custom_document.pdf")

    assert captured_paths == ["custom_document.pdf"]
    assert documents == expected_documents
