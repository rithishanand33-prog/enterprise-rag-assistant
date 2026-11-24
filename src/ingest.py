
from pathlib import Path
import tempfile

from langchain_community.document_loaders import PyPDFLoader


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def load_pdf(pdf_path="sample_employee_handbook.pdf"):
    """Load pages from a PDF file."""

    print("Loading PDF document...")

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    print(f"\nSuccessfully loaded {len(documents)} pages!")

    return documents


def validate_pdf_upload(uploaded_file):
    """Validate an uploaded PDF before processing."""

    if not uploaded_file.name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")

    file_content = uploaded_file.getvalue()

    if not file_content:
        raise ValueError("The uploaded file is empty.")

    if len(file_content) > MAX_FILE_SIZE:
        raise ValueError("PDF file exceeds the 10 MB upload limit.")

    if not file_content.startswith(b"%PDF-"):
        raise ValueError("Invalid PDF file. Please upload a valid PDF.")

    return file_content


def load_uploaded_pdf(uploaded_file):
    """Validate and load an uploaded PDF safely."""

    print(f"Loading uploaded PDF: {uploaded_file.name}")

    file_content = validate_pdf_upload(uploaded_file)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(file_content)
        temp_path = temp_file.name

    try:
        documents = load_pdf(temp_path)

        if not documents:
            raise ValueError("The PDF contains no pages.")

        if not any(
            document.page_content.strip()
            for document in documents
        ):
            raise ValueError(
                "The PDF contains no extractable text."
            )

        for document in documents:
            document.metadata["source"] = uploaded_file.name

        return documents

    finally:
        Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    load_pdf()
