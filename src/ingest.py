
from pathlib import Path
import io
import tempfile

import pymupdf
import pytesseract
from PIL import Image

from langchain_community.document_loaders import PyPDFLoader


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def extract_page_text_with_ocr(pdf_document, page_index):
    """Extract text from a scanned PDF page using OCR."""

    page = pdf_document.load_page(page_index)

    # Render the PDF page as an image.
    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(3, 3),
        alpha=False
    )

    # Convert the image into a format Pillow can read.
    image_bytes = pixmap.tobytes("png")
    image = Image.open(io.BytesIO(image_bytes))

    # Extract text using Tesseract.
    extracted_text = pytesseract.image_to_string(
        image,
        lang="eng"
    )

    return extracted_text.strip()


def load_pdf(pdf_path="sample_employee_handbook.pdf"):
    """Load PDF pages and use OCR for pages without extractable text."""

    print("Loading PDF document...")

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    print(f"\nSuccessfully loaded {len(documents)} pages!")

    # Identify pages that need OCR.
    pages_needing_ocr = [
        index
        for index, document in enumerate(documents)
        if not document.page_content.strip()
    ]

    if pages_needing_ocr:
        print(
            f"\nOCR required for {len(pages_needing_ocr)} "
            "scanned or image-only page(s)."
        )

        with pymupdf.open(str(pdf_path)) as pdf_document:

            for page_index in pages_needing_ocr:
                print(f"Running OCR on page {page_index + 1}...")

                extracted_text = extract_page_text_with_ocr(
                    pdf_document,
                    page_index
                )

                documents[page_index].page_content = extracted_text

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
    """Validate and load an uploaded PDF safely, including scanned pages."""

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
                "The PDF contains no extractable text, "
                "even after OCR processing."
            )

        for document in documents:
            document.metadata["source"] = uploaded_file.name

        return documents

    finally:
        Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    load_pdf()
