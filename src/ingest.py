
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from pathlib import Path
import tempfile


def load_pdf(pdf_path="sample_employee_handbook.pdf"):

    print("Loading PDF document...")

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    print(f"\nSuccessfully loaded {len(documents)} pages!")

    return documents


def load_uploaded_pdf(uploaded_file):

    print(f"Loading uploaded PDF: {uploaded_file.name}")

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    try:
        documents = load_pdf(temp_path)

        for document in documents:
            document.metadata["source"] = uploaded_file.name

        return documents

    finally:
        Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    load_pdf()
