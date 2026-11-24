
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingest import load_pdf


def create_chunks(documents=None):

    print("Loading documents...")

    if documents is None:
        documents = load_pdf()

    print("\nSplitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)

    print(f"\nSuccessfully created {len(chunks)} chunks!")

    return chunks


if __name__ == "__main__":
    create_chunks()
