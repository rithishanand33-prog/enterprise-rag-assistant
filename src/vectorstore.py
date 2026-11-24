
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.chunk import create_chunks


def create_vectorstore(chunks=None):

    if chunks is None:
        print("Loading text chunks...")
        chunks = create_chunks()

    if not chunks:
        raise ValueError("No document chunks available.")

    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("\nCreating FAISS vector database...")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    print("\nFAISS vector database created successfully!")

    return vectorstore


if __name__ == "__main__":
    create_vectorstore()
