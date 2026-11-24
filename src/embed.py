from langchain_community.embeddings import HuggingFaceEmbeddings
from src.chunk import create_chunks


def generate_embeddings():

    print("Loading text chunks...")

    chunks = create_chunks()

    print("\nLoading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("\nGenerating embeddings...")

    vectors = embeddings.embed_documents(
        [chunk.page_content for chunk in chunks]
    )

    print(f"\nSuccessfully generated {len(vectors)} embeddings!")

    print("\nEmbedding dimensions:", len(vectors[0]))

    return vectors


if __name__ == "__main__":
    generate_embeddings()