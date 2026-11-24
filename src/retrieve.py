
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def load_vectorstore():
    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Loading FAISS vector database...")

    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def retrieve_documents(vectorstore, question, k=3):
    print("\nSearching for relevant information...")

    results = vectorstore.similarity_search(
        question,
        k=k
    )

    return results


if __name__ == "__main__":

    db = load_vectorstore()

    question = "How many paid annual leave days do employees receive?"

    results = retrieve_documents(db, question)

    print("\nQuestion:", question)
    print("\nRelevant information:")

    for i, doc in enumerate(results, start=1):

        page = doc.metadata.get("page")
        page_label = page + 1 if isinstance(page, int) else "unknown"

        print(f"\n--- Result {i} | Page {page_label} ---")
        print(doc.page_content)
