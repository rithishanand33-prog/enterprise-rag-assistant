
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def load_vectorstore():

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("\nLoading FAISS vector database...")

    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("\nFAISS vector database loaded successfully!")

    return vectorstore


def retrieve_documents(question):

    vectorstore = load_vectorstore()

    print("\nSearching for relevant documents...")

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    results = retriever.invoke(question)

    print(f"\nFound {len(results)} relevant chunks!")

    for i, document in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(document.page_content)

    return results


if __name__ == "__main__":

    question = "How many annual leave days do employees receive?"

    retrieve_documents(question)
