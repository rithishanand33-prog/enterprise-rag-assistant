
from src.retriever import load_vectorstore
from src.llm import get_llm


def answer_question(question, vectorstore, llm):

    print("\nRetrieving relevant documents...")

    documents = vectorstore.similarity_search(question, k=3)

    if not documents:
        return "I could not find relevant information in the uploaded documents."

    # Prepare context with document source information
    context_parts = []

    for document in documents:
        source = document.metadata.get("source", "Unknown document")
        page = document.metadata.get("page")

        if isinstance(page, int):
            page_number = page + 1
            source_info = f"Source: {source}, Page: {page_number}"
        else:
            source_info = f"Source: {source}"

        context_parts.append(
            f"{source_info}\n{document.page_content}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a reliable Enterprise AI Assistant.

Your task is to answer questions using ONLY the provided document context.

STRICT RULES:

1. Use only facts explicitly supported by the context.
2. Do not use outside knowledge or make assumptions.
3. Do not invent company policies, dates, numbers, or procedures.
4. If the context does not contain enough information to answer,
   respond exactly:
   "I could not find this information in the uploaded documents."
5. If only part of the question can be answered, clearly explain
   which information is available and which is missing.
6. Keep answers clear, professional, and concise.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    print("Generating grounded answer using Gemini...")

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, str):
        answer = content.strip()

    elif isinstance(content, list):
        answer = "\n".join(
            item.get("text", "")
            if isinstance(item, dict)
            else getattr(item, "text", "")
            for item in content
        ).strip()

    else:
        answer = str(content).strip()

    # Do not show citations when the assistant abstains
    if "I could not find this information in the uploaded documents." in answer:
        return answer

    # Collect source references
    sources = set()

    for document in documents:
        source = document.metadata.get("source", "Unknown document")
        page = document.metadata.get("page")

        if isinstance(page, int):
            sources.add(f"{source}, Page {page + 1}")
        else:
            sources.add(source)

    source_text = "\n".join(
        f"- {source}" for source in sorted(sources)
    )

    return f"{answer}\n\n**Sources:**\n{source_text}"


if __name__ == "__main__":

    print("\n🤖 Enterprise RAG Assistant")

    print("\nLoading AI components...")

    vectorstore = load_vectorstore()
    llm = get_llm()

    print("\nAssistant is ready!")
    print("Type 'exit' to quit.")

    while True:

        question = input("\nAsk your question: ").strip()

        if question.lower() == "exit":
            print("\nGoodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        answer = answer_question(
            question,
            vectorstore,
            llm
        )

        print("\n========== FINAL ANSWER ==========")
        print(answer)
