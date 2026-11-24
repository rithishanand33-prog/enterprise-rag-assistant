
from langchain_core.documents import Document
from src.rag import answer_question


class FakeVectorStore:

    def __init__(self, documents):
        self.documents = documents

    def similarity_search(self, question, k=3):
        return self.documents[:k]


class FakeLLM:

    def __init__(self, response):
        self.response = response
        self.last_prompt = ""

    def invoke(self, prompt):
        self.last_prompt = prompt
        return type(
            "FakeResponse",
            (),
            {"content": self.response}
        )()


def test_answer_contains_source_citation():

    documents = [
        Document(
            page_content="Employees receive 12 days of annual leave.",
            metadata={
                "source": "employee_handbook.pdf",
                "page": 1
            }
        )
    ]

    vectorstore = FakeVectorStore(documents)

    llm = FakeLLM("Employees receive 12 days of annual leave.")

    answer = answer_question(
        "How many annual leave days do employees receive?",
        vectorstore,
        llm
    )

    assert "12 days" in answer
    assert "employee_handbook.pdf" in answer
    assert "Page 2" in answer


def test_empty_retrieval_returns_message():

    vectorstore = FakeVectorStore([])
    llm = FakeLLM("This should not be called.")

    answer = answer_question(
        "What is the company policy?",
        vectorstore,
        llm
    )

    assert "could not find relevant information" in answer.lower()


def test_prompt_contains_grounding_instructions():

    documents = [
        Document(
            page_content="Employees receive 12 days of annual leave.",
            metadata={"source": "handbook.pdf", "page": 1}
        )
    ]

    vectorstore = FakeVectorStore(documents)
    llm = FakeLLM("Employees receive 12 days of annual leave.")

    answer_question(
        "How many annual leave days are available?",
        vectorstore,
        llm
    )

    assert "ONLY" in llm.last_prompt
    assert "Do not invent" in llm.last_prompt
