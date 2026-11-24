
from langchain_core.documents import Document

from src.rag import answer_question, ABSTAIN_MESSAGE


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents

    def similarity_search_with_score(self, question, k=3):
        return [
            (document, 0.5)
            for document in self.documents[:k]
        ]


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.last_prompt = ""
        self.called = False

    def invoke(self, prompt):
        self.called = True
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

    llm = FakeLLM(
        "Employees receive 12 days of annual leave."
    )

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

    llm = FakeLLM(
        "This should not be called."
    )

    answer = answer_question(
        "What is the company policy?",
        vectorstore,
        llm
    )

    assert ABSTAIN_MESSAGE in answer
    assert llm.called is False


def test_prompt_contains_grounding_instructions():

    documents = [
        Document(
            page_content="Employees receive 12 days of annual leave.",
            metadata={
                "source": "handbook.pdf",
                "page": 1
            }
        )
    ]

    vectorstore = FakeVectorStore(documents)

    llm = FakeLLM(
        "Employees receive 12 days of annual leave."
    )

    answer_question(
        "How many annual leave days are available?",
        vectorstore,
        llm
    )

    assert "ONLY" in llm.last_prompt
    assert "Never invent" in llm.last_prompt
    assert "outside knowledge" in llm.last_prompt.lower()


def test_distance_threshold_rejects_irrelevant_documents():

    documents = [
        Document(
            page_content="Some unrelated handbook content.",
            metadata={
                "source": "handbook.pdf",
                "page": 1
            }
        )
    ]

    vectorstore = FakeVectorStore(documents)

    llm = FakeLLM(
        "This should not be called."
    )

    answer = answer_question(
        "What is the company revenue?",
        vectorstore,
        llm,
        max_distance=0.1
    )

    assert ABSTAIN_MESSAGE in answer
    assert llm.called is False
