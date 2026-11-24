
from langchain_core.documents import Document

from src.rag import answer_question, ABSTAIN_MESSAGE


class FakeVectorStore:
    def __init__(self, documents, distances=None):
        self.documents = documents
        self.distances = distances or [0.5] * len(documents)

    def similarity_search_with_score(self, question, k=3):
        return [
            (document, distance)
            for document, distance in zip(
                self.documents[:k],
                self.distances[:k]
            )
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


def make_document(
    content="Employees receive 12 days of annual leave.",
    source="handbook.pdf",
    page=1
):
    metadata = {}

    if source is not None:
        metadata["source"] = source

    if page is not None:
        metadata["page"] = page

    return Document(
        page_content=content,
        metadata=metadata
    )


def test_answer_contains_source_citation():
    document = make_document(
        source="employee_handbook.pdf",
        page=1
    )

    vectorstore = FakeVectorStore([document])
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

    assert answer == ABSTAIN_MESSAGE
    assert llm.called is False


def test_prompt_contains_grounding_instructions():
    document = make_document()

    vectorstore = FakeVectorStore([document])
    llm = FakeLLM("Employees receive 12 days of annual leave.")

    answer_question(
        "How many annual leave days are available?",
        vectorstore,
        llm
    )

    assert "ONLY" in llm.last_prompt
    assert "Never invent" in llm.last_prompt
    assert "outside knowledge" in llm.last_prompt.lower()
    assert "untrusted reference data" in llm.last_prompt


def test_distance_threshold_rejects_irrelevant_documents():
    document = make_document()

    vectorstore = FakeVectorStore(
        [document],
        distances=[0.5]
    )

    llm = FakeLLM("This should not be called.")

    answer = answer_question(
        "What is the company revenue?",
        vectorstore,
        llm,
        max_distance=0.1
    )

    assert answer == ABSTAIN_MESSAGE
    assert llm.called is False


def test_multiple_documents_are_used_as_context():
    documents = [
        make_document(
            content="Employees receive 12 days of annual leave.",
            source="leave.pdf",
            page=1
        ),
        make_document(
            content="Employees receive 8 days of paid sick leave.",
            source="sick_leave.pdf",
            page=2
        )
    ]

    vectorstore = FakeVectorStore(
        documents,
        distances=[0.4, 0.7]
    )

    llm = FakeLLM("The documents describe annual and sick leave.")

    answer = answer_question(
        "Tell me about employee leave.",
        vectorstore,
        llm
    )

    assert "12 days" in llm.last_prompt
    assert "8 days" in llm.last_prompt
    assert "leave.pdf" in answer
    assert "sick_leave.pdf" in answer


def test_missing_source_metadata_uses_unknown_document():
    document = make_document(
        source=None,
        page=None
    )

    vectorstore = FakeVectorStore([document])
    llm = FakeLLM("Employees receive 12 days of annual leave.")

    answer = answer_question(
        "How many annual leave days are available?",
        vectorstore,
        llm
    )

    assert "Unknown document" in answer
    assert "Sources:" in answer


def test_empty_llm_response_returns_abstention():
    document = make_document()

    vectorstore = FakeVectorStore([document])
    llm = FakeLLM("   ")

    answer = answer_question(
        "How many annual leave days are available?",
        vectorstore,
        llm
    )

    assert answer == ABSTAIN_MESSAGE


def test_llm_abstention_does_not_include_sources():
    document = make_document()

    vectorstore = FakeVectorStore([document])
    llm = FakeLLM(ABSTAIN_MESSAGE)

    answer = answer_question(
        "What is the company revenue?",
        vectorstore,
        llm
    )

    assert answer == ABSTAIN_MESSAGE
    assert "Sources:" not in answer


def test_structured_llm_response_is_supported():
    document = make_document()

    vectorstore = FakeVectorStore([document])

    llm = FakeLLM([
        {"text": "Employees receive "},
        {"text": "12 days of annual leave."}
    ])

    answer = answer_question(
        "How many annual leave days are available?",
        vectorstore,
        llm
    )

    assert "12 days" in answer
    assert "Sources:" in answer


def test_document_at_exact_threshold_is_accepted():
    document = make_document()

    vectorstore = FakeVectorStore(
        [document],
        distances=[1.48]
    )

    llm = FakeLLM("Employees receive 12 days of annual leave.")

    answer = answer_question(
        "How many annual leave days are available?",
        vectorstore,
        llm,
        max_distance=1.48
    )

    assert llm.called is True
    assert "12 days" in answer


def test_document_above_threshold_is_rejected():
    document = make_document()

    vectorstore = FakeVectorStore(
        [document],
        distances=[1.49]
    )

    llm = FakeLLM("This should not be called.")

    answer = answer_question(
        "How many annual leave days are available?",
        vectorstore,
        llm,
        max_distance=1.48
    )

    assert answer == ABSTAIN_MESSAGE
    assert llm.called is False
