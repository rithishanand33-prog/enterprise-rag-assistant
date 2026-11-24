
from unittest.mock import Mock

from langchain_core.documents import Document

from src.rag import (
    ABSTAIN_MESSAGE,
    answer_question,
    handle_conversational_message,
)


def create_document(
    content="Employees receive 12 days of annual leave.",
    source="employee_handbook.pdf",
    page=1,
):
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "page": page,
        },
    )


def create_vectorstore(results):
    vectorstore = Mock()
    vectorstore.similarity_search_with_score.return_value = results
    return vectorstore


def create_llm(response="Employees receive 12 days of annual leave."):
    llm = Mock()
    llm.invoke.return_value = Mock(content=response)
    return llm


def test_answer_contains_source_citation():
    document = create_document()
    vectorstore = create_vectorstore([(document, 0.5)])
    llm = create_llm()

    answer = answer_question(
        "How many annual leave days do employees receive?",
        vectorstore,
        llm,
    )

    assert "12 days" in answer
    assert "employee_handbook.pdf" in answer
    assert "Page 2" in answer


def test_empty_retrieval_returns_message():
    vectorstore = create_vectorstore([])
    llm = create_llm()

    answer = answer_question(
        "What is the company policy?",
        vectorstore,
        llm,
    )

    assert answer == ABSTAIN_MESSAGE
    llm.invoke.assert_not_called()


def test_prompt_contains_grounding_instructions():
    document = create_document()
    vectorstore = create_vectorstore([(document, 0.5)])
    llm = create_llm()

    answer_question(
        "How many annual leave days do employees receive?",
        vectorstore,
        llm,
    )

    prompt = llm.invoke.call_args[0][0]

    assert "ONLY the supplied document context" in prompt
    assert "Never use outside knowledge or assumptions" in prompt
    assert ABSTAIN_MESSAGE in prompt


def test_distance_threshold_rejects_irrelevant_documents():
    document = create_document()
    vectorstore = create_vectorstore([(document, 2.0)])
    llm = create_llm()

    answer = answer_question(
        "What is the company revenue?",
        vectorstore,
        llm,
    )

    assert answer == ABSTAIN_MESSAGE
    llm.invoke.assert_not_called()


def test_multiple_documents_are_used_as_context():
    document_one = create_document(
        content="Annual leave is 12 days.",
        source="leave_policy.pdf",
        page=1,
    )

    document_two = create_document(
        content="Sick leave is 8 days.",
        source="sick_policy.pdf",
        page=2,
    )

    vectorstore = create_vectorstore([
        (document_one, 0.5),
        (document_two, 0.7),
    ])

    llm = create_llm("Annual leave is 12 days and sick leave is 8 days.")

    answer_question(
        "What are the leave policies?",
        vectorstore,
        llm,
    )

    prompt = llm.invoke.call_args[0][0]

    assert "Annual leave is 12 days." in prompt
    assert "Sick leave is 8 days." in prompt


def test_missing_source_metadata_uses_unknown_document():
    document = Document(
        page_content="Employees receive annual leave.",
        metadata={},
    )

    vectorstore = create_vectorstore([(document, 0.5)])
    llm = create_llm("Employees receive annual leave.")

    answer = answer_question(
        "What leave do employees receive?",
        vectorstore,
        llm,
    )

    assert "Unknown document" in answer


def test_empty_llm_response_returns_abstention():
    document = create_document()
    vectorstore = create_vectorstore([(document, 0.5)])
    llm = create_llm("")

    answer = answer_question(
        "How many annual leave days do employees receive?",
        vectorstore,
        llm,
    )

    assert answer == ABSTAIN_MESSAGE


def test_llm_abstention_does_not_include_sources():
    document = create_document()
    vectorstore = create_vectorstore([(document, 0.5)])
    llm = create_llm(ABSTAIN_MESSAGE)

    answer = answer_question(
        "What is the company revenue?",
        vectorstore,
        llm,
    )

    assert answer == ABSTAIN_MESSAGE
    assert "Sources:" not in answer


def test_structured_llm_response_is_supported():
    document = create_document()
    vectorstore = create_vectorstore([(document, 0.5)])

    llm = create_llm([
        {"type": "text", "text": "Employees receive 12 days."}
    ])

    answer = answer_question(
        "How many annual leave days do employees receive?",
        vectorstore,
        llm,
    )

    assert "Employees receive 12 days." in answer


def test_document_at_exact_threshold_is_accepted():
    document = create_document()
    vectorstore = create_vectorstore([(document, 1.48)])
    llm = create_llm()

    answer = answer_question(
        "How many annual leave days do employees receive?",
        vectorstore,
        llm,
    )

    assert "12 days" in answer
    llm.invoke.assert_called_once()


def test_document_above_threshold_is_rejected():
    document = create_document()
    vectorstore = create_vectorstore([(document, 1.49)])
    llm = create_llm()

    answer = answer_question(
        "What is the company revenue?",
        vectorstore,
        llm,
    )

    assert answer == ABSTAIN_MESSAGE
    llm.invoke.assert_not_called()


# --------------------------------------------------
# Conversational handling tests
# --------------------------------------------------

def test_greeting_returns_friendly_response():
    response = handle_conversational_message("Hi")

    assert response is not None
    assert "Hello" in response


def test_thank_you_returns_response():
    response = handle_conversational_message("Thank you")

    assert response is not None
    assert "welcome" in response.lower()


def test_capabilities_question_returns_description():
    response = handle_conversational_message("What can you do?")

    assert response is not None
    assert "uploaded company PDFs" in response


def test_document_question_is_not_handled_as_conversation():
    response = handle_conversational_message(
        "How many annual leave days do employees receive?"
    )

    assert response is None


def test_goodbye_returns_response():
    response = handle_conversational_message("Goodbye")

    assert response is not None
    assert "Goodbye" in response
