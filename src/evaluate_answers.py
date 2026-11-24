
import json
from pathlib import Path

from src.chunk import create_chunks
from src.vectorstore import create_vectorstore
from src.llm import get_llm
from src.rag import answer_question


def load_evaluation_dataset():

    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "evaluation_dataset.json"

    with open(dataset_path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_answers():

    print("\n========== RAG ANSWER QUALITY EVALUATION ==========")

    dataset = load_evaluation_dataset()

    print("\nLoading documents...")

    chunks = create_chunks()

    print("\nCreating vector database...")

    vectorstore = create_vectorstore(chunks)

    print("\nLoading Gemini model...")

    llm = get_llm()

    total_tests = len(dataset)
    passed_tests = 0
    total_keyword_coverage = 0

    for index, item in enumerate(dataset, start=1):

        question = item["question"]
        expected_keywords = item["expected_keywords"]

        print(f"\n{'=' * 60}")
        print(f"Test {index}: {question}")

        answer = answer_question(
            question,
            vectorstore,
            llm
        )

        print("\nGenerated Answer:")
        print(answer)

        answer_lower = answer.lower()

        matched_keywords = [
            keyword
            for keyword in expected_keywords
            if keyword.lower() in answer_lower
        ]

        keyword_coverage = (
            len(matched_keywords) / len(expected_keywords)
            if expected_keywords else 0
        )

        total_keyword_coverage += keyword_coverage

        print("\nExpected Keywords:", expected_keywords)
        print("Matched Keywords:", matched_keywords)

        print(
            f"Answer Keyword Coverage: "
            f"{len(matched_keywords)}/{len(expected_keywords)}"
        )

        if keyword_coverage == 1:
            passed_tests += 1
            print("Test Result: PASSED")
        else:
            print("Test Result: NEEDS REVIEW")

    average_coverage = (
        total_keyword_coverage / total_tests * 100
        if total_tests else 0
    )

    pass_rate = (
        passed_tests / total_tests * 100
        if total_tests else 0
    )

    print("\n========== FINAL ANSWER EVALUATION ==========")

    print(f"Total Questions: {total_tests}")
    print(f"Passed Questions: {passed_tests}")
    print(f"Pass Rate: {pass_rate:.2f}%")
    print(f"Average Answer Keyword Coverage: {average_coverage:.2f}%")

    print("\nAnswer evaluation completed!")


if __name__ == "__main__":
    evaluate_answers()
