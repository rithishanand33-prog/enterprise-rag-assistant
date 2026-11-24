
import json
import re
from pathlib import Path

from src.chunk import create_chunks
from src.vectorstore import create_vectorstore
from src.llm import get_llm
from src.rag import answer_question, ABSTAIN_MESSAGE


def load_evaluation_dataset():

    project_root = Path(__file__).resolve().parent.parent
    dataset_path = project_root / "data" / "evaluation_dataset.json"

    with open(dataset_path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(text):
    """
    Normalize whitespace and convert text to lowercase.

    This prevents line breaks and extra spaces from
    causing incorrect keyword matching.
    """

    return re.sub(r"\s+", " ", str(text)).strip().lower()


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

    positive_tests = 0
    positive_passed = 0

    negative_tests = 0
    negative_passed = 0

    total_keyword_coverage = 0

    for index, item in enumerate(dataset, start=1):

        question = item["question"]
        expected_keywords = item.get("expected_keywords", [])
        should_abstain = item.get("should_abstain", False)

        print(f"\n{'=' * 60}")
        print(f"Test {index}: {question}")

        print(
            "Expected behavior:",
            "ABSTAIN" if should_abstain else "ANSWER"
        )

        answer = answer_question(
            question,
            vectorstore,
            llm
        )

        print("\nGenerated Answer:")
        print(answer)

        normalized_answer = normalize_text(answer)

        # -----------------------------------------
        # NEGATIVE TEST: EXPECTED ABSTENTION
        # -----------------------------------------

        if should_abstain:

            negative_tests += 1

            abstained_correctly = (
                normalize_text(ABSTAIN_MESSAGE)
                in normalized_answer
            )

            if abstained_correctly:

                negative_passed += 1
                passed_tests += 1

                print("\nCorrectly abstained: YES")
                print("Test Result: PASSED")

            else:

                print("\nCorrectly abstained: NO")
                print("Test Result: FAILED")

            continue

        # -----------------------------------------
        # POSITIVE TEST: EXPECTED ANSWER
        # -----------------------------------------

        positive_tests += 1

        normalized_keywords = [
            normalize_text(keyword)
            for keyword in expected_keywords
        ]

        matched_keywords = [
            keyword
            for keyword, normalized_keyword
            in zip(expected_keywords, normalized_keywords)
            if normalized_keyword in normalized_answer
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

        if expected_keywords and keyword_coverage == 1:

            positive_passed += 1
            passed_tests += 1

            print("Test Result: PASSED")

        else:

            print("Test Result: NEEDS REVIEW")

    # -----------------------------------------
    # FINAL EVALUATION SUMMARY
    # -----------------------------------------

    overall_accuracy = (
        passed_tests / total_tests * 100
        if total_tests else 0
    )

    positive_pass_rate = (
        positive_passed / positive_tests * 100
        if positive_tests else 0
    )

    negative_pass_rate = (
        negative_passed / negative_tests * 100
        if negative_tests else 0
    )

    average_keyword_coverage = (
        total_keyword_coverage / positive_tests * 100
        if positive_tests else 0
    )

    print("\n")
    print("=" * 60)
    print("========== FINAL ANSWER EVALUATION ==========")

    print(f"\nTotal Questions: {total_tests}")

    print(
        f"Overall Passed: "
        f"{passed_tests}/{total_tests}"
    )

    print(
        f"Overall Accuracy: "
        f"{overall_accuracy:.2f}%"
    )

    print("\n--- ANSWERABLE QUESTIONS ---")

    print(f"Total: {positive_tests}")
    print(f"Passed: {positive_passed}")

    print(
        f"Answer Pass Rate: "
        f"{positive_pass_rate:.2f}%"
    )

    print(
        f"Average Keyword Coverage: "
        f"{average_keyword_coverage:.2f}%"
    )

    print("\n--- UNANSWERABLE QUESTIONS ---")

    print(f"Total: {negative_tests}")
    print(f"Correctly Rejected: {negative_passed}")

    print(
        f"Abstention Accuracy: "
        f"{negative_pass_rate:.2f}%"
    )

    print("\nEvaluation completed!")


if __name__ == "__main__":

    evaluate_answers()
