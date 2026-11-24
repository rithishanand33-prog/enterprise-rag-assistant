
import json
from pathlib import Path

from src.chunk import create_chunks
from src.vectorstore import create_vectorstore


ABSTAIN_MESSAGE = (
    "I could not find this information in the uploaded documents."
)


def load_evaluation_dataset():

    project_root = Path(__file__).resolve().parent.parent

    dataset_path = project_root / "data" / "evaluation_dataset.json"

    with open(dataset_path, "r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_retrieval(k=3):

    print("\n========== RAG RETRIEVAL EVALUATION ==========")

    dataset = load_evaluation_dataset()

    print("\nLoading documents and creating vector database...")

    chunks = create_chunks()

    vectorstore = create_vectorstore(chunks)

    positive_total = 0
    negative_total = 0

    page_hits = 0
    total_keyword_coverage = 0

    negative_distances = []
    positive_distances = []

    for index, item in enumerate(dataset, start=1):

        question = item["question"]

        expected_page = item.get("expected_page")

        expected_keywords = item.get("expected_keywords", [])

        should_abstain = item.get("should_abstain", False)

        print("\n" + "=" * 70)

        print(f"\nTest {index}: {question}")

        print(
            f"Expected behavior: "
            f"{'ABSTAIN' if should_abstain else 'ANSWER'}"
        )

        results_with_scores = (
            vectorstore.similarity_search_with_score(
                question,
                k=k
            )
        )

        print("\nRetrieval distances:")

        for document, distance in results_with_scores:

            print(f"Distance: {float(distance):.4f}")

        results = [
            (document, float(distance))
            for document, distance in results_with_scores
        ]

        if results:

            best_distance = results[0][1]

            print(f"\nBest retrieval distance: {best_distance:.4f}")

        else:

            best_distance = None

            print("\nNo documents retrieved.")

        if should_abstain:

            negative_total += 1

            if best_distance is not None:
                negative_distances.append(best_distance)

            print("\nNegative question: no expected handbook answer.")

            print("Retrieval results are recorded for threshold analysis.")

            continue

        positive_total += 1

        if best_distance is not None:
            positive_distances.append(best_distance)

        retrieved_pages = [
            document.metadata.get("page", -1) + 1
            for document, distance in results
        ]

        retrieved_text = " ".join(
            document.page_content.lower()
            for document, distance in results
        )

        page_found = (
            expected_page in retrieved_pages
            if expected_page is not None
            else False
        )

        matched_keywords = [
            keyword
            for keyword in expected_keywords
            if keyword.lower() in retrieved_text
        ]

        keyword_coverage = (
            len(matched_keywords) / len(expected_keywords)
            if expected_keywords
            else 0
        )

        if page_found:
            page_hits += 1

        total_keyword_coverage += keyword_coverage

        print(f"\nExpected page: {expected_page}")

        print(f"Retrieved pages: {retrieved_pages}")

        print(
            f"Expected page found: "
            f"{'YES' if page_found else 'NO'}"
        )

        print(
            f"Keyword coverage: "
            f"{len(matched_keywords)}/{len(expected_keywords)}"
        )

        print(f"Matched keywords: {matched_keywords}")

    # Positive evaluation metrics.

    page_hit_rate = (
        page_hits / positive_total * 100
        if positive_total
        else 0
    )

    average_keyword_coverage = (
        total_keyword_coverage / positive_total * 100
        if positive_total
        else 0
    )

    print("\n" + "=" * 70)

    print("\n========== POSITIVE EVALUATION SUMMARY ==========")

    print(f"Answerable questions: {positive_total}")

    print(f"Expected pages retrieved: {page_hits}/{positive_total}")

    print(f"Page hit rate: {page_hit_rate:.2f}%")

    print(
        f"Average keyword coverage: "
        f"{average_keyword_coverage:.2f}%"
    )

    # Negative evaluation summary.

    print("\n========== NEGATIVE EVALUATION SUMMARY ==========")

    print(f"Unanswerable questions: {negative_total}")

    print(
        "These questions need to be rejected by the RAG "
        "relevance check."
    )

    # Compare best-match distances.

    print("\n========== DISTANCE ANALYSIS ==========")

    if positive_distances:

        print("\nAnswerable questions:")

        print(f"Minimum: {min(positive_distances):.4f}")

        print(f"Maximum: {max(positive_distances):.4f}")

        print(
            f"Average: "
            f"{sum(positive_distances) / len(positive_distances):.4f}"
        )

    if negative_distances:

        print("\nUnanswerable questions:")

        print(f"Minimum: {min(negative_distances):.4f}")

        print(f"Maximum: {max(negative_distances):.4f}")

        print(
            f"Average: "
            f"{sum(negative_distances) / len(negative_distances):.4f}"
        )

    print("\nEvaluation completed!")

    print(
        "\nNOTE: No distance threshold has been applied yet. "
        "We are collecting measurements before selecting one."
    )


if __name__ == "__main__":

    evaluate_retrieval()
