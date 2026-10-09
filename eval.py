
from answer import generate_answer, FALLBACK
from retrieval import build_index, search


TEST_CASES = [
    {
        "name": "Annual leave",
        "question": "How much vacation can a full-time employee take each year?",
        "expected_source": "leave_policy.txt",
        "required_groups": [
            ["20"],
            ["day"],
        ],
        "should_fallback": False,
    },
    {
        "name": "Expense deadline",
        "question": "How long after a business trip can I submit expenses?",
        "expected_source": "expense_policy.txt",
        "required_groups": [
            ["15"],
            ["calendar"],
        ],
        "should_fallback": False,
    },
    {
        "name": "Remote work",
        "question": "Can I work from home two days every week?",
        "expected_source": "workplace_policy.txt",
        "required_groups": [
            ["two", "2"],
            ["manager", "approval"],
        ],
        "should_fallback": False,
    },
    {
        "name": "Unsupported benefit",
        "question": (
            "Does Acme pay for international flights "
            "for employees' family members?"
        ),
        "expected_source": None,
        "required_groups": [],
        "should_fallback": True,
    },
]


def answer_is_correct(answer, test_case):
    text = answer.lower()

    if test_case["should_fallback"]:
        return (
            "couldn't find sufficient information" in text
        )

    return all(
        any(term in text for term in group)
        for group in test_case["required_groups"]
    )


def evaluate():
    # Build the index once for this evaluation run.
    collection = build_index()

    answerable_distances = []
    unsupported_distances = []

    retrieval_passes = 0
    answer_passes = 0

    print("\n===== RAG EVALUATION =====")

    for test in TEST_CASES:
        print(f"\n--- {test['name']} ---")
        print(f"Question: {test['question']}")

        # Evaluate retrieval independently of generation.
        matches = search(
            collection,
            test["question"],
            top_k=3,
        )

        for match in matches:
            print(
                f"Retrieved: {match['source']} | "
                f"distance={match['distance']:.4f}"
            )

        best_distance = matches[0]["distance"] if matches else None

        if test["should_fallback"]:
            if best_distance is not None:
                unsupported_distances.append(best_distance)
        elif best_distance is not None:
            answerable_distances.append(best_distance)

        expected_source = test["expected_source"]

        if expected_source is None:
            retrieval_ok = None
        else:
            retrieval_ok = any(
                match["source"] == expected_source
                for match in matches
            )

            retrieval_passes += int(retrieval_ok)

        print(f"Retrieval source check: {retrieval_ok}")

        # Evaluate the full RAG pipeline.
        answer, sources = generate_answer(
            test["question"],
            collection,
        )

        answer_ok = answer_is_correct(answer, test)

        print(f"Answer: {answer}")
        print(f"Answer check: {'PASS' if answer_ok else 'FAIL'}")

        if answer_ok:
            answer_passes += 1

    answerable_count = sum(
        not test["should_fallback"]
        for test in TEST_CASES
    )

    print("\n===== SUMMARY =====")
    print(
        f"Retrieval source checks: "
        f"{retrieval_passes}/{answerable_count}"
    )
    print(
        f"Answer checks: "
        f"{answer_passes}/{len(TEST_CASES)}"
    )

    if answerable_distances and unsupported_distances:
        max_answerable = max(answerable_distances)
        min_unsupported = min(unsupported_distances)

        print("\n===== DISTANCE ANALYSIS =====")
        print(
            f"Worst answerable distance: {max_answerable:.4f}"
        )
        print(
            f"Best unsupported-query distance: "
            f"{min_unsupported:.4f}"
        )

        if max_answerable < min_unsupported:
            suggested = (
                max_answerable + min_unsupported
            ) / 2

            print(
                f"Possible threshold to investigate: "
                f"{suggested:.4f}"
            )
            print(
                "This is a candidate for further testing, "
                "not an automatically safe threshold."
            )
        else:
            print(
                "The distances overlap. A single distance "
                "threshold cannot reliably separate these "
                "answerable and unsupported questions."
            )


if __name__ == "__main__":
    evaluate()
