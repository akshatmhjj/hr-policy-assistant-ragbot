
import ollama

from retrieval import build_index, search


CHAT_MODEL = "llama3.2:latest"

# Initial threshold. We will validate this against real test results.
MAX_DISTANCE = 0.45

FALLBACK = (
    "I couldn't find sufficient information in the provided "
    "HR policies to answer that question."
)


def generate_answer(question, collection):
    # Step 1: Retrieve relevant chunks
    matches = search(collection, question, top_k=1)

    if not matches:
        return FALLBACK, []

    # Step 2: Reject results that are too dissimilar
    best_distance = matches[0]["distance"]

    if best_distance > MAX_DISTANCE:
        return FALLBACK, []

    # Step 3: Prepare evidence for the chat model
    evidence = []

    for match in matches:
        evidence.append(
            f"Source: {match['source']}\n"
            f"Chunk ID: {match['chunk_id']}\n"
            f"Text: {match['text']}"
        )

    context = "\n\n---\n\n".join(evidence)

    # Step 4: Generate an answer using local Ollama
    response = ollama.chat(
        model=CHAT_MODEL,        
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an HR policy assistant. "
                    "Answer the user's question using the supplied policy text. "
                    "If the text explicitly contains the answer, provide it. "
                    "If the text does not contain the answer, say: "
                    "I couldn't find sufficient information in the provided "
                    "HR policies to answer that question. "
                    "Never invent policy details."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Policy text:\n{context}\n\n"
                    "What does the policy say? Give the direct answer."
                ),
            },
        ],
        options={"temperature": 0},
    )

    answer = response["message"]["content"].strip()

    # Step 5: Return the answer and evidence used
    return answer, matches


if __name__ == "__main__":
    print("Building the HR policy index...")
    collection = build_index()

    print("\nHR Policy Assistant is ready.")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        try:
            answer, sources = generate_answer(question, collection)

            print(f"\nAssistant: {answer}")

            if sources:
                print("\nRetrieved sources:")

                for source in sources:
                    print(
                        f"- {source['source']} | "
                        f"{source['chunk_id']} | "
                        f"distance={source['distance']:.4f}"
                    )
                    # print(source["text"])

        except Exception as error:
            print(f"\nAn error occurred: {error}")
            print("Check that Ollama is running and the models are available.")
