
from pathlib import Path

import chromadb
import ollama

from ingest import load_documents, build_chunk_records


# Project configuration
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_data"

EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "hr_policy_chunks"


def build_index():
    # 1. Load documents and split them into chunks
    documents = load_documents()
    chunks = build_chunk_records(documents)

    if not chunks:
        raise ValueError("No chunks found. Check your data folder.")

    # 2. Convert every chunk into an embedding
    response = ollama.embed(
        model=EMBED_MODEL,
        input=[chunk["text"] for chunk in chunks],
    )
    embeddings = response["embeddings"]

    # 3. Create a local, persistent Chroma database
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Rebuild this project's collection to avoid stale chunks
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # 4. Store vectors, text, IDs, and source filenames
    collection.add(
        ids=[chunk["chunk_id"] for chunk in chunks],
        embeddings=embeddings,
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {"source": chunk["source"]}
            for chunk in chunks
        ],
    )

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Vector database: {DB_DIR}")

    return collection


def search(collection, question, top_k=3):
    # Convert the question into a vector using the SAME model
    response = ollama.embed(
        model=EMBED_MODEL,
        input=question,
    )
    question_embedding = response["embeddings"][0]

    # Find the most similar stored vectors
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    matches = []

    for i, text in enumerate(results["documents"][0]):
        matches.append({
        "chunk_id": results["ids"][0][i],
        "text": text,
        "source": results["metadatas"][0][i]["source"],
        "distance": results["distances"][0][i],
})

    return matches


if __name__ == "__main__":
    collection = build_index()

    print("\nHR Policy Semantic Search")
    print("Enter a question, or type 'exit' to stop.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        matches = search(collection, question)

        for rank, match in enumerate(matches, start=1):
            print(f"\n--- Result {rank} ---")
            print(f"Source: {match['source']}")
            print(f"Cosine distance: {match['distance']:.4f}")
            print(match["text"])
