
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

def load_documents():
    documents = []

    # Find every .txt file inside the data folder
    for file_path in sorted(DATA_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            print(f"Skipping empty file: {file_path.name}")
            continue

        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents


def split_into_chunks(text, chunk_size=500, overlap=100):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def build_chunk_records(documents):
    records = []
    next_id = 1

    for document in documents:
        chunks = split_into_chunks(document["text"])

        for chunk in chunks:
            records.append({
                "chunk_id": f"chunk_{next_id}",
                "source": document["source"],
                "text": chunk
            })
            next_id += 1

    return records


if __name__ == "__main__":
    documents = load_documents()
    chunks = build_chunk_records(documents)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    for chunk in chunks:
        print("\n" + "=" * 60)
        print(f"ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(f"Characters: {len(chunk['text'])}")
        print(chunk["text"])
