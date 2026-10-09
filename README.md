# HR Policy Assistant

An AI-powered HR Policy Assistant that helps employees find answers to questions about company policies using natural language.

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from HR policy documents and generate answers grounded in the available evidence. It runs locally using Ollama for embeddings and language model inference, with ChromaDB for vector storage and semantic search.

## Key Features

- **Natural Language Q&A:** Ask questions about HR policies in plain English.
- **Document-Based Answers:** Generate responses using retrieved HR policy content.
- **Source Citations:** Display the source document, chunk ID, and retrieved text used to support an answer.
- **Semantic Search:** Find relevant policy information based on meaning rather than exact keyword matches.
- **Fallback Handling:** Indicate when the available policy documents do not contain sufficient information to answer a question.
- **Interactive Chat Interface:** Provide a simple Streamlit interface suitable for employee use and stakeholder demonstrations.
- **Local AI Processing:** Use locally running Ollama models for embeddings and answer generation.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Embedding Model | Nomic Embed Text |
| Language Model | Ollama-compatible local LLM |
| Vector Database | ChromaDB |
| Retrieval Approach | Semantic Search |
| AI Architecture | Retrieval-Augmented Generation (RAG) |

## How It Works

1. HR policy documents are loaded and divided into smaller text chunks.
2. An embedding model converts the chunks into numerical vectors.
3. ChromaDB stores the vectors alongside the original text and source metadata.
4. When a user asks a question, the system retrieves semantically relevant chunks.
5. A local language model generates an answer using the retrieved policy evidence.
6. The interface displays the answer and its sources, or a fallback when sufficient evidence is unavailable.

## Project Objective

The objective is to demonstrate how a locally running AI application can make organizational knowledge easier to access while improving transparency through source references and reducing unsupported answers through evidence-based retrieval and fallback handling.

## Scope and Limitations

This project is a proof of concept using a small, fictional HR policy corpus. Answer quality depends on document coverage, retrieval relevance, and language model behavior. The fallback mechanism is an initial safeguard, not a guarantee against every incorrect answer.

The application is intended for experimentation and demonstration, not as an authoritative source for real HR decisions.