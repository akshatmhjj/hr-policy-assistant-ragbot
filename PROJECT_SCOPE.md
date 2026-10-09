# HR Policy Assistant — Project Scope

## 1. Business Problem
Employees spend time searching lengthy HR documents to find information about leave, attendance, holidays, and reimbursement policies.

## 2. Target User
Employees and HR personnel who need quick answers to HR policy questions.

## 3. Proposed Solution
A chat-based assistant that answers questions using an uploaded collection of HR policy documents.

## 4. Core Features
- Accept HR policy documents as input.
- Allow users to ask questions in natural language.
- Retrieve relevant document chunks.
- Generate answers using Claude and retrieved evidence.
- Display the source document and relevant text used for each answer.
- Respond with "I couldn't find sufficient information in the provided policies" when evidence is insufficient.

## 5. Technology Stack
- Language: Python
- UI: Streamlit
- LLM: Anthropic Claude API
- Retrieval: Embeddings and a vector store (to be selected later)

## 6. Scope for the First Demo
The initial version will support a small collection of HR policy documents and demonstrate:
1. A question that can be answered from the documents.
2. An answer with a visible source.
3. A question that cannot be answered from the documents, triggering the fallback.

## 7. Success Criteria
- Answers are grounded in the provided documents.
- Source references point to the relevant document chunks.
- Unsupported questions trigger the fallback instead of an invented answer.
- A non-technical stakeholder can use the interface without assistance.