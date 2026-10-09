
from pathlib import Path

import streamlit as st

from answer import generate_answer
from retrieval import build_index


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FALLBACK = (
    "I couldn't find sufficient information in the provided "
    "HR policies to answer that question."
)


# ---------- Page configuration ----------

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("HR Policy Assistant")
st.caption(
    "Ask questions about company HR policies. "
    "Answers are generated locally using Ollama."
)


# ---------- Build the index once per Streamlit session ----------

@st.cache_resource
def get_collection():
    return build_index()


try:
    collection = get_collection()
except Exception as error:
    st.error(
        "Could not initialize the policy search system. "
        "Check that Ollama is running and the required models "
        "are installed."
    )
    st.exception(error)
    st.stop()


# ---------- Sidebar: document corpus ----------

with st.sidebar:
    st.header("Policy Documents")
    st.write(
        "The assistant searches the following local policy files."
    )

    policy_files = sorted(DATA_DIR.glob("*.txt"))

    if policy_files:
        for file_path in policy_files:
            st.markdown(f"- `{file_path.name}`")
    else:
        st.warning("No policy documents found in the data folder.")

    st.divider()

    st.subheader("How it works")
    st.markdown(
        """
        1. Your question is converted into an embedding.
        2. Relevant policy chunks are retrieved.
        3. A local language model generates an answer.
        4. Sources are displayed for inspection.
        """
    )

    st.info(
        "If the policies do not contain sufficient evidence, "
        "the assistant should say it cannot answer."
    )


# ---------- Chat history ----------

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources):
    if not sources:
        return

    with st.expander(
        f"View retrieved sources ({len(sources)})",
        expanded=False,
    ):
        for index, source in enumerate(sources, start=1):
            st.markdown(
                f"**Source {index}: `{source['source']}`**"
            )
            st.caption(
                f"Chunk ID: {source['chunk_id']} · "
                f"Cosine distance: {source['distance']:.4f}"
            )
            st.text(source["text"])

            if index < len(sources):
                st.divider()


# Render previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("fallback"):
            st.warning(message["content"])
        else:
            st.markdown(message["content"])

        if message["role"] == "assistant":
            render_sources(message.get("sources", []))


# ---------- Process a new question ----------

question = st.chat_input("Ask a question about HR policies...")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies and generating an answer..."):
            try:
                answer, sources = generate_answer(
                    question,
                    collection,
                )

                is_fallback = (
                    FALLBACK.lower() in answer.lower()
                )

                if is_fallback:
                    st.warning(answer)
                else:
                    st.markdown(answer)

                render_sources(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "fallback": is_fallback,
                    }
                )

            except Exception:
                error_message = (
                    "The assistant encountered an error. "
                    "Please check the local Ollama service "
                    "and try again."
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                        "fallback": True,
                    }
                )
