"""Simple Streamlit UI for the RAG app."""

from pathlib import Path

import streamlit as st

from rag import answer_question, get_retriever, load_vectorstore

st.set_page_config(page_title="Meta Policy RAG")
st.title("Meta Policy Q&A")
st.caption("Ask questions about Meta's Privacy Policy and Instagram's Terms of Use.")


@st.cache_resource
def get_cached_retriever():
    # Local Qdrant allows one client per folder, so open it once and reuse it.
    return get_retriever(load_vectorstore())


# A form sends the typed text and the button click together (Enter also submits).
with st.form("question_form"):
    question = st.text_input("Your question", placeholder="What data does Meta collect?")
    submitted = st.form_submit_button("Ask")

if submitted and question:
    with st.spinner("Searching documents..."):
        answer, docs = answer_question(get_cached_retriever(), question)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for i, doc in enumerate(docs, start=1):
        source = Path(doc.metadata["source"]).name
        with st.expander(f"{i}. {source} — page {doc.metadata['page'] + 1}"):
            st.write(doc.page_content)
