"""
app.py
------
Streamlit interface for the Domain-Specific RAG Chatbot.

- Sidebar: PDF uploader + "Process Documents" button
- Main area: chat input, chat history, sources shown under each answer
- "Clear Chat" button resets the conversation (not the index)
"""

import streamlit as st
from dotenv import load_dotenv

from document_loader import validate_file, extract_all
from vector_store import VectorStore
from rag_pipeline import answer_question

load_dotenv()

st.set_page_config(page_title="Domain RAG Chatbot", page_icon="📄", layout="wide")

# ---------- Session state ----------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role", "content", "sources"}

# ---------- Sidebar ----------
with st.sidebar:
    st.header("📄 Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.caption("Uploaded files:")
        for f in uploaded_files:
            st.write(f"• {f.name}")

    process_clicked = st.button("Process Documents", type="primary", use_container_width=True)

    if process_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        else:
            errors = []
            valid_files = []
            for f in uploaded_files:
                ok, err = validate_file(f)
                if ok:
                    valid_files.append(f)
                else:
                    errors.append(err)

            for e in errors:
                st.error(e)

            if valid_files:
                with st.spinner("Extracting text and building the search index..."):
                    pages = extract_all(valid_files)
                    try:
                        st.session_state.vector_store.build_index(pages)
                        st.session_state.vector_store.save()
                        st.session_state.docs_processed = True
                        st.success(f"Processed {len(valid_files)} document(s), "
                                   f"{len(pages)} page(s) with text.")
                    except ValueError as e:
                        st.error(str(e))

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.caption(
        "⚠️ Answers are generated from your uploaded documents. "
        "Always verify high-stakes information before relying on it."
    )

# ---------- Main area ----------
st.title("Domain-Specific RAG Chatbot")
st.caption("Ask questions about the documents you've uploaded and processed.")

if not st.session_state.docs_processed:
    # Try loading a previously saved index from disk
    if st.session_state.vector_store.load():
        st.session_state.docs_processed = True
        st.info("Loaded a previously saved document index.")

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(f"📄 {s['source']}, page {s['page']} (score: {s['score']})")

# Chat input
question = st.chat_input("Ask a question about your documents...")

if question:
    if not st.session_state.docs_processed:
        st.warning("Please upload and process at least one PDF before asking questions.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = answer_question(st.session_state.vector_store, question)
                st.write(result["answer"])
                if result["sources"]:
                    with st.expander("Sources"):
                        for s in result["sources"]:
                            st.write(f"📄 {s['source']}, page {s['page']} (score: {s['score']})")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })
