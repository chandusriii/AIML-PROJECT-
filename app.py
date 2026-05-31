from pathlib import Path
import shutil

import streamlit as st

from rag_core.chunking import chunk_pages
from rag_core.config import TOP_K, UPLOADS_DIR
from rag_core.pdf_utils import extract_pdf_pages
from rag_core.qa import answer_question
from rag_core.vector_store import VectorStore

st.set_page_config(page_title="RAG Chat with PDFs", page_icon="📚", layout="wide")
st.title("📚 RAG Chat with Your PDFs")
st.caption("Upload PDFs, build embeddings, and ask questions with citations.")

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
store = VectorStore()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Performance")
    perf_mode = st.selectbox(
        "Speed profile",
        options=["fast", "accurate"],
        index=0,
        help="Fast is quicker with shorter context and fewer retrieved chunks.",
    )

    st.header("Answer Mode")
    answer_mode = st.selectbox(
        "Choose response backend",
        options=["ollama", "openai", "local"],
        index=0,
    )
    default_model = "phi3:mini" if perf_mode == "fast" else "llama3.2:3b"
    ollama_model = st.text_input("Ollama model", value=default_model)

    st.header("0) Quick Demo Loader")
    demo_pdf_path = st.text_input(
        "Local PDF path for one-click demo load",
        value="",
        placeholder=r"Example: C:\Users\YourName\Downloads\sample.pdf",
    )
    if st.button("Load Demo PDF", use_container_width=True):
        src = Path(demo_pdf_path.strip())
        if not src.exists():
            st.error(f"File not found: {src}")
        elif src.suffix.lower() != ".pdf":
            st.error("Please provide a .pdf file path.")
        else:
            target = Path(UPLOADS_DIR) / src.name
            shutil.copy2(src, target)
            st.success(f"Loaded demo PDF: {target.name}")

    st.header("1) Upload PDFs")
    files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Save Uploaded PDFs", use_container_width=True):
        if not files:
            st.warning("Please upload at least one PDF.")
        else:
            saved = 0
            for f in files:
                target = Path(UPLOADS_DIR) / f.name
                target.write_bytes(f.getbuffer())
                saved += 1
            st.success(f"Saved {saved} PDF(s) to data/uploads.")

    st.header("2) Build Index")
    if st.button("Build Index", use_container_width=True):
        pdf_paths = sorted(Path(UPLOADS_DIR).glob("*.pdf"))
        if not pdf_paths:
            st.error("No PDFs found in data/uploads. Upload and save files first.")
        else:
            pages = []
            for pdf in pdf_paths:
                pages.extend(extract_pdf_pages(pdf))
            chunks = chunk_pages(pages)
            store.build_index(chunks)
            st.success(f"Index built from {len(pdf_paths)} file(s), {len(chunks)} chunks.")

st.header("3) Ask Questions")
query = st.text_input("Ask a question about your uploaded PDFs")

if st.button("Get Answer", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        try:
            top_k = 2 if perf_mode == "fast" else TOP_K
            max_chars_per_chunk = 420 if perf_mode == "fast" else 900
            hits = store.search(query, top_k=top_k)
            answer = answer_question(
                query,
                hits,
                mode=answer_mode,
                ollama_model=ollama_model,
                max_chars_per_chunk=max_chars_per_chunk,
            )
            st.session_state.chat_history.append(
                {
                    "question": query,
                    "answer": answer,
                    "citations": hits,
                }
            )
            st.subheader("Answer")
            st.write(answer)

            st.subheader("Citations")
            if not hits:
                st.info("No supporting chunks found.")
            for i, h in enumerate(hits, start=1):
                with st.expander(f"[{i}] {h['source']} (page {h['page']})"):
                    st.write(h["text"])
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Chat History")
with col2:
    if st.button("Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.success("History cleared.")

if not st.session_state.chat_history:
    st.caption("No previous questions yet.")
else:
    for idx, item in enumerate(reversed(st.session_state.chat_history), start=1):
        with st.expander(f"{idx}. {item['question']}"):
            st.write(item["answer"])
            st.markdown("**Citations**")
            if not item["citations"]:
                st.caption("No citations found for this query.")
            for i, cite in enumerate(item["citations"], start=1):
                st.markdown(f"- [{i}] {cite['source']} (page {cite['page']})")
