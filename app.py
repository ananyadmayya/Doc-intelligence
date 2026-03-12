"""
app.py — Streamlit UI for Self-Corrective Agentic RAG.
Level 1 + Level 2 + Level 3 modifications.

Run with:
    streamlit run app.py
"""

import logging
import os
import tempfile
import time

import streamlit as st

from retriever import ingest_pdf, ingest_url, clear_collection, get_document_preview
from graph_builder import run_query

logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Autonomous Document Intelligence",
    page_icon="🧠",
    layout="wide",
)

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
if "chat_history"   not in st.session_state: st.session_state.chat_history   = []
if "query_history"  not in st.session_state: st.session_state.query_history  = []
if "ingested_files" not in st.session_state: st.session_state.ingested_files = []
if "doc_summaries"  not in st.session_state: st.session_state.doc_summaries  = {}

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("🧠 Autonomous Document Intelligence")
st.caption("Self-Corrective Agentic RAG · LangGraph · ChromaDB · Groq")

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:

    # ── API Key ───────────────────────────────
    st.header("🔑 API Key")
    if not os.environ.get("GROQ_API_KEY"):
        api_key = st.text_input("Groq API Key", type="password")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
            st.success("API key set ✔")
    else:
        st.success("API key loaded from environment ✔")

    st.divider()

    # ── PDF Upload ────────────────────────────
    st.header("📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("📥 Ingest Documents", disabled=not uploaded_files):
        progress = st.progress(0)
        total = len(uploaded_files)
        for i, uploaded_file in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f"__{uploaded_file.name}"
            ) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                preview = get_document_preview(tmp_path)
                n_chunks = ingest_pdf(tmp_path)
                st.success(f"✅ {uploaded_file.name} → {n_chunks} chunks")
                if uploaded_file.name not in st.session_state.ingested_files:
                    st.session_state.ingested_files.append(uploaded_file.name)
                    st.session_state.doc_summaries[uploaded_file.name] = preview
            except Exception as exc:
                st.error(f"❌ {uploaded_file.name}: {exc}")
            finally:
                os.unlink(tmp_path)
            progress.progress((i + 1) / total)
        st.balloons()

    st.divider()

    # ── URL Ingestion ─────────────────────────
    st.header("🌐 Ingest from URL")
    url_input = st.text_input(
        "Paste a webpage URL",
        placeholder="https://example.com/article"
    )
    if st.button("🔗 Ingest URL", disabled=not url_input.strip()):
        with st.spinner("Scraping and ingesting..."):
            try:
                n_chunks = ingest_url(url_input.strip())
                st.success(f"✅ Ingested {n_chunks} chunks from URL")
                domain = url_input.strip().split("/")[2]
                if domain not in st.session_state.ingested_files:
                    st.session_state.ingested_files.append(domain)
            except Exception as exc:
                st.error(f"❌ URL ingestion failed: {exc}")

    st.divider()

    # ── Knowledge Base Manager ────────────────
    st.header("📚 Knowledge Base")

    if st.session_state.ingested_files:
        st.caption(f"{len(st.session_state.ingested_files)} source(s) ingested")
        for fname in list(st.session_state.ingested_files):
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"📄 `{fname}`")
            if col2.button("✕", key=f"del_{fname}"):
                st.session_state.ingested_files.remove(fname)
                if fname in st.session_state.doc_summaries:
                    del st.session_state.doc_summaries[fname]
                st.rerun()

        if st.session_state.doc_summaries:
            with st.expander("📋 Document Previews", expanded=False):
                for fname, preview in st.session_state.doc_summaries.items():
                    st.markdown(f"**{fname}**")
                    st.text_area(
                        label="",
                        value=preview[:500] + "..." if len(preview) > 500 else preview,
                        height=100,
                        key=f"preview_{fname}",
                        disabled=True,
                    )
                    st.divider()
    else:
        st.caption("No sources ingested yet.")

    if st.button("🗑️ Clear All Documents", type="secondary"):
        clear_collection()
        st.session_state.ingested_files = []
        st.session_state.doc_summaries  = {}
        st.session_state.chat_history   = []
        st.warning("Knowledge base and chat history cleared.")

    st.divider()

    # ── Query History ─────────────────────────
    st.header("🕘 Recent Queries")
    if st.session_state.query_history:
        selected_query = st.selectbox(
            "Re-run a previous query:",
            options=[""] + list(reversed(st.session_state.query_history[-10:])),
            format_func=lambda x: "Select a query..." if x == "" else (x[:55] + "..." if len(x) > 55 else x),
        )
    else:
        st.caption("No queries yet.")
        selected_query = ""

# ─────────────────────────────────────────────
# Conversation History — collapsed expander
# ─────────────────────────────────────────────
if st.session_state.chat_history:
    with st.expander(f"💬 Conversation History ({len(st.session_state.chat_history)} messages)", expanded=False):
        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(entry["question"])
            with st.chat_message("assistant"):
                st.markdown(entry["answer"])
                st.caption(
                    f"⏱ {entry['elapsed']}s · "
                    f"📝 {entry['word_count']} words · "
                    f"Grader: {entry['grader_verdict'].upper()} · "
                    f"Rewrites: {entry['rewrite_count']}"
                )

# ─────────────────────────────────────────────
# Query Input
# ─────────────────────────────────────────────
st.subheader("❓ Ask a Question")

default_query = selected_query if "selected_query" in dir() and selected_query else ""

query = st.text_area(
    "Enter your question",
    value=default_query,
    placeholder="e.g. What is the main finding? • What methodology was used? • Summarise the conclusion.",
    height=100,
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    run_btn = st.button("🚀 Run Query", type="primary", disabled=not query.strip())
with col2:
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()

# ─────────────────────────────────────────────
# Run Pipeline
# ─────────────────────────────────────────────
if run_btn and query.strip():
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Please enter your Groq API key in the sidebar first.")
        st.stop()

    if query.strip() not in st.session_state.query_history:
        st.session_state.query_history.append(query.strip())

    result     = None
    start_time = time.time()

    with st.status("Running agentic pipeline…", expanded=True) as status:
        try:
            st.write("🔍 Retrieving relevant chunks from ChromaDB...")
            st.write("⚖️ Grading retrieval quality...")
            result = run_query(
                query.strip(),
                conversation_history=st.session_state.chat_history[-3:],
            )
            if result["rewrite_count"] > 0:
                st.write(f"✏️ Query rewritten {result['rewrite_count']} time(s) to improve retrieval...")
            st.write("✍️ Generating answer with citations...")
            status.update(label="Pipeline complete ✅", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Pipeline failed ❌", state="error")
            st.error(f"Pipeline error: {exc}")
            st.stop()

    if result:
        elapsed    = round(time.time() - start_time, 1)
        word_count = len(result["answer"].split())

        # Save to chat history
        st.session_state.chat_history.append({
            "question":       query.strip(),
            "answer":         result["answer"],
            "elapsed":        elapsed,
            "word_count":     word_count,
            "grader_verdict": result["grader_verdict"],
            "rewrite_count":  result["rewrite_count"],
            "citations":      result["citations"],
        })

        # ── Answer ────────────────────────────────
        st.subheader("📝 Answer")
        st.markdown(result["answer"])
        st.caption(f"⏱ Generated in {elapsed}s · 📝 {word_count} words")

        # ── Export button ─────────────────────────
        export_text = (
            f"Question: {query.strip()}\n\n"
            f"Answer:\n{result['answer']}\n\n"
            f"Sources: {', '.join(result['citations'])}\n"
            f"Grader Verdict: {result['grader_verdict'].upper()}\n"
            f"Query Rewrites: {result['rewrite_count']}\n"
            f"Generated in: {elapsed}s\n"
        )
        st.download_button(
            label="📥 Export Answer",
            data=export_text,
            file_name="answer_export.txt",
            mime="text/plain",
        )

        # ── Confidence Indicator ──────────────────
        if result.get("context_docs"):
            avg_distance = sum(d["distance"] for d in result["context_docs"]) / len(result["context_docs"])
            confidence   = max(0.0, min(1.0, round(1 - avg_distance, 2)))

            if confidence >= 0.25:
                conf_color, conf_label = "🟢", "High"
            elif confidence >= 0.15:
                conf_color, conf_label = "🟡", "Medium"
            else:
                conf_color, conf_label = "🔴", "Low"

            st.markdown(f"**Retrieval Confidence:** {conf_color} {conf_label} ({int(confidence * 100)}%)")
            st.progress(confidence)

        # ── Pipeline Diagnostics ──────────────────
        with st.expander("🔍 Pipeline Diagnostics", expanded=True):
            cols = st.columns(3)
            cols[0].metric("Grader Verdict", result["grader_verdict"].upper())
            cols[1].metric("Query Rewrites", result["rewrite_count"])
            cols[2].metric("Sources Found",  len(result["citations"]))

            if result["citations"]:
                st.markdown("**Citations:**")
                for src in result["citations"]:
                    st.markdown(f"- `{src}`")

        # ── Retrieved Chunks ──────────────────────
        with st.expander("📄 Retrieved Context Chunks", expanded=False):
            st.caption("Exact text chunks the Grader evaluated before generating the answer.")
            if result.get("context_docs"):
                for i, doc in enumerate(result["context_docs"], 1):
                    clean_name = os.path.basename(doc["source"]).split("__")[-1]
                    st.markdown(f"**Chunk {i}** — `{clean_name}` — score: `{doc['distance']}`")
                    st.text_area(
                        label="",
                        value=doc["content"],
                        height=120,
                        key=f"chunk_{i}",
                        disabled=True,
                    )
                    st.divider()

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "Architecture: Retriever → Grader → (Rewriter ↻) → Generator · "
    "Powered by LangGraph + ChromaDB + Groq"
)