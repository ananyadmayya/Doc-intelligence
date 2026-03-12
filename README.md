# 🧠 Autonomous Document Intelligence

A Self-Corrective Agentic RAG system built with LangGraph, ChromaDB, and Groq.

## What Makes This Different From Basic RAG

Most RAG systems blindly retrieve chunks and generate answers even when the context is irrelevant. This system uses a **LangGraph state machine** with a self-corrective loop:
```
Question → Retrieve → Grade → Generate
                         ↓ (if bad)
                      Rewrite Query → Retrieve again (max 2 retries)
```

## Features

- ✅ Self-corrective retrieval with Grader node
- ✅ Query rewriting when retrieval fails
- ✅ Retrieval confidence indicator
- ✅ Pipeline diagnostics panel
- ✅ Chat history with conversation memory
- ✅ Web URL ingestion
- ✅ Academic paper section-based chunking
- ✅ Answer export
- ✅ Per-file knowledge base management

## Tech Stack

- **LangGraph** — agentic state machine
- **ChromaDB** — vector store
- **Groq** — LLM inference (llama-3.1-8b-instant)
- **Streamlit** — UI
- **PyMuPDF** — PDF parsing

## How To Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture
```
Retriever → Grader → Rewriter (↻ max 2) → Generator
```

Built as a portfolio project targeting AI Engineering roles.
