"""
retriever.py — Document ingestion and vector retrieval using ChromaDB + FastEmbed.

Level 3 upgrades:
  - Academic paper chunking by section headings
  - URL ingestion via web scraping
"""

import logging
import re
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
COLLECTION_NAME = "document_intelligence"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 80
TOP_K           = 4

ACADEMIC_HEADINGS = [
    "abstract", "introduction", "background", "related work",
    "literature review", "methodology", "methods", "approach",
    "experiment", "evaluation", "results", "discussion",
    "conclusion", "future work", "references", "appendix",
]


# ─────────────────────────────────────────────
# ChromaDB client
# ─────────────────────────────────────────────
def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path="./chroma_store")
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=DefaultEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


# ─────────────────────────────────────────────
# PDF parsing
# ─────────────────────────────────────────────
def _extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        logger.info("Extracted %d characters from '%s'", len(text), pdf_path)
        return text
    except Exception as exc:
        logger.error("Failed to read PDF '%s': %s", pdf_path, exc)
        raise


# ─────────────────────────────────────────────
# Smart chunking
# ─────────────────────────────────────────────
def _is_academic_paper(text: str) -> bool:
    text_lower = text.lower()
    matches = sum(1 for h in ACADEMIC_HEADINGS if h in text_lower)
    return matches >= 3


def _chunk_by_sections(text: str) -> List[str]:
    heading_pattern = re.compile(
        r'\n\s*(?:' + '|'.join(ACADEMIC_HEADINGS) + r')[^\n]*\n',
        re.IGNORECASE
    )
    parts = heading_pattern.split(text)
    headers = heading_pattern.findall(text)
    chunks = []
    for i, part in enumerate(parts):
        header = headers[i - 1].strip() if i > 0 and i - 1 < len(headers) else ""
        section_text = (header + "\n" + part).strip() if header else part.strip()
        if not section_text:
            continue
        if len(section_text) <= CHUNK_SIZE * 2:
            chunks.append(section_text)
        else:
            chunks.extend(_chunk_text(section_text))
    return [c for c in chunks if len(c) > 30]


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 30]


def _smart_chunk(text: str) -> List[str]:
    if _is_academic_paper(text):
        logger.info("Academic paper detected — using section-based chunking")
        chunks = _chunk_by_sections(text)
        if chunks:
            return chunks
    return _chunk_text(text)


# ─────────────────────────────────────────────
# URL ingestion
# ─────────────────────────────────────────────
def ingest_url(url: str) -> int:
    """
    Scrape a webpage and ingest its text content into ChromaDB.
    Returns number of chunks stored.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Run: pip install requests beautifulsoup4")

    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r'\n{3,}', '\n\n', text)

    if len(text) < 100:
        raise ValueError("Page content too short or blocked.")

    chunks = _smart_chunk(text)
    collection = _get_collection()

    from urllib.parse import urlparse
    parsed = urlparse(url)
    source_name = parsed.netloc + parsed.path

    ids = [f"url_{abs(hash(url))}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_name, "chunk_index": i, "type": "url"} for i in range(len(chunks))]

    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    logger.info("Ingested %d chunks from URL '%s'", len(chunks), url)
    return len(chunks)


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def ingest_pdf(pdf_path: str) -> int:
    """Parse a PDF, chunk smartly, upsert into ChromaDB."""
    text = _extract_text_from_pdf(pdf_path)
    chunks = _smart_chunk(text)

    collection = _get_collection()
    stem = Path(pdf_path).stem

    ids = [f"{stem}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": pdf_path, "chunk_index": i} for i in range(len(chunks))]

    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    logger.info("Ingested %d chunks from '%s' into ChromaDB.", len(chunks), pdf_path)
    return len(chunks)


def get_document_preview(pdf_path: str) -> str:
    """Return first 1500 characters of a PDF for summary generation."""
    try:
        text = _extract_text_from_pdf(pdf_path)
        return text[:1500].strip()
    except Exception:
        return ""


def retrieve(query: str, top_k: int = TOP_K) -> List[dict]:
    """Retrieve top-k relevant chunks for a query."""
    collection = _get_collection()
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
    )
    docs = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        docs.append({
            "content":     doc,
            "source":      meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", -1),
            "distance":    round(dist, 4),
        })
    logger.info("Retrieved %d chunks for query: '%s'", len(docs), query[:60])
    return docs


def clear_collection() -> None:
    """Drop and recreate the ChromaDB collection."""
    client = chromadb.PersistentClient(path="./chroma_store")
    client.delete_collection(COLLECTION_NAME)
    client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=DefaultEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("ChromaDB collection '%s' cleared.", COLLECTION_NAME)