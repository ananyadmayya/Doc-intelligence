"""
graph_builder.py — LangGraph state machine for Self-Corrective Agentic RAG.

Level 3 upgrades:
  - Conversation memory (previous Q&A passed as context)
  - Clean answer output (no citation paths in answer text)
"""

import json
import logging
import os
from typing import List, Literal, TypedDict

from langgraph.graph import END, StateGraph
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from retriever import retrieve

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────
LLM = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=2048,
    api_key=os.environ.get("GROQ_API_KEY", ""),
)

# ─────────────────────────────────────────────
# Graph State
# ─────────────────────────────────────────────
class RAGState(TypedDict):
    original_query:   str
    current_query:    str
    context_docs:     List[dict]
    grader_verdict:   Literal["yes", "no", ""]
    answer:           str
    citations:        List[str]
    rewrite_count:    int
    conversation_history: List[dict]   # Level 3: memory


MAX_REWRITES = 2


# ─────────────────────────────────────────────
# Node: Retriever
# ─────────────────────────────────────────────
def retriever_node(state: RAGState) -> RAGState:
    logger.info("[retriever] query: %s", state["current_query"])
    docs = retrieve(state["current_query"])
    return {**state, "context_docs": docs}


# ─────────────────────────────────────────────
# Node: Grader
# ─────────────────────────────────────────────
GRADER_SYSTEM = """You are an expert relevance judge for a RAG pipeline.
Given a user query and retrieved text snippets, decide whether the snippets
contain enough information to answer the query.

Respond ONLY with a JSON object:
  {"verdict": "yes"}   — context is sufficient
  {"verdict": "no"}    — context is irrelevant or too sparse

No extra text."""


def grader_node(state: RAGState) -> RAGState:
    context_text = "\n---\n".join(d["content"] for d in state["context_docs"])
    messages = [
        SystemMessage(content=GRADER_SYSTEM),
        HumanMessage(content=f"Query: {state['current_query']}\n\nContext:\n{context_text}"),
    ]
    try:
        response = LLM.invoke(messages)
        parsed = json.loads(response.content.strip())
        verdict = parsed.get("verdict", "no").lower()
    except Exception as exc:
        logger.warning("[grader] parse error (%s) — defaulting to 'no'", exc)
        verdict = "no"
    logger.info("[grader] verdict: %s", verdict)
    return {**state, "grader_verdict": verdict}


# ─────────────────────────────────────────────
# Node: Rewriter
# ─────────────────────────────────────────────
REWRITER_SYSTEM = """You are a query optimisation specialist.
The search query failed to retrieve relevant documents.
Rewrite it to be clearer, more specific, and match technical vocabulary in documents.
Return ONLY the improved query string. No explanation."""


def rewriter_node(state: RAGState) -> RAGState:
    messages = [
        SystemMessage(content=REWRITER_SYSTEM),
        HumanMessage(content=(
            f"Original question: {state['original_query']}\n"
            f"Failed query: {state['current_query']}\n"
            "Provide a better query:"
        )),
    ]
    response = LLM.invoke(messages)
    new_query = response.content.strip()
    logger.info("[rewriter] new query: %s", new_query)
    return {
        **state,
        "current_query": new_query,
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }


# ─────────────────────────────────────────────
# Node: Generator
# ─────────────────────────────────────────────
GENERATOR_SYSTEM = """You are an expert document analyst.
Answer the user's question using ONLY the provided context snippets.
- Give a thorough, detailed answer using all relevant information from the context.
- Use bullet points or paragraphs to structure your answer clearly.
- Include all key details, numbers, names, and facts mentioned in the context.
- Do NOT include citation numbers or source paths anywhere in your answer.
- Do NOT write a Citations line in your answer.
- If the context is insufficient, say so honestly.
- If conversation history is provided, use it to understand follow-up questions."""


def generator_node(state: RAGState) -> RAGState:
    context_blocks = []
    for i, doc in enumerate(state["context_docs"], 1):
        clean_name = os.path.basename(doc["source"]).split("__")[-1]
        context_blocks.append(
            f"[{i}] Source: {clean_name} (chunk {doc['chunk_index']})\n{doc['content']}"
        )
    context_text = "\n\n".join(context_blocks)

    # Build conversation history context
    history_text = ""
    history = state.get("conversation_history", [])
    if history:
        recent = history[-3:]  # last 3 exchanges
        history_lines = []
        for entry in recent:
            history_lines.append(f"User: {entry['question']}")
            history_lines.append(f"Assistant: {entry['answer'][:200]}")
        history_text = "\n\nPrevious conversation:\n" + "\n".join(history_lines)

    messages = [
        SystemMessage(content=GENERATOR_SYSTEM),
        HumanMessage(content=(
            f"Question: {state['original_query']}"
            f"{history_text}\n\n"
            f"Context:\n{context_text}"
        )),
    ]
    response = LLM.invoke(messages)
    answer = response.content.strip()
    citations = list({
        os.path.basename(doc["source"]).split("__")[-1]
        for doc in state["context_docs"]
    })
    logger.info("[generator] answer generated (%d chars)", len(answer))
    return {**state, "answer": answer, "citations": citations}


# ─────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────
def route_after_grader(state: RAGState) -> Literal["generator", "rewriter", "generator_fallback"]:
    if state["grader_verdict"] == "yes":
        return "generator"
    if state.get("rewrite_count", 0) >= MAX_REWRITES:
        logger.warning("[router] max rewrites reached — forcing generation")
        return "generator_fallback"
    return "rewriter"


# ─────────────────────────────────────────────
# Build graph
# ─────────────────────────────────────────────
def build_graph():
    builder = StateGraph(RAGState)
    builder.add_node("retriever",          retriever_node)
    builder.add_node("grader",             grader_node)
    builder.add_node("rewriter",           rewriter_node)
    builder.add_node("generator",          generator_node)
    builder.add_node("generator_fallback", generator_node)

    builder.set_entry_point("retriever")
    builder.add_edge("retriever", "grader")
    builder.add_edge("rewriter",  "retriever")
    builder.add_conditional_edges(
        "grader",
        route_after_grader,
        {
            "generator":          "generator",
            "rewriter":           "rewriter",
            "generator_fallback": "generator_fallback",
        },
    )
    builder.add_edge("generator",          END)
    builder.add_edge("generator_fallback", END)
    return builder.compile()


rag_graph = build_graph()


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────
def run_query(query: str, conversation_history: List[dict] = None) -> dict:
    """
    Execute the full RAG pipeline for a given query.
    Optionally pass conversation_history for memory-aware answers.
    """
    initial_state: RAGState = {
        "original_query":       query,
        "current_query":        query,
        "context_docs":         [],
        "grader_verdict":       "",
        "answer":               "",
        "citations":            [],
        "rewrite_count":        0,
        "conversation_history": conversation_history or [],
    }
    final_state = rag_graph.invoke(initial_state)
    return {
        "answer":        final_state["answer"],
        "citations":     final_state["citations"],
        "rewrite_count": final_state["rewrite_count"],
        "grader_verdict":final_state["grader_verdict"],
        "context_docs":  final_state["context_docs"],
    }