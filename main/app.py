# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:00:52 2026

@author: Sharad
"""

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import time
from datetime import datetime

# ── Config ────────────────────────────────────────
CHROMA_PATH      = "D:/OneDrive - Sutra Management/Desktop/bfsi_rag/chroma_store"
COLLECTION_NAME  = "bfsi_docs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K            = 3
OLLAMA_URL       = "http://localhost:11434/api/generate"
LLM_MODEL        = "phi3:mini"
CONFIDENCE_THRESHOLD = 0.55
NUM_PREDICT      = 450
# ──────────────────────────────────────────────────


@st.cache_resource
def load_resources():
    """Load embedding model and ChromaDB once. Warm up Phi-3 at startup."""
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    client      = chromadb.PersistentClient(path=CHROMA_PATH)
    collection  = client.get_collection(COLLECTION_NAME)

    # Warm-up: load Phi-3 into memory at app startup
    try:
        requests.post(OLLAMA_URL, json={
            "model": LLM_MODEL,
            "prompt": "Hi",
            "stream": False,
            "keep_alive": "30m"
        }, timeout=180)
    except Exception:
        pass

    return embed_model, collection


def ask(query: str, embed_model, collection, top_k: int = TOP_K) -> dict:
    timings    = {}
    timestamps = {}

    timestamps["query_received"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pipeline_start = time.time()

    # Stage 1: Embed
    t0 = time.time()
    query_embedding = embed_model.encode([query]).tolist()
    timings["embed_s"] = round(time.time() - t0, 4)

    # Stage 2: Retrieve
    t0 = time.time()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    timings["retrieval_s"] = round(time.time() - t0, 4)

    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    best_distance  = distances[0] if distances else 1.0
    low_confidence = best_distance > CONFIDENCE_THRESHOLD

    # Stage 3: Build prompt
    t0 = time.time()
    context = "\n\n---\n\n".join(chunks)
    prompt = f"""You are a compliance assistant for a financial institution.
Your role is to answer questions about regulatory documents accurately and completely.

INSTRUCTIONS:
1. Answer using ONLY the information explicitly stated in the CONTEXT below.
2. Do NOT infer, imply, or reason beyond what is directly written in the context.
3. If the question asks for specific figures, thresholds, or limits — state ALL that are explicitly mentioned. Do not add any that are not directly stated.
4. Keep your answer concise and factual. Do not speculate about what might be implied.
5. If multiple conditions or amounts are explicitly stated, list each one clearly.
6. At the end, cite the source clause or page number if visible in the context.
7. If the answer is genuinely not present, say exactly: "I could not find relevant information in the provided documents." Do not guess.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER (use only explicitly stated facts — no inference or speculation):"""
    timings["prompt_build_s"] = round(time.time() - t0, 4)
    prompt_chars = len(prompt)

    # Stage 4: LLM call
    timestamps["llm_call_start"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    t0 = time.time()
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": {"num_predict": NUM_PREDICT},
                "temperature": 0
            },
            timeout=180
        )
        answer = response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        answer = "ERROR: Ollama is not running. Start it with: ollama serve"
    except requests.exceptions.Timeout:
        answer = "ERROR: Request timed out. Model may be overloaded."

    timings["llm_call_s"] = round(time.time() - t0, 4)
    timestamps["llm_call_complete"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Confidence override — check LLM's own response
    NOT_FOUND_PHRASES = [
        "could not find relevant information",
        "not present in the context",
        "cannot be determined"
    ]
    answer_indicates_not_found = any(
        phrase in answer.lower() for phrase in NOT_FOUND_PHRASES
    )
    low_confidence = low_confidence or answer_indicates_not_found

    timings["total_s"] = round(time.time() - pipeline_start, 4)

    sources = [
        {"source": m["source"], "chunk_index": m["chunk_index"],
         "distance": round(d, 4), "text": c}
        for m, d, c in zip(metadatas, distances, chunks)
    ]

    return {
        "query":                      query,
        "answer":                     answer,
        "sources":                    sources,
        "best_distance":              round(best_distance, 4),
        "low_confidence":             low_confidence,
        "answer_indicates_not_found": answer_indicates_not_found,
        "timings":                    timings,
        "prompt_chars":               prompt_chars,
        "answer_chars":               len(answer),
        "chunks_retrieved":           len(chunks),
    }


def render_result_block(answer, best_distance, low_confidence,
                        sources, elapsed, timings=None, prompt_chars=None):
    st.write(answer)

    badge_col, time_col = st.columns([3, 1])
    with badge_col:
        if low_confidence:
            st.warning(f"⚠ Low confidence — best match distance: {best_distance}")
        else:
            st.success(f"✓ Confidence OK — best match distance: {best_distance}")
    with time_col:
        st.caption(f"⏱ {elapsed:.2f}s")

    if timings:
        with st.expander("⚙ Performance breakdown"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Embed",       f"{timings['embed_s']*1000:.0f} ms")
            c2.metric("Retrieve",    f"{timings['retrieval_s']*1000:.0f} ms")
            c3.metric("LLM call",    f"{timings['llm_call_s']:.2f} s")
            c4.metric("Prompt size", f"{prompt_chars:,} chars" if prompt_chars else "—")

    with st.expander("📄 View source chunks (verify citations here)"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**Source {i}** — `{s['source']}` | "
                        f"chunk #{s['chunk_index']} | distance: {s['distance']}")
            st.text(s["text"][:600] + ("..." if len(s["text"]) > 600 else ""))
            st.divider()


# ── Page config ───────────────────────────────────
st.set_page_config(
    page_title="BFSI Compliance Assistant",
    page_icon="📋",
    layout="wide"
)

# ── Session state ─────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_stats" not in st.session_state:
    st.session_state.query_stats = {
        "total": 0, "high_confidence": 0, "low_confidence": 0
    }
if "llm_times" not in st.session_state:
    st.session_state.llm_times = []

# ── Load resources ────────────────────────────────
embed_model, collection = load_resources()

# ── Header ────────────────────────────────────────
st.title("📋 BFSI Compliance Document Assistant")
st.caption(
    "Answers are grounded in retrieved source text from RBI KYC Master Direction. "
    "Always verify against source chunks before acting on any response."
)

# ── Chat history ──────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "sources" in msg:
            render_result_block(
                msg["content"],
                msg["best_distance"],
                msg["low_confidence"],
                msg["sources"],
                msg.get("elapsed", 0),
                msg.get("timings"),
                msg.get("prompt_chars")
            )
        else:
            st.write(msg["content"])

# ── Chat input ────────────────────────────────────
if query := st.chat_input("Ask a question about the KYC Master Direction..."):

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            start_time = time.time()
            result     = ask(query, embed_model, collection)
            elapsed    = time.time() - start_time

        render_result_block(
            result["answer"],
            result["best_distance"],
            result["low_confidence"],
            result["sources"],
            elapsed,
            result["timings"],
            result["prompt_chars"]
        )

    # Update session stats
    st.session_state.query_stats["total"] += 1
    if result["low_confidence"]:
        st.session_state.query_stats["low_confidence"] += 1
    else:
        st.session_state.query_stats["high_confidence"] += 1
    st.session_state.llm_times.append(result["timings"]["llm_call_s"])

    # Save to chat history
    st.session_state.messages.append({
        "role":          "assistant",
        "content":       result["answer"],
        "sources":       result["sources"],
        "best_distance": result["best_distance"],
        "low_confidence":result["low_confidence"],
        "elapsed":       elapsed,
        "timings":       result["timings"],
        "prompt_chars":  result["prompt_chars"]
    })

# ── Sidebar ───────────────────────────────────────
with st.sidebar:
    st.subheader("📊 Session Stats")
    stats = st.session_state.query_stats

    st.metric("Total Queries", stats["total"])
    col1, col2 = st.columns(2)
    col1.metric("✓ High Confidence", stats["high_confidence"])
    col2.metric("⚠ Low Confidence",  stats["low_confidence"])

    if stats["total"] > 0:
        pct = (stats["high_confidence"] / stats["total"]) * 100
        st.caption(f"{pct:.0f}% of queries answered with high confidence")

    if st.session_state.llm_times:
        avg_llm = sum(st.session_state.llm_times) / len(st.session_state.llm_times)
        st.metric("Avg LLM Time", f"{avg_llm:.1f}s")
        st.caption("First query includes model warm-up cost")

    st.divider()
    st.caption("Model: phi3:mini via Ollama")
    st.caption("Embeddings: all-MiniLM-L6-v2")
    st.caption("Vector store: ChromaDB (local)")
    st.caption("PII detection: Presidio (coming soon)")