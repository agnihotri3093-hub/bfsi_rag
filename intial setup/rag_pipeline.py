import requests
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath("D:\OneDrive\Desktop\bfsi_rag\retriever.py"))) # add retriever.py's directory to path
from retriever import retrieve
from datetime import datetime
import csv
import os

# ── Config ────────────────────────────────────────────────
OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "phi3:mini"
TOP_K         = 3       # chunks to retrieve — 3 is enough for Phi-3's context
AUDIT_LOG     = "D:/OneDrive - Sutra Management/Desktop/bfsi_rag/audit_log.csv"
# ─────────────────────────────────────────────────────────


def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Assembles the full prompt sent to Phi-3.
    Injects retrieved chunks as context.
    Instructs LLM to answer only from context — no hallucination.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, 1):
        context_blocks.append(
            f"[Source {i}: {chunk['source']} — Page area {chunk['chunk_index']}]\n"
            f"{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are a BFSI compliance assistant. Answer the question using ONLY 
the context provided below. If the answer is not in the context, say 
"I could not find this in the provided documents" — do not guess.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""
    return prompt


def query_ollama(prompt: str) -> str:
    """Send prompt to Phi-3 via Ollama and return response text."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        return "ERROR: Ollama is not running. Start it with 'ollama serve' in terminal."
    except Exception as e:
        return f"ERROR: {str(e)}"


def log_audit(query: str, chunks: list[dict], answer: str):
    """Append one row to audit log CSV."""
    file_exists = os.path.isfile(AUDIT_LOG)
    with open(AUDIT_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow([
                "Timestamp", "Query", "Chunks Retrieved",
                "Top Source", "Top Distance", "Answer"
            ])
        writer.writerow([
            datetime.utcnow().isoformat(),
            query,
            len(chunks),
            chunks[0]["source"] if chunks else "none",
            chunks[0]["distance"] if chunks else "none",
            answer[:500]   # truncate long answers in log
        ])


def ask(query: str, verbose: bool = True) -> dict:
    """
    Full RAG pipeline:
    query → retrieve chunks → build prompt → ask Phi-3 → log → return
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print(f"{'='*60}")

    # Step 1 — Retrieve
    chunks = retrieve(query, top_k=TOP_K)

    if verbose:
        print(f"\n[RETRIEVAL] {len(chunks)} chunks retrieved")
        for i, c in enumerate(chunks, 1):
            print(f"  Rank {i} | Distance: {c['distance']} | {c['source']} chunk #{c['chunk_index']}")

    # Step 2 — Build prompt
    prompt = build_prompt(query, chunks)

    if verbose:
        print(f"\n[PROMPT] Sending {len(prompt)} characters to {OLLAMA_MODEL}...")

    # Step 3 — Query Phi-3
    answer = query_ollama(prompt)

    # Step 4 — Log audit entry
    log_audit(query, chunks, answer)

    if verbose:
        print(f"\n[ANSWER]")
        print("-" * 60)
        print(answer)
        print("-" * 60)
        print(f"\n[SOURCES USED]")
        for i, c in enumerate(chunks, 1):
            print(f"  {i}. {c['source']} — chunk #{c['chunk_index']} (distance: {c['distance']})")

    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "distance": c["distance"]
            }
            for c in chunks
        ]
    }


if __name__ == "__main__":
    # Test with the same three queries from retrieval validation
    test_queries = [
        "What documents are required for KYC verification?",
        "What are the rules for politically exposed persons?",
        "How often should KYC be updated for high risk customers?"
    ]

    for query in test_queries:
        result = ask(query)
        print("\n")
