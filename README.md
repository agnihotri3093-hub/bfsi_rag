# BFSI Compliance Document Assistant

A locally-run RAG (Retrieval-Augmented Generation) system for querying BFSI regulatory documents — built as an implementation blueprint for AI deployment in regulated environments where data residency, auditability, and answer verification are non-negotiable.


## What This Project Demonstrates

This was built as a hands-on exploration of AI implementation constraints specific to regulated industries — not as a generic RAG tutorial. The questions it was built to answer:

- What does "data never leaves the machine" actually require, architecturally?
- What does a confidence score need to measure to be trustworthy, not just present?
- Where does an LLM's output need a human-verifiable anchor, and how do you build the UI to make that anchor easy to reach, not optional?
- What is the actual latency cost structure of CPU-only local inference, and which levers genuinely move it?

**Everything runs on a local CPU-only machine. Zero API cost. Zero data leaves the device.**

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   PDF Doc   │ ──▶ │   Chunking   │ ──▶ │  Embedding   │ ──▶ │   ChromaDB   │
│  (PyMuPDF)  │     │ (paragraph + │     │ (all-MiniLM- │     │  (local,     │
│             │     │   overlap)   │     │   L6-v2)     │     │  persistent) │
└─────────────┘     └──────────────┘     └─────────────┘     └──────┬───────┘
                                                                      │
                                                                      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Streamlit  │ ◀── │  Confidence  │ ◀── │   Phi-3      │ ◀── │   Query +    │
│     UI      │     │   Scoring    │     │   Mini       │     │   Top-K      │
│             │     │ (dual-layer) │     │  (Ollama)    │     │   Retrieval  │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
```

**Stack:** Python · Streamlit · ChromaDB · sentence-transformers · Ollama (Phi-3 Mini) · PyMuPDF

---

## What Was Measured

Every performance claim below came from instrumented pipeline runs, not estimation.

| Stage | Typical latency | % of total |
|-------|-----------------|------------|
| Embed query | 25–50 ms | <0.1% |
| Retrieve (ChromaDB) | 2–60 ms | <0.1% |
| LLM generation | 7s – 165s | ~99.9% |

**Retrieval is effectively free. Generation is the entire latency story.** This single finding redirected all subsequent optimization effort away from the vector pipeline and onto prompt size and inference behavior.

---

## Known Limitations

- **Latency is not production-grade for live chat.** 80–165s per novel query on CPU is appropriate for a back-office compliance lookup tool, not a real-time customer-facing assistant. GPU inference or a hosted model would be the production fix.
- **No PII detection layer is currently wired into the pipeline.** Microsoft Presidio was evaluated for query-side PII screening (PAN, Aadhaar, phone number patterns) but is not yet integrated into `app.py`.
- **No audit logging is currently active in this version.** A CSV-based audit trail (timestamp, query, retrieved chunks, confidence, latency) was prototyped but removed from the shipped app in favor of keeping the UI layer minimal; the `ask()` function already returns every field needed to log this if re-added.
- **Chunking is not regulatory-clause-aware.** It approximates clause boundaries via paragraph splitting; it does not parse the document's actual clause/sub-clause numbering structure.

