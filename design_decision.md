## Key Design Decisions

### Why local inference (Ollama + Phi-3 Mini), not an API

Regulatory documents — KYC records, AML policy, credit files — cannot leave client infrastructure in most BFSI engagements. Running inference locally via Ollama eliminates this constraint entirely: no document text, query, or generated answer ever crosses the network boundary. Phi-3 Mini (3.7GB) was selected specifically because it runs on CPU-only hardware within a 15GB RAM constraint — the kind of machine a compliance team would realistically be issued, not a GPU workstation.

### Why source chunks are always shown alongside the answer

This is the single most important design decision in the project, and it came directly from a debugging finding (see Learnings below): the LLM can state a fact correctly while citing the wrong source for it. A generated answer is not a substitute for the source document — it is a navigation aid to it. The UI never lets a user see an answer without also being one click away from the raw text that produced it.

### Why confidence scoring is dual-layer, not single-metric

Vector distance alone is an unreliable confidence signal. A query can retrieve a *semantically related but factually insufficient* chunk — meaning the distance score is good even though the LLM has nothing to actually answer with. The system runs two independent checks and treats either one failing as low confidence:

1. **Retrieval confidence** — is the best-matching chunk's cosine distance below threshold (0.55)?
2. **Generation confidence** — does the LLM's own answer contain a "could not find" signal?

```python
low_confidence = (best_distance > CONFIDENCE_THRESHOLD) or answer_indicates_not_found
```

### Why temperature=0

Compliance Q&A has no room for creative variation. Two identical queries should produce the same answer. Default sampling temperature introduced answer instability — the same query sometimes returned a complete, correct answer and sometimes an incomplete one. Setting `temperature=0` made generation deterministic and was the single highest-leverage fix in the project for answer reliability.

### Why paragraph-level chunking with overlap, not fixed-size chunking

Regulatory documents are structured around numbered clauses with sub-clauses (a, b, c...) that depend on shared context (an `Explanation:` note, a parent clause heading). Naive fixed-size chunking splits these mid-definition. Paragraph-boundary chunking with overlap keeps related sub-clauses together more often, at the cost of variable chunk size — a deliberate trade-off favoring semantic integrity over uniform chunk length.

