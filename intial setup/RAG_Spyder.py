import fitz
import chromadb
from sentence_transformers import SentenceTransformer
import re, os
import requests
import importlib
import time
from datetime import datetime

packages = ["fitz", "chromadb", "sentence_transformers", 
            "streamlit", "presidio_analyzer", "presidio_anonymizer", 
            "spacy", "requests"]
for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f"{pkg}: OK")
    except ImportError:
        print(f"{pkg}: MISSING")

pdf_path = "D:/OneDrive - Sutra Management/Desktop/bfsi_rag/rbi/kyc.pdf"
doc = fitz.open(pdf_path)
full_text = ""
for page_num, page in enumerate(doc):
    full_text += f"\n[PAGE {page_num+1}]\n{page.get_text()}"
doc.close()

print(f"Total characters: {len(full_text)}")
print(full_text[:500])  # preview first 500 chars


def chunk_text(text, chunk_size=400, overlap=80):
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk += ("\n\n" if current_chunk else "") + para
        else:
            if current_chunk:
                chunks.append(current_chunk)
            carry = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = carry + ("\n\n" if carry else "") + para
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

chunks = chunk_text(full_text)

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks, show_progress_bar=True)

client = chromadb.PersistentClient(path="chroma_store")
try:
    client.delete_collection("bfsi_docs")
except:
    pass
collection = client.get_or_create_collection("bfsi_docs", metadata={"hnsw:space": "cosine"})

ids = [f"kyc_chunk_{i}" for i in range(len(chunks))]
metadatas = [{"source": "kyc_guidelines.pdf", "chunk_index": i} for i in range(len(chunks))]

collection.add(ids=ids, embeddings=embeddings.tolist(), documents=chunks, metadatas=metadatas)
print("Stored successfully")

query = "What percentage of ownership defines a beneficial owner in a trust under this Master Direction?"
query_embedding = model.encode([query]).tolist()

results = collection.query(query_embeddings=query_embedding, n_results=3, 
                            include=["documents", "metadatas", "distances"])

for i in range(len(results["documents"][0])):
    print(f"Distance: {results['distances'][0][i]:.4f}")
    print(results["documents"][0][i][:200])
    print("---"*30)
    
def clean_chunk(text):
    text = re.sub(r'\[PAGE \d+\]', '', text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    return text.strip()

context = "\n\n---\n\n".join(clean_chunk(c) for c in results["documents"][0])
prompt = f"""You are a compliance assistant. Answer using ONLY this context.
If not found, say so.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

response = requests.post("http://localhost:11434/api/generate", 
                          json={"model": "phi3:mini", "prompt": prompt, "stream": False},
                          timeout=400)
print(response.json()["response"])



# ── Config (validated values) ────────────────────
CHROMA_PATH = "chroma_store"
COLLECTION_NAME = "bfsi_docs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "phi3:mini"
CONFIDENCE_THRESHOLD = 0.55

def ask(query: str, top_k: int = TOP_K) -> dict:
    """
    Full RAG pipeline as a single callable function.
    Assumes `model` and `collection` already exist in notebook scope.
    """
    timings = {}
    timestamps = {}

    # ── Session start ──────────────────────────────────
    timestamps["query_received"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pipeline_start = time.time()

    # ── Stage 1: Embed query ───────────────────────────
    t0 = time.time()
    query_embedding = model.encode([query]).tolist()
    timings["embed_s"] = round(time.time() - t0, 4)
    timestamps["embed_complete"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Stage 2: Retrieve chunks ───────────────────────
    t0 = time.time()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    timings["retrieval_s"] = round(time.time() - t0, 4)
    timestamps["retrieval_complete"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    best_distance = distances[0] if distances else 1.0
    best_chunk = chunks[0] if chunks else ''
    low_confidence = best_distance > CONFIDENCE_THRESHOLD

    # ── Stage 3: Build prompt ──────────────────────────
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

ANSWER (be complete — include all relevant figures, conditions, and limits mentioned in the context):"""
    timings["prompt_build_s"] = round(time.time() - t0, 4)
    prompt_chars = len(prompt)

    # ── Stage 4: LLM call ──────────────────────────────
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
                "options": {"num_predict": 250}
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

    # ── Confidence override: LLM self-report ───────────
    NOT_FOUND_PHRASES = [
        "could not find relevant information",
        "not present in the context",
        "cannot be determined"
    ]
    answer_indicates_not_found = any(
        phrase in answer.lower() for phrase in NOT_FOUND_PHRASES
    )
    low_confidence = low_confidence or answer_indicates_not_found

    # ── Stage 5: Wrap up ───────────────────────────────
    timings["total_s"] = round(time.time() - pipeline_start, 4)
    timestamps["pipeline_complete"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sources = [
        {"source": m["source"], "chunk_index": m["chunk_index"],
         "distance": round(d, 4), "text": c}
        for m, d, c in zip(metadatas, distances, chunks)
    ]

    return {
        # Core output
        "query": query,
        "answer": answer,
        "sources": sources,
        "best_distance": round(best_distance, 4),
        "best_chunk": best_chunk,
        "low_confidence": low_confidence,
        "answer_indicates_not_found": answer_indicates_not_found,

        # Audit fields
        "timestamps": timestamps,
        "timings": timings,
        "prompt_chars": prompt_chars,
        "answer_chars": len(answer),
        "chunks_retrieved": len(chunks),
        "top_k_requested": top_k,
        "model": LLM_MODEL,
        "pii_detected": False  # placeholder — Presidio integration goes here later
    }


# print(f"\nLow confidence: {result['low_confidence']}")
# print(f"timings:          {result['timings']}")
# print(f"prompt_chars:     {result['prompt_chars']}")
# print(f"answer_chars:     {result['answer_chars']}")
# print(f"chunks_retrieved: {result['chunks_retrieved']}")
# print(f"top_k_requested:  {result['top_k_requested']}")
# print(f"model:            {result['model']}")
# print(f"low_confidence:   {result['low_confidence']}")
# print(f"best_distance:    {result['best_distance']}")
# print(f"best_chunk:       {result['best_chunk']}"[:100])
# print(f"answer:           {result['answer']}")

result = ask("What is the monetary limit for opening a Small Account?")
print(result["answer"])
print(f"timings:          {result['timings']}")

result = ask("What percentage of ownership defines a beneficial owner in a trust?")
print(result["answer"])
print(f"timings:          {result['timings']}")
