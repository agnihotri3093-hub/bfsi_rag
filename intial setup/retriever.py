import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────
CHROMA_PATH = "D:/OneDrive - Sutra Management/Desktop/bfsi_rag/chroma_store"
COLLECTION_NAME = "bfsi_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5  # how many chunks to retrieve per query
# ─────────────────────────────────────────────────────────

# Lazy loading singleton pattern
_model = None
_collection = None

def _load():
    global _model, _collection #assign global variables
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(COLLECTION_NAME)

def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Takes a plain English query.
    Returns a list of dicts: {text, source, chunk_index, distance}
    """
    _load()

    query_embedding = _model.encode([query]).tolist()

    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"] #required by LLM
    )

    output = []
    for i in range(len(results["documents"][0])): #[0] as chroma can search for mulitple queries at once, and we need single result
        output.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "distance": round(results["distances"][0][i], 4)
        })

    return output


if __name__ == "__main__":
    # Validation test — run 3 KYC-relevant queries and inspect results
    test_queries = [
        "What documents are required for KYC verification?",
        "What are the rules for politically exposed persons?",
        "How often should KYC be updated for high risk customers?"
    ]

    print("=" * 60)
    print("RETRIEVAL VALIDATION")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQUERY: {query}")
        print("-" * 50)
        chunks = retrieve(query)
        for rank, chunk in enumerate(chunks, 1):
            print(f"  Rank {rank} | Distance: {chunk['distance']} | Chunk #{chunk['chunk_index']}")
            print(f"  Preview: {chunk['text'][:200].strip()}...")
            print()