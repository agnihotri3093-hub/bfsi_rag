import fitz  # PyMuPDF to read pdf and convert to text
import chromadb # ChromaDB client for vector storage
from sentence_transformers import SentenceTransformer #transform text to vector embeddings
import os # for file/exception handling
import re # for splitting text into paragraphs (not essential but helps create more natural chunks)

# ── Config ────────────────────────────────────────────────
DOCS_FOLDER = "D:/OneDrive/Desktop/bfsi_rag/rbi" #location of source PDF (directory)
CHROMA_PATH = "D:/OneDrive/Desktop/bfsi_rag/chroma_store" #location of chroma db physical files (directory)
COLLECTION_NAME = "bfsi_docs" #name of colection in chroma db to store vectors (string)
CHUNK_SIZE = 400        # characters per chunk (approx 1-2 paragraphs)
CHUNK_OVERLAP = 80      # characters of overlap between chunks
EMBED_MODEL = "all-MiniLM-L6-v2" #selection of sentence transformer model for embeddings (string)
# ─────────────────────────────────────────────────────────


def extract_text_from_pdf(pdf_path: str) -> str: #input is file path and output is PDF content as string
    """Extract full text from a PDF file."""
    doc = fitz.open(pdf_path) #open PDf using fitz
    full_text = "" #initialize empty string
    for page_num, page in enumerate(doc): #loop & enumeration = page no + page object
        text = page.get_text() # extract text from page
        # Tag each page so we can trace chunks back to source later
        full_text += f"\n[PAGE {page_num + 1}]\n{text}" #persist page no. in full string for chunk to back reference
    doc.close() #free up memory
    return full_text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]: #input is string, chunk size + overlap size has default values)
    """
    Split text into overlapping chunks.
    Tries to split at paragraph boundaries first; falls back to character count.
    """
    # Split on double newlines (paragraph boundaries)
    paragraphs = re.split(r'\n\s*\n', text) #re module(pattern,input), r' means raw text no escaping required, s* means any whitespace-tab, space, new line * means multiple times
    paragraphs = [p.strip() for p in paragraphs if p.strip()] #expression - remove whitespace | loop | condition= non null

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        # If adding this paragraph keeps us under limit, add it
        if len(current_chunk) + len(para) + 2 <= chunk_size: #+2 for \n\n
            current_chunk += ("\n\n" if current_chunk else "") + para #if condition ensures if no current chunk do not add \n\n
        else:
            # Save current chunk if it has content
            if current_chunk:
                chunks.append(current_chunk)
            # Start new chunk with overlap from end of previous
            if len(current_chunk) > overlap:
                carry = current_chunk[-overlap:]
            else:
                carry = current_chunk
            current_chunk = carry + ("\n\n" if carry else "") + para

    if current_chunk:
        chunks.append(current_chunk) # to save final unfinished chunk

    return chunks


def ingest_document(pdf_path: str):
    """Full pipeline: PDF → chunks → embeddings → ChromaDB."""
    filename = os.path.basename(pdf_path)
    print(f"\n{'='*50}")
    print(f"Ingesting: {filename}")
    print(f"{'='*50}")

    # Step 1: Extract text
    print("Step 1/4 — Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path) #Macro to extract text from PDF using fitz
    print(f"  Extracted {len(text):,} characters")

    # Step 2: Chunk
    print("Step 2/4 — Chunking text...")
    chunks = chunk_text(text) #Macro to split in chunk
    print(f"  Created {len(chunks)} chunks (size ~{CHUNK_SIZE} chars, overlap {CHUNK_OVERLAP})")

    # Step 3: Embed
    print("Step 3/4 — Generating embeddings (this takes a minute first time)...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(chunks, show_progress_bar=True)

    # ── DETAILED INSPECTION ────────────────────────────────────────────────
    print(f"\n[EMBEDDING INFO]")
    print(f"  • Data Type (Object Class): {type(embeddings)}")
    print(f"  • Matrix Shape: {embeddings.shape}")
    print(f"  • Total Chunks Embedded: {embeddings.shape[0]}")
    print(f"  • Vector Dimensions per Chunk: {embeddings.shape[1]}")
    print(f"  • Numerical Data Type: {embeddings.dtype}")
    
   # ── INSPECT 3 CHUNKS AND THEIR VECTORS ──────────────────────────────────
    print(f"\n{'─'*20} MULTI-CHUNK INSPECTION {'─'*20}")
    
    # Determine how many chunks we actually have (in case the doc is tiny)
    num_to_show = min(3, len(chunks))
    
    for i in range(num_to_show):
        print(f"\n👉 [CHUNK {i + 1} / {len(chunks)}]")
        
        # 1. Print a snippet of the actual text
        text_snippet = chunks[i].replace('\n', ' ')[:120]  # First 120 chars on one line
        print(f"  • Text Snippet : \"{text_snippet}...\"")
        
        # 2. Grab and print the vector info for this specific row
        current_vector = embeddings[i]
        print(f"  • Vector Shape : {current_vector.shape} (Dimensions)")
        print(f"  • Vector Sample: {current_vector[:5].tolist()} ...") 
        print(f"  • Value Range  : Min ({current_vector.min():.4f}) to Max ({current_vector.max():.4f})")
    
    print(f"{'─'*64}\n")

    # Step 4: Store in ChromaDB
    print("Step 4/4 — Storing in ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"})  # cosine similarity Because Cosine Similarity only looks at the angle $\theta$, it completely ignores the length of the vector. This means a 3-sentence chunk and a 10-sentence chunk about the exact same topic (e.g., "How to open a corporate bank account") will have an angle close to $0^\circ$ (Cosine value near $1.0$), signaling a near-perfect match, despite their differing lengths.
     # Collection for this doc to allow re-ingestion
    try:
        
        if collection.get(where={"source": filename})["ids"]:
            print("Collection already exists")
            return 0
    except Exception:
        pass
    # Build metadata for each chunk
    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(), #translates that complex NumPy memory matrix back into a standard Python list of lists of floats.
        documents=chunks,
        metadatas=metadatas
    )

    print(f"\nDone. {len(chunks)} chunks stored in ChromaDB at '{CHROMA_PATH}/'")
    print(f"Collection: '{COLLECTION_NAME}'")
    return len(chunks)


if __name__ == "__main__":
    pdf_path = os.path.join(DOCS_FOLDER, "kyc.pdf")

    if not os.path.exists(pdf_path):
        print(f"ERROR: File not found — {pdf_path}")
    else:
        chunk_count = ingest_document(pdf_path)
        print(f"\nIngestion complete. {chunk_count} chunks ready for retrieval.")
