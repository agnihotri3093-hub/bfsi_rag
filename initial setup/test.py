# save as test_setup.py in bfsi-rag folder
import fitz
from sentence_transformers import SentenceTransformer
import chromadb
import requests

# Test 1 - PDF reading
doc = fitz.open("D:/OneDrive/Desktop/bfsi_rag/rbi/kyc.pdf")
print(f"PDF OK - {len(doc)} pages loaded")

# Test 2 - Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
vector = model.encode("What is KYC?")
print(f"Embeddings OK - vector size {len(vector)}")

# Test 3 - ChromaDB
client = chromadb.Client()
collection = client.create_collection("test")
print("ChromaDB OK")

# Test 4 - Ollama
response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "phi3:mini", "prompt": "What is KYC?", "stream": False}
)
code=response.status_code
print(f"Ollama OK - Status Code: {code}, Response: {response.json()['response'][:100]}")
