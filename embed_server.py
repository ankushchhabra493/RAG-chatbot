from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

app = FastAPI()

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")
try:
    # Try to get existing collection
    collection = client.get_or_create_collection(
        name="pubmed-central",
        metadata={"description": "PubMed articles collection"}
    )
    print(f"Connected to collection 'pubmed-central'")
except Exception as e:
    print(f"Error initializing collection: {e}")
    raise

class TextRequest(BaseModel):
    text: str

@app.post("/embed")
async def create_embedding(request: TextRequest):
    # Generate embedding using sentence-transformer
    embedding = model.encode(request.text)
    
    # Print detailed embedding information
    print("\n=== Embedding Details ===")
    print(f"Input text: {request.text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 10 values: {embedding[:10]}")
    print(f"Min value: {np.min(embedding):.6f}")
    print(f"Max value: {np.max(embedding):.6f}")
    print(f"Mean value: {np.mean(embedding):.6f}")
    print("=====================\n")
    
    return {"embedding": embedding.tolist()}

def get_medical_category(query):
    """Determine medical category from query text"""
    query_lower = query.lower()
    if any(word in query_lower for word in ["asthma", "breathing", "inhaler"]):
        return "respiratory"
    elif any(word in query_lower for word in ["diabetes", "blood sugar"]):
        return "diabetes"
    elif any(word in query_lower for word in ["heart", "cardiac", "blood pressure"]):
        return "cardiology"
    return None

class QueryRequest(BaseModel):
    text: str
    embedding: List[float]
    n_results: int = 3

@app.post("/query")
async def query_similar(request: QueryRequest):
    print(f"\n=== Searching documents for: {request.text} ===")
    try:
        results = collection.query(
            query_embeddings=[request.embedding],
            n_results=request.n_results,
            include=["metadatas", "distances", "documents"]
        )
        
        if not results["ids"]:
            print("No matching documents found")
            return {"results": [], "count": 0}
            
        print(f"Found {len(results['documents'][0])} matching documents")
        return {
            "results": results["documents"][0],
            "count": len(results["documents"][0])
        }
            
    except Exception as e:
        print(f"Error during search: {e}")
        return {"results": [], "count": 0}

@app.post("/add_documents")
async def add_documents(documents: List[str]):
    try:
        # Generate embeddings for documents
        embeddings = model.encode(documents).tolist()
        
        # Add to ChromaDB with sequential IDs
        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=[f"doc{i}" for i in range(len(documents))]
        )
        return {"status": "success", "count": len(documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Make sure this file is named exactly 'embed_server.py' (not 'embed_server.py.txt' or similar)
# And run uvicorn from the directory containing this file:
#   uvicorn embed_server:app --host 0.0.0.0 --port 9000

# Explanation:
# We use the sentence-transformers library to load a pre-trained model ('all-MiniLM-L6-v2').
# When a POST request is made to /embed with a JSON payload {"text": "..."},
# the server encodes the text into a vector (embedding) using the model.
# The embedding is returned as a list of floats in the response.
# This allows the Go backend to get vector representations for any input text via HTTP.
