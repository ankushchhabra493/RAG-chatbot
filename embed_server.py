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

@app.post("/query")
async def query_similar(request: TextRequest):
    # Get embedding for query text
    query_embedding = model.encode(request.text)
    print(f"\nQuerying with text: {request.text}")
    print(f"Generated embedding shape: {query_embedding.shape}")
    
    # Query ChromaDB collection
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )
    
    # Format response properly
    documents = results["documents"][0] if results["documents"] else []
    return {
        "results": documents,
        "count": len(documents)
    }

class QueryRequest(BaseModel):
    query_embeddings: List[List[float]]
    n_results: int = 3

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
