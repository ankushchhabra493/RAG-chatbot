from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')

class EmbedRequest(BaseModel):
    text: str

@app.post("/embed")
async def embed(req: EmbedRequest):
    emb = model.encode([req.text])[0]
    return {"embedding": emb.tolist()}

# Make sure this file is named exactly 'embed_server.py' (not 'embed_server.py.txt' or similar)
# And run uvicorn from the directory containing this file:
#   uvicorn embed_server:app --host 0.0.0.0 --port 9000
