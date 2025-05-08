# RAG Chatbot Setup & Run Instructions

## 1. Clone the repository

```sh
git clone https://github.com/ankushchhabra493/RAG-chatbot.git
cd RAG-chatbot
```

## 2. Install Go dependencies

```sh
go mod tidy
```

## 3. (Recommended) Create a Python virtual environment

```sh
python3 -m venv rag-venv
source rag-venv/bin/activate
```

## 4. Install Python dependencies for the embedding server

```sh
pip install fastapi uvicorn sentence-transformers
```

## 5. Start the Python embedding server

```sh
uvicorn embed_server:app --host 0.0.0.0 --port 9000
```

## 6. Start ChromaDB

If you don't have ChromaDB, install it:

```sh
pip install chromadb
```

Then start ChromaDB (in a new terminal):

```sh
chroma run
```

## 7. Start the Go backend

```sh
go run main.go
```

## 8. Open the frontend

Open your browser and go to:

```
http://localhost:8080
```

## Notes

- **You must re-ingest any required data into ChromaDB** (the `chroma/` directory is not included in the repo).
- **Do not commit your virtual environment or ChromaDB data to git.**
- **API keys and secrets are not included.** Set them as environment variables if needed.

---
