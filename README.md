\# mini-flex (LLMFlex-style tiny RAG)



Local-first RAG starter:

\- LLM via OpenAI-compatible API (Ollama by default)

\- Dummy embeddings (fast) or SentenceTransformers (swap later)

\- FAISS vector store

\- Simple RAG pipeline



\## Quickstart

```powershell

python -m venv .venv

.\\.venv\\Scripts\\activate

pip install -r requirements.txt  # (optional if you make one)

\# or pip install httpx python-dotenv faiss-cpu tqdm streamlit sentence-transformers

\# (or use dummy embeddings already in repo)



\# Run local model

ollama run llama3.2:1b   # first time pulls the model



\# Build index

python build\_index.py



\# Ask RAG

python ask\_rag.py



