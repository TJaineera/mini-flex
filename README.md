# mini-flex — Local-First Tiny RAG

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Local LLM](https://img.shields.io/badge/LLM-Ollama-black.svg)]()

A minimal, *newbie-friendly* Retrieval-Augmented Generation (RAG) starter:

-  **Local model** via [Ollama](https://ollama.com/) (OpenAI-compatible API)
-  **Embeddings**: fast **dummy** embedder for instant setup or real **SentenceTransformers**
-  **Vector DB**: **FAISS** (CPU)
-  **UI**: lightweight **Streamlit** chat with file upload, rebuild index, and citations
-  **Modular design** — clear folders for `llms/`, `embeddings/`, `vectorstore/`, `rag/`, and `ui/`

---

##  Quickstart

```powershell
# 1) Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt
# or minimal:
pip install httpx python-dotenv faiss-cpu tqdm streamlit

# 3) (optional) Real embeddings
# pip install sentence-transformers

# 4) (recommended) Local LLM via Ollama
# Download: https://ollama.com/download/windows
ollama pull llama3.2:1b

# 5) Copy environment example
copy .env.example .env
notepad .env

# 6) Build the vector index
python build_index.py

# 7) Ask a question in CLI
python ask_rag.py

# 8) Run the browser UI
streamlit run mini_flex/ui/app.py

No API key required with Ollama.
Switch to OpenAI/Azure anytime by editing .env.

Architecture

[data/*.txt] ──chunk──▶ [embeddings] ──add──▶ [FAISS]
                                   │
                                   └─search (top-k)──▶ [context] + [question]
                                                        │
                                                        ▼
                                        OpenAI-compatible Chat API (Ollama)
                                                        │
                                                        ▼
                                                 answer + citations

Core modules

| Layer           | File                                   | Description                             |
| --------------- | -------------------------------------- | --------------------------------------- |
|  LLM            | `mini_flex/llms/openai_like.py`        | universal wrapper (Ollama/OpenAI/Azure) |
|  Embeddings     | `mini_flex/embeddings/hf_embed.py`     | dummy or SentenceTransformers           |
|  Vector Store   | `mini_flex/vectorstore/faiss_store.py` | FAISS index, cosine similarity          |
|  Chunking       | `mini_flex/utils/chunk.py`             | split long docs with overlap            |
|  Config         | `mini_flex/utils/config.py`            | loads `.env`                            |
|  RAG Pipeline   | `mini_flex/rag/pipeline.py`            | retrieve → augment → generate           |
|  UI             | `mini_flex/ui/app.py`                  | Streamlit chat, upload, rebuild         |

Environment

Copy and adjust:

OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3.2:1b
DATA_DIR=data
MEMORY_PATH=chat_memory.json

For OpenAI platform:

OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

Environment

Copy and adjust:

OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=llama3.2:1b
DATA_DIR=data
MEMORY_PATH=chat_memory.json


For OpenAI platform:

OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

🔄 Switching to Real Embeddings

Install:

pip install sentence-transformers


Replace mini_flex/embeddings/hf_embed.py with:

from sentence_transformers import SentenceTransformer
from typing import List

class HFEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]):
        return self.model.encode(texts, normalize_embeddings=True)


Rebuild index:

python build_index.py

Common Commands
# add new text files
python build_index.py

# ask via CLI
python ask_rag.py

# run Streamlit app
streamlit run mini_flex/ui/app.py

Troubleshooting
Issue	Fix
401 Unauthorized	Use a valid OpenAI key or Ollama
ModuleNotFoundError: faiss	pip install faiss-cpu
SentenceTransformers slow first run	Downloads model once (400MB)
Ollama connection error	ollama serve then check http://localhost:11434/api/tags
