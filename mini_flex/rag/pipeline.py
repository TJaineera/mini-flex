import pickle, faiss
from mini_flex.embeddings.hf_embed import HFEmbeddings
from mini_flex.vectorstore.faiss_store import FaissStore
from mini_flex.llms.openai_like import OpenAILike

PROMPT_TEMPLATE = """You answer using ONLY the provided context. If unsure, say you don't know.

Question: {question}

Context:
{context}

Answer:"""

class SimpleRAG:
    def __init__(self, store_path: str = "faiss_store.pkl"):
        with open(store_path, "rb") as f:
            data = pickle.load(f)
        # embedding model = MiniLM-L6-v2 (384 dims)
        self.emb = HFEmbeddings()
        self.store = FaissStore(dim=384)
        self.store.docs = data["docs"]
        self.store.index = faiss.deserialize_index(data["index"])
        # uses your .env (Ollama/OpenAI) via OpenAILike
        self.llm = OpenAILike()

    def ask(self, question: str, k: int = 3) -> str:
        qv = self.emb.embed_texts([question])
        hits = self.store.search(qv, k=k)
        context = "\n\n---\n\n".join([h[2] for h in hits]) if hits else "No context found."
        prompt = PROMPT_TEMPLATE.format(question=question, context=context)
        return self.llm.chat([
            {"role": "system", "content": "You are a careful cybersecurity assistant."},
            {"role": "user", "content": prompt}
        ])
