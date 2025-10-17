import pickle, faiss
from mini_flex.embeddings.hf_embed import HFEmbeddings
from mini_flex.vectorstore.faiss_store import FaissStore

with open("faiss_store.pkl", "rb") as f:
    data = pickle.load(f)

emb = HFEmbeddings()
store = FaissStore(dim=384)  # 384 for MiniLM-L6-v2
store.docs = data["docs"]
store.index = faiss.deserialize_index(data["index"])

q = "How does RAG help reduce hallucinations?"
qv = emb.embed_texts([q])
hits = store.search(qv, k=2)
for i, score, text in hits:
    print(f"[{score:.3f}] {text[:80]}...")
