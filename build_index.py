import os, glob, pickle, time
from tqdm import tqdm
from mini_flex.embeddings.hf_embed import HFEmbeddings
from mini_flex.vectorstore.faiss_store import FaissStore
from mini_flex.utils.config import DATA_DIR
from mini_flex.utils.chunk import chunk_text
import faiss

os.makedirs(DATA_DIR, exist_ok=True)

# Gather text files
files = glob.glob(f"{DATA_DIR}/*.txt")

docs = []
start_time = time.time()

if files:
    print(f"🧾 Found {len(files)} file(s) — chunking and indexing...")
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        chunks = chunk_text(raw, chunk_size=900, overlap=150)
        for ch in chunks:
            docs.append(f"[FILE: {os.path.basename(fp)}]\n{ch}")
else:
    print("⚠️ No .txt files found in data/. Using fallback sample content.")
    seed = [
        "Zero Trust requires continuous verification and least-privilege access.",
        "Indicators of Compromise include suspicious IPs, hashes, and domains.",
        "RAG combines vector search with LLMs to ground answers in documents."
    ]
    for s in seed:
        docs.append(f"[FILE: sample.txt]\n{s}")

print(f"📚 Total chunks to embed: {len(docs)}")

# Create embeddings with progress bar
emb = HFEmbeddings()
batch_size = 16
vectors = []
for i in tqdm(range(0, len(docs), batch_size), desc="Embedding texts"):
    batch = docs[i:i+batch_size]
    vecs = emb.embed_texts(batch)
    vectors.append(vecs)
import numpy as np
vectors = np.vstack(vectors)

# Build FAISS index
store = FaissStore(dim=vectors.shape[1])
store.add(vectors, docs)

with open("faiss_store.pkl", "wb") as f:
    pickle.dump({"docs": store.docs, "index": faiss.serialize_index(store.index)}, f)

elapsed = time.time() - start_time
print(f"\n✅ Indexed {len(docs)} chunks from {len(files) or 1} file(s).")
print(f"⏱️ Finished in {elapsed:.2f} seconds.")
