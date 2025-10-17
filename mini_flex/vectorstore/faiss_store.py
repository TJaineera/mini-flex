import faiss
import numpy as np
from typing import List, Tuple

class FaissStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # cosine if vectors normalized
        self.docs: List[str] = []

    def add(self, vectors: np.ndarray, texts: List[str]):
        self.index.add(vectors.astype(np.float32))
        self.docs.extend(texts)

    def search(self, query_vec: np.ndarray, k: int = 3) -> List[Tuple[int, float, str]]:
        D, I = self.index.search(query_vec.astype(np.float32), k)
        results = []
        for i, score in zip(I[0], D[0]):
            if i == -1:
                continue
            results.append((int(i), float(score), self.docs[i]))
        return results
