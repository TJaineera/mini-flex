from typing import List

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    text = " ".join(text.split())
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        # extend to last space so we don't cut words in half
        if end < n:
            space = text.rfind(" ", start, end)
            if space != -1 and space > start + 200:  # avoid tiny fragments
                end = space
        chunks.append(text[start:end].strip())
        start = max(end - overlap, 0)
        if start == 0 and end == n:
            break
    return [c for c in chunks if c]
