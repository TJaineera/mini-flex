from typing import Optional, List, Dict
import httpx
from mini_flex.utils.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL

class OpenAILike:
    """
    Minimal wrapper for any OpenAI-compatible Chat Completions API.
    Reads base URL, key, and default model from .env via config.py
    """
    def __init__(self, model: Optional[str] = None, timeout: int = 60):
        self.model = model or OPENAI_MODEL
        self.client = httpx.Client(
            base_url=OPENAI_BASE_URL,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            timeout=timeout
        )

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        messages = [{"role": "system"|"user"|"assistant", "content": "..."}]
        returns assistant text
        """
        resp = self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
