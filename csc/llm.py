import os
import requests


class BaseLLM:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL")

        if not self.api_key or not self.base_url or not self.model:
            raise ValueError(
                "Set LLM_API_KEY, LLM_BASE_URL and LLM_MODEL before using "
                "OpenAICompatibleLLM."
            )

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Answer the task using the supplied compiled state. "
                            "Do not invent unsupported facts."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
