import os
import re
import time
from typing import Optional
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class BaseLLM:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class MockLLM(BaseLLM):
    """
    Offline Mock LLM for testing evaluation pipelines without API keys.
    Extracts answer candidates from the prompt context.
    """
    def __init__(self, answer_map: Optional[dict] = None):
        self.answer_map = answer_map or {}

    def generate(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        for query, ans in self.answer_map.items():
            if query.lower() in prompt_lower and ans.lower() in prompt_lower:
                return ans

        match = re.search(r"COMPILED COGNITIVE STATE:\s*\n*(.*)", prompt, re.DOTALL)
        if match:
            state_text = match.group(1).strip()
            first_line = state_text.split("\n")[0]
            return first_line[:50]
        return "Unknown"


class OpenAICompatibleLLM(BaseLLM):
    """
    OpenAI-compatible LLM client supporting OpenAI, Groq, Gemini, and Ollama
    with automatic rate-limit backoff and retries.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 5,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
            raise ValueError(
                "LLM_API_KEY is not set. Please set LLM_API_KEY and LLM_BASE_URL, "
                "or pass api_key to OpenAICompatibleLLM."
            )

    def generate(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert question-answering agent. "
                        "Answer the question directly and accurately based ONLY on the provided "
                        "COMPILED COGNITIVE STATE. Keep your answer extremely concise (an entity, name, date, or short phrase)."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }

        retries = 0
        while retries < self.max_retries:
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code == 429:
                    retries += 1
                    err_msg = response.text
                    wait_match = re.search(r"try again in ([\d\.]+)s", err_msg)
                    wait_time = float(wait_match.group(1)) + 0.5 if wait_match else (2.0 * retries)
                    time.sleep(wait_time)
                    continue

                if not response.ok:
                    try:
                        err_data = response.json()
                        msg = err_data.get("error", {}).get("message", response.text)
                    except Exception:
                        msg = response.text
                    raise RuntimeError(f"LLM API Error ({response.status_code}): {msg}")

                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                # Strip out reasoning tokens like <think>...</think> if present
                clean_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return clean_content or content

            except (requests.RequestException, RuntimeError) as e:
                retries += 1
                if retries >= self.max_retries:
                    raise e
                time.sleep(1.5 * retries)

        raise RuntimeError("Max retries exceeded for LLM generation.")
