import os
import re
from typing import Optional

_ACTIVE_BACKEND = os.getenv("CSC_TOKENIZER_BACKEND", "qwen").lower()
_QWEN_TOKENIZER = None
_TIKTOKEN_ENCODING = None


def _get_qwen_tokenizer():
    global _QWEN_TOKENIZER
    if _QWEN_TOKENIZER is None:
        try:
            from tokenizers import Tokenizer
            # Load Qwen2.5 / Qwen3 BPE tokenizer
            _QWEN_TOKENIZER = Tokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        except Exception:
            _QWEN_TOKENIZER = False
    return _QWEN_TOKENIZER if _QWEN_TOKENIZER is not False else None


def _get_tiktoken_encoding():
    global _TIKTOKEN_ENCODING
    if _TIKTOKEN_ENCODING is None:
        try:
            import tiktoken
            _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TIKTOKEN_ENCODING = False
    return _TIKTOKEN_ENCODING if _TIKTOKEN_ENCODING is not False else None


def set_default_backend(backend: str):
    """Set the active default tokenizer backend ('qwen', 'cl100k', or 'word')."""
    global _ACTIVE_BACKEND
    _ACTIVE_BACKEND = backend.lower()


def get_default_backend() -> str:
    """Get the active default tokenizer backend."""
    return _ACTIVE_BACKEND


def get_tokenizer_info() -> dict:
    """Return metadata about the current tokenizer backend and vocabulary."""
    backend = _ACTIVE_BACKEND
    if backend == "qwen":
        qwen = _get_qwen_tokenizer()
        if qwen is not None:
            return {"backend": "qwen", "model": "Qwen/Qwen2.5-7B", "vocab_size": qwen.get_vocab_size()}
    
    tik = _get_tiktoken_encoding()
    if tik is not None:
        return {"backend": "cl100k_base", "model": "tiktoken/cl100k_base", "vocab_size": 100277}
    
    return {"backend": "word", "model": "whitespace_fallback", "vocab_size": -1}


def approx_tokens(text: str, backend: Optional[str] = None) -> int:
    """Exact subword token count via Qwen, tiktoken, or word-count fallback."""
    target_backend = (backend or _ACTIVE_BACKEND).lower()
    
    if target_backend == "qwen":
        qwen = _get_qwen_tokenizer()
        if qwen is not None:
            return max(1, len(qwen.encode(text).ids))
        # Fallback to tiktoken if Qwen unavailable
        tik = _get_tiktoken_encoding()
        if tik is not None:
            return max(1, len(tik.encode(text)))
            
    elif target_backend in ("cl100k", "tiktoken"):
        tik = _get_tiktoken_encoding()
        if tik is not None:
            return max(1, len(tik.encode(text)))

    return max(1, len(re.findall(r"\S+", text)))


def count_tokens(text: str, backend: Optional[str] = None) -> int:
    """Alias for exact token counting."""
    return approx_tokens(text, backend=backend)


def truncate_to_tokens(text: str, budget: int, backend: Optional[str] = None) -> str:
    """Truncates text to exact token budget under the selected tokenizer backend."""
    target_backend = (backend or _ACTIVE_BACKEND).lower()

    if target_backend == "qwen":
        qwen = _get_qwen_tokenizer()
        if qwen is not None:
            enc = qwen.encode(text)
            if len(enc.ids) <= budget:
                return text
            return qwen.decode(enc.ids[:max(1, budget)])
        tik = _get_tiktoken_encoding()
        if tik is not None:
            tokens = tik.encode(text)
            if len(tokens) <= budget:
                return text
            return tik.decode(tokens[:max(1, budget)])

    elif target_backend in ("cl100k", "tiktoken"):
        tik = _get_tiktoken_encoding()
        if tik is not None:
            tokens = tik.encode(text)
            if len(tokens) <= budget:
                return text
            return tik.decode(tokens[:max(1, budget)])

    words = re.findall(r"\S+", text)
    return " ".join(words[:max(1, budget)])

