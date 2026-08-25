import re

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _ENCODING = None


def approx_tokens(text: str) -> int:
    """Exact subword token count via tiktoken if available, else word count fallback."""
    if _ENCODING is not None:
        return max(1, len(_ENCODING.encode(text)))
    return max(1, len(re.findall(r"\S+", text)))


def count_tokens(text: str) -> int:
    """Alias for exact token counting."""
    return approx_tokens(text)


def truncate_to_tokens(text: str, budget: int) -> str:
    """Truncates text to exact token budget."""
    if _ENCODING is not None:
        tokens = _ENCODING.encode(text)
        if len(tokens) <= budget:
            return text
        return _ENCODING.decode(tokens[:max(1, budget)])
    
    words = re.findall(r"\S+", text)
    return " ".join(words[:max(1, budget)])
