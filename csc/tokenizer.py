import re


def approx_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\\S+", text)))


def truncate_to_tokens(text: str, budget: int) -> str:
    words = re.findall(r"\\S+", text)
    return " ".join(words[:max(1, budget)])
