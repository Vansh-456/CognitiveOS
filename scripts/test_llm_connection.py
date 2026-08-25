import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from csc.llm import OpenAICompatibleLLM


def test_connection():
    api_key = os.getenv("LLM_API_KEY", "")
    base_url = os.getenv("LLM_BASE_URL", "")
    model = os.getenv("LLM_MODEL", "")

    print("=" * 60)
    print("TESTING LLM API CONNECTION")
    print("=" * 60)
    print(f"Base URL : {base_url}")
    print(f"Model    : {model}")
    masked_key = (api_key[:6] + "..." + api_key[-4:]) if len(api_key) > 10 else ("EMPTY" if not api_key else "INVALID")
    print(f"API Key  : {masked_key}")
    print("-" * 60)

    if not api_key:
        print("ERROR: LLM_API_KEY is empty in .env!")
        print("Please paste your Groq API key (starting with gsk_...) into .env")
        return

    try:
        llm = OpenAICompatibleLLM(api_key=api_key, base_url=base_url, model=model)
        print("Sending test query: 'Say OK'...")
        response = llm.generate("Say OK")
        print(f"\nSUCCESS! Model responded:\n-> '{response}'")
        print("=" * 60)
    except Exception as e:
        print(f"\nCONNECTION FAILED:\n{e}")
        print("=" * 60)


if __name__ == "__main__":
    test_connection()
