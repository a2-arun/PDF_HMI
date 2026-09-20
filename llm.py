"""Turns retrieved textbook context into a student-friendly answer.

Uses a local model served by Ollama (https://ollama.com) so the assistant
can run fully offline with no per-query API cost — important for a robot
that may not have reliable internet access. The model name is a config
value, not hardcoded, so swapping between something as small as `tinyllama`
and a stronger instruction-following model like `llama3.2` is a one-line
change (see README for the trade-off).
"""
import requests

from config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS

SYSTEM_PROMPT = (
    "You are a friendly, patient teaching assistant helping a school student. "
    "Answer the student's question using ONLY the textbook content provided below. "
    "Explain it in simple, age-appropriate language, as if speaking directly to the "
    "student. Do not mention 'the textbook', 'the excerpt', or the retrieval process — "
    "just answer naturally. If the provided content does not actually answer the "
    "question, say you're not sure rather than guessing. Keep the answer focused: a "
    "few clear sentences, not an essay."
)


class OllamaUnavailableError(RuntimeError):
    """Raised when Ollama can't be reached or fails to generate a response."""


def build_prompt(question: str, context: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Textbook content:\n{context}\n\n"
        f"Student's question: {question}\n\n"
        f"Answer:"
    )


def is_available() -> bool:
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        return response.ok
    except requests.RequestException:
        return False


def generate(question: str, context: str) -> str:
    prompt = build_prompt(question, context)
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3},
            },
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaUnavailableError(
            f"Could not reach Ollama at {OLLAMA_HOST} using model '{OLLAMA_MODEL}'. "
            f"Install Ollama from https://ollama.com, run `ollama pull {OLLAMA_MODEL}`, "
            f"and make sure `ollama serve` is running."
        ) from exc

    text = response.json().get("response", "").strip()
    return _dedupe_lines(text)


def _dedupe_lines(text: str) -> str:
    """Small local models sometimes repeat a line verbatim; drop repeats."""
    seen = set()
    out = []
    for line in text.splitlines():
        key = line.strip()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(line)
    return "\n".join(out).strip()
