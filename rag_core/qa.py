import os
import json
import urllib.request
import urllib.error
from typing import Dict, List

from openai import OpenAI


def _build_context(hits: List[Dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[{i}] Source={h['source']} Page={h['page']}\n{h['text']}"
        )
    return "\n\n".join(blocks)


def _local_fallback_answer(question: str, hits: List[Dict]) -> str:
    if not hits:
        return "I could not find relevant context in the indexed PDFs."
    lines = [f"Question: {question}", "", "Relevant excerpts:"]
    for i, h in enumerate(hits, start=1):
        snippet = h["text"][:280].strip()
        lines.append(f"[{i}] {snippet} ({h['source']}, page {h['page']})")
    lines.append("")
    lines.append("Set OLLAMA/OpenAI mode for a stronger synthesized answer.")
    return "\n".join(lines)


def _answer_with_ollama(question: str, hits: List[Dict], model: str) -> str:
    context = _build_context(hits)
    prompt = (
        "Answer the question using only the context below. "
        "If not found, say you do not know. Include bracket citations like [1], [2].\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False}
    ).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "").strip()


def answer_question(
    question: str,
    hits: List[Dict],
    mode: str = "ollama",
    ollama_model: str = "llama3.2:3b",
) -> str:
    if mode == "local":
        return _local_fallback_answer(question, hits)

    if mode == "ollama":
        try:
            return _answer_with_ollama(question, hits, ollama_model)
        except (urllib.error.URLError, TimeoutError, OSError):
            return _local_fallback_answer(question, hits)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _local_fallback_answer(question, hits)

    client = OpenAI(api_key=api_key)
    context = _build_context(hits)
    prompt = (
        "Answer the question using only the context below. "
        "If not found, say you do not know. Include bracket citations like [1], [2].\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0.2,
        )
        return response.output_text.strip()
    except Exception:
        return _local_fallback_answer(question, hits)
