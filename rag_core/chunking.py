from typing import Dict, List

from rag_core.config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_pages(pages: List[Dict]) -> List[Dict]:
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": page["source"],
                        "page": page["page"],
                    }
                )
            if end == len(text):
                break
            start = max(end - CHUNK_OVERLAP, 0)
    return chunks
