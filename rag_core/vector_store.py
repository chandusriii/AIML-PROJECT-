import json
import re
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_core.config import EMBED_MODEL_NAME, FAISS_INDEX_PATH, METADATA_PATH, STORAGE_DIR


class VectorStore:
    def __init__(self) -> None:
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    def _encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(embeddings, dtype="float32")

    def build_index(self, chunks: List[Dict]) -> None:
        if not chunks:
            raise ValueError("No chunks found. Upload readable PDFs first.")

        vectors = self._encode([c["text"] for c in chunks])
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        faiss.write_index(index, str(FAISS_INDEX_PATH))

        with METADATA_PATH.open("w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=True, indent=2)

    def load(self):
        if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError("No index found. Click 'Build Index' first.")

        index = faiss.read_index(str(FAISS_INDEX_PATH))
        with METADATA_PATH.open("r", encoding="utf-8") as f:
            chunks = json.load(f)
        return index, chunks

    def search(self, query: str, top_k: int = 4) -> List[Dict]:
        index, chunks = self.load()
        pool_k = max(top_k * 5, 20)
        qvec = self._encode([query])
        scores, idxs = index.search(qvec, pool_k)

        query_terms = set(_tokenize(query))

        results = []
        for score, i in zip(scores[0], idxs[0]):
            if i == -1:
                continue
            hit = chunks[i].copy()
            lexical_overlap = _overlap_score(query_terms, set(_tokenize(hit["text"])))
            hit["score"] = float((0.8 * float(score)) + (0.2 * lexical_overlap))
            results.append(hit)
        ranked = sorted(results, key=lambda x: x["score"], reverse=True)
        return _diversify(ranked, top_k=top_k)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _overlap_score(query_terms: set, chunk_terms: set) -> float:
    if not query_terms:
        return 0.0
    return len(query_terms & chunk_terms) / max(len(query_terms), 1)


def _diversify(ranked_hits: List[Dict], top_k: int) -> List[Dict]:
    selected = []
    seen = set()
    for hit in ranked_hits:
        key = (hit["source"], hit["page"])
        if key in seen:
            continue
        selected.append(hit)
        seen.add(key)
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        for hit in ranked_hits:
            if hit in selected:
                continue
            selected.append(hit)
            if len(selected) >= top_k:
                break
    return selected
