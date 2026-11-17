"""Retrieval-augmented answers over a small built-in knowledge base (BGE embeddings + Llama 3.3 on Workers AI)."""
import json, math, re
from pathlib import Path

MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
EMBED = "@cf/baai/bge-base-en-v1.5"
DOCS = json.loads((Path(__file__).resolve().parents[2] / "data" / "docs.json").read_text())
SYSTEM = "Answer ONLY from the numbered sources. Cite them like [1]. If the sources do not contain the answer, say you don't know."


def cosine(a, b):
    d = x = y = 0.0
    for p, q in zip(a, b):
        d += p * q; x += p * p; y += q * q
    den = math.sqrt(x) * math.sqrt(y)
    return d / (den or 1)


def top_k(qv, vecs, k=3):
    return sorted(({"i": i, "score": cosine(qv, v)} for i, v in enumerate(vecs)), key=lambda h: -h["score"])[:k]


def text(r):
    v = r.get("response") if isinstance(r, dict) else None
    return v if isinstance(v, str) else json.dumps(v if v is not None else r, separators=(",", ":"), ensure_ascii=False)


def cap(s, n=500):
    return isinstance(s, str) and bool(s.strip()) and len(s) <= n


class Rag:
    """ai(model, input) -> dict is injected so tests run offline. The document index is cached per instance (per isolate)."""

    def __init__(self, ai):
        self.ai = ai
        self.index = None

    def embed(self, texts):
        return self.ai(EMBED, {"text": texts})["data"]

    def answer(self, question, k=3):
        if self.index is None:
            self.index = self.embed([f"{d['title']}. {d['text']}" for d in DOCS])
        qv = self.embed([question])[0]
        hits = [{**DOCS[h["i"]], "score": round(h["score"], 3)} for h in top_k(qv, self.index, k)]
        context = "\n".join(f"[{n + 1}] {h['title']}: {h['text']}" for n, h in enumerate(hits))
        r = self.ai(MODEL, {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}], "max_tokens": 400})
        return {"answer": text(r), "sources": [{"ref": n + 1, "title": h["title"], "score": h["score"]} for n, h in enumerate(hits)]}


def handle(rag, method, path, body=None):
    """Pure router: (status, json); status 0 = serve the page."""
    if path == "/api/docs":
        return 200, [{"id": d["id"], "title": d["title"]} for d in DOCS]
    if method == "POST" and path == "/api/ask":
        q = (body or {}).get("question") if isinstance(body, dict) else None
        if not cap(q):
            return 400, {"error": "question required (<=500 chars)"}
        try:
            return 200, rag.answer(q)
        except Exception as e:
            return 502, {"error": "Agent error: " + str(e)}
    return 0, None
