"""Live eval against the deployed RAG agent (BGE embeddings + Llama 3.3 on Workers AI).
hit@1 / hit@3: gold document is in the top 1 / top 3 retrieved sources.
citation accuracy: the answer cites at least one [n] and every cited [n] is the gold source.
answer accuracy: the answer contains an expected key phrase from the gold document.
Usage: BASE=https://... python3 evals/run.py"""
import json, math, os, re, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path

here = Path(__file__).parent
BASE = os.environ.get("BASE", "https://agent-05-rag.braylo777.workers.dev")
GAP = float(os.environ.get("EVAL_GAP_MS", "12500")) / 1000
cases = json.loads((here / "cases.json").read_text())


def call(url, body):
    for _ in range(6):
        time.sleep(GAP)
        t = time.time()
        req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
        try:
            r = urllib.request.urlopen(req, timeout=60); status = r.status; raw = r.read()
        except urllib.error.HTTPError as e:
            status = e.code; raw = e.read()
        ms = (time.time() - t) * 1000
        if status == 429:
            time.sleep(30); continue
        try:
            return {"status": status, "ms": ms, "body": json.loads(raw)}
        except ValueError:
            return {"status": status, "ms": ms, "body": None}
    raise RuntimeError("rate limited repeatedly")


pct = lambda xs, p: round(sorted(xs)[max(0, math.ceil(p / 100 * len(xs)) - 1)]) if xs else None
ratio = lambda k, n: {"correct": k, "n": n, "rate": round(k / n, 4) if n else None}
has = lambda s, keys: any(k.lower() in str(s or "").lower() for k in keys)
rows, lat = [], []
for i, c in enumerate(cases):
    r = call(BASE + "/api/ask", {"question": c["q"]})
    if r["status"] == 200:
        lat.append(r["ms"])
    b = r["body"] or {}
    src = b.get("sources") or []; ans = b.get("answer") or b.get("error") or ""
    gold_ref = next((s["ref"] for s in src if s["title"] == c["gold"]), None)
    cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", ans)}, key=lambda x: ans.find(f"[{x}]"))
    row = {"i": i, "q": c["q"], "gold": c["gold"], "retrieved": [s["title"] for s in src], "cited": cited, "answer": ans,
           "hit1": bool(src) and src[0]["title"] == c["gold"], "hit3": gold_ref is not None,
           "citation_ok": bool(cited) and all(n == gold_ref for n in cited), "answer_ok": r["status"] == 200 and has(ans, c["expect"])}
    rows.append(row)
    print(i, row["hit1"], row["hit3"], row["citation_ok"], row["answer_ok"])
n = len(rows); k = lambda f: sum(1 for r in rows if r[f])
out = {"project": "agent-05-rag", "base": BASE, "models": {"embed": "@cf/baai/bge-base-en-v1.5", "llm": "@cf/meta/llama-3.3-70b-instruct-fp8-fast"},
       "run_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"), "n_cases": n, "corpus_docs": 12, "k": 3,
       "metrics": {"hit_at_1": ratio(k("hit1"), n), "hit_at_3": ratio(k("hit3"), n), "citation_accuracy": ratio(k("citation_ok"), n), "answer_accuracy": ratio(k("answer_ok"), n),
                   "latency_ms": {"n": len(lat), "p50": pct(lat, 50), "p95": pct(lat, 95), "note": "end-to-end /api/ask, measured from the eval client"}}, "cases": rows}
(here / "results.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out["metrics"], indent=2))
