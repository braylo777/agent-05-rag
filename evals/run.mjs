// Live eval against the deployed RAG agent (BGE embeddings + Llama 3.3 on Workers AI).
// hit@1 / hit@3: gold document is in the top 1 / top 3 retrieved sources.
// citation accuracy: the answer cites at least one [n] and every cited [n] is the gold source.
// answer accuracy: the answer contains an expected key phrase from the gold document.
import fs from "node:fs";
import { call, pct, ratio, has } from "./lib.mjs";
const BASE = process.env.BASE || "https://agent-05-rag.braylo777.workers.dev";
const cases = JSON.parse(fs.readFileSync(new URL("./cases.json", import.meta.url)));
const rows = [], lat = [];
for (const [i, c] of cases.entries()) {
  const r = await call(BASE + "/api/ask", { method: "POST", body: JSON.stringify({ question: c.q }) });
  if (r.status === 200) lat.push(r.ms);
  const src = r.body?.sources || [], ans = r.body?.answer ?? r.body?.error ?? "";
  const goldRef = src.find((s) => s.title === c.gold)?.ref;
  const cited = [...new Set([...ans.matchAll(/\[(\d+)\]/g)].map((m) => +m[1]))];
  rows.push({ i, q: c.q, gold: c.gold, retrieved: src.map((s) => s.title), cited, answer: ans,
    hit1: src[0]?.title === c.gold, hit3: !!goldRef, citation_ok: cited.length > 0 && cited.every((n) => n === goldRef), answer_ok: r.status === 200 && has(ans, c.expect) });
  console.log(i, rows.at(-1).hit1, rows.at(-1).hit3, rows.at(-1).citation_ok, rows.at(-1).answer_ok);
}
const n = rows.length, k = (f) => rows.filter(f).length;
const out = { project: "agent-05-rag", base: BASE, models: { embed: "@cf/baai/bge-base-en-v1.5", llm: "@cf/meta/llama-3.3-70b-instruct-fp8-fast" }, run_at: new Date().toISOString(), n_cases: n, corpus_docs: 12, k: 3,
  metrics: { hit_at_1: ratio(k((r) => r.hit1), n), hit_at_3: ratio(k((r) => r.hit3), n), citation_accuracy: ratio(k((r) => r.citation_ok), n), answer_accuracy: ratio(k((r) => r.answer_ok), n),
    latency_ms: { n: lat.length, p50: pct(lat, 50), p95: pct(lat, 95), note: "end-to-end /api/ask, measured from the eval client" } }, cases: rows };
fs.writeFileSync(new URL("./results.json", import.meta.url), JSON.stringify(out, null, 2));
console.log(JSON.stringify(out.metrics, null, 2));
