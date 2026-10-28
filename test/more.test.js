import { test } from "node:test";
import assert from "node:assert/strict";
import { cosine, topK, ragAnswer, _reset } from "../src/rag.js";
import { DOCS } from "../src/docs.js";
import worker from "../src/index.js";

test("cosine handles zero vectors and scale", () => { assert.equal(cosine([0, 0], [1, 1]), 0); assert.ok(Math.abs(cosine([2, 0], [5, 0]) - 1) < 1e-12); });
test("topK respects k and ties stay stable", () => { const r = topK([1, 0], [[1, 0], [1, 0], [0, 1]], 2); assert.deepEqual(r.map((h) => h.i), [0, 1]); });
test("docs have unique ids and non-empty text", () => { assert.equal(new Set(DOCS.map((d) => d.id)).size, DOCS.length); assert.ok(DOCS.every((d) => d.title && d.text.length > 50)); });
test("index is embedded once per isolate", async () => {
  _reset(); let batches = 0;
  const env = { AI: { run: async (_m, inp) => (inp.text ? (inp.text.length > 1 && batches++, { data: inp.text.map(() => [1, 0]) }) : { response: "x [1]" }) } };
  await ragAnswer(env, "a"); await ragAnswer(env, "b"); assert.equal(batches, 1);
});
test("sources carry ref numbers and rounded scores", async () => {
  _reset(); const env = { AI: { run: async (_m, inp) => (inp.text ? { data: inp.text.map((_, i) => [1, i / 10]) } : { response: "y" }) } };
  const r = await ragAnswer(env, "q", 3); assert.deepEqual(r.sources.map((s) => s.ref), [1, 2, 3]); assert.ok(r.sources.every((s) => typeof s.score === "number"));
});
test("HTTP: /api/docs lists titles; /api/ask validates", async () => {
  const d = await (await worker.fetch(new Request("https://x/api/docs"), {})).json(); assert.equal(d.length, 12); assert.ok(!("text" in d[0]));
  const r = await worker.fetch(new Request("https://x/api/ask", { method: "POST", body: "{}" }), {}); assert.equal(r.status, 400);
  const big = await worker.fetch(new Request("https://x/api/ask", { method: "POST", body: JSON.stringify({ question: "x".repeat(501) }) }), {}); assert.equal(big.status, 400);
});
test("HTTP: model failure becomes 502", async () => {
  _reset(); const env = { AI: { run: async () => { throw new Error("quota"); } } };
  const r = await worker.fetch(new Request("https://x/api/ask", { method: "POST", body: JSON.stringify({ question: "hi" }) }), env); assert.equal(r.status, 502);
});
