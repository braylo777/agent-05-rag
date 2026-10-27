import { test } from "node:test";
import assert from "node:assert/strict";
import { cosine, topK, ragAnswer } from "../src/rag.js";
import { DOCS } from "../src/docs.js";

test("cosine", () => { assert.equal(cosine([1, 0], [1, 0]), 1); assert.equal(cosine([1, 0], [0, 1]), 0); });
test("topK ranks by similarity", () => { assert.deepEqual(topK([1, 0], [[0, 1], [1, 0], [1, 1]], 2).map((h) => h.i), [1, 2]); });
test("ragAnswer retrieves and cites", async () => {
  // Fake embeddings: one-hot on whether text mentions "MCP".
  const vec = (t) => (/Model Context Protocol|MCP/.test(t) ? [1, 0] : [0, 1]);
  const env = { AI: { run: async (model, inp) => inp.text ? { data: inp.text.map(vec) } : (assert.match(inp.messages[1].content, /\[1\] Model Context Protocol/), { response: "MCP is a standard [1]." }) } };
  const r = await ragAnswer(env, "What is MCP?", 1);
  assert.equal(r.sources[0].title, "Model Context Protocol"); assert.match(r.answer, /\[1\]/);
  assert.equal(DOCS.length, 12);
});
