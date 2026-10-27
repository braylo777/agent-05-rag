import { EMBED, ask } from "./core.js";
import { DOCS } from "./docs.js";

export function cosine(a, b) {
  let d = 0, x = 0, y = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; x += a[i] * a[i]; y += b[i] * b[i]; }
  return d / (Math.sqrt(x) * Math.sqrt(y) || 1);
}
export function topK(qv, vecs, k = 3) {
  return vecs.map((v, i) => ({ i, score: cosine(qv, v) })).sort((a, b) => b.score - a.score).slice(0, k);
}

let INDEX = null; // per-isolate cache of document embeddings
export async function embed(env, texts) { return (await env.AI.run(EMBED, { text: texts })).data; }

export async function ragAnswer(env, question, k = 3) {
  if (!INDEX) INDEX = await embed(env, DOCS.map((d) => `${d.title}. ${d.text}`));
  const [qv] = await embed(env, [question]);
  const hits = topK(qv, INDEX, k).map(({ i, score }) => ({ ...DOCS[i], score: +score.toFixed(3) }));
  const context = hits.map((h, n) => `[${n + 1}] ${h.title}: ${h.text}`).join("\n");
  const answer = await ask(env, "Answer ONLY from the numbered sources. Cite them like [1]. If the sources do not contain the answer, say you don't know.", `Sources:\n${context}\n\nQuestion: ${question}`);
  return { answer, sources: hits.map((h, n) => ({ ref: n + 1, title: h.title, score: h.score })) };
}
export function _reset() { INDEX = null; }
