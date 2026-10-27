# agent-05-rag

A **retrieval-augmented generation (RAG)** agent that answers only from its knowledge base and cites its sources. It runs serverless on Cloudflare Workers with the free Workers AI tier. Project 5 of a 10-part agentic AI portfolio.

**Live demo:** https://agent-05-rag.braylo777.workers.dev

## What it is
Ask a question. The agent finds the most relevant passages in a 12-document knowledge base about agentic AI, answers from them with `[n]` citations, and says "I don't know" when the sources don't cover it. The response lists each source with its similarity score.

## How it works
```
docs ─▶ bge-base embeddings (cached per isolate)
question ─▶ embedding ─▶ cosine similarity ─▶ top-3 passages
                                               │
                       grounded prompt + [n] sources ─▶ LLM ─▶ cited answer
```
- `src/docs.js`: the knowledge base. Replace it with your own documents.
- `src/rag.js`: embedding, cosine top-k retrieval and the grounded prompt.
- No vector database is needed at this size: vectors are computed once per isolate and kept in memory. For large corpora, swap in Cloudflare Vectorize.
- Guardrails: 5 requests/min per IP, 500-char cap, answer-only-from-sources instruction.

## Run it
```bash
npm install
npm test            # offline unit tests (Node 20+), LLM is mocked
npx wrangler dev    # local; Workers AI runs via your Cloudflare login
npx wrangler deploy # free tier, no paid API keys
```

## Stack
Cloudflare Workers · Workers AI (`@cf/meta/llama-3.3-70b-instruct-fp8-fast`, `@cf/baai/bge-base-en-v1.5` embeddings) · vanilla JS · node:test

## License
MIT
