# agent-05-rag

A **retrieval-augmented generation (RAG)** agent that answers only from its knowledge base and cites its sources. It runs serverless on Cloudflare Workers on the free Workers AI tier. Project 5 of a 10-part agentic AI portfolio.

**Live demo:** https://agent-05-rag.braylo777.workers.dev

![Demo screenshot](docs/screenshot.png)

## What it is
Ask a question. The agent finds the most relevant passages in a 12-document knowledge base about agentic AI, answers from them with `[n]` citations, and says it doesn't know when the sources don't cover it. Each response lists the retrieved sources with their similarity scores.

## Architecture
```mermaid
flowchart LR
  U[Browser / API client] -->|POST /api/ask| W[Cloudflare Worker]
  W --> RL{Rate limit<br/>5 req/min/IP}
  RL --> E1[BGE embeddings<br/>Workers AI]
  D[(12 docs<br/>data/docs.json)] -->|embedded once per isolate| IDX[In-memory vector index]
  E1 -->|query vector| S[Cosine top-k = 3]
  IDX --> S
  S --> P[Grounded prompt<br/>numbered sources]
  P --> L[Llama 3.3 70B<br/>Workers AI]
  L -->|answer with citations + sources| U
```

## How it works
- `data/docs.json`: the knowledge base. Replace it with your own documents.
- `src/agent05/rag.py` and `worker/src/rag.rs`: embeddings, cosine top-k retrieval and the grounded prompt ("answer only from the numbered sources, cite like [1]").
- No vector database is needed at this size: document vectors are computed once per isolate and kept in memory. For a large corpus, swap in Cloudflare Vectorize.
- Guardrails: 5 requests/min per IP, 500-character cap, answer-only-from-sources instruction.

## Eval results
Run live against the deployed demo (`python3 evals/run.py`) on 20 hand-written questions, each with one gold document and an expected key phrase. Raw per-case output is in [`evals/results.json`](evals/results.json).

| Metric | Result | Definition |
|---|---|---|
| Retrieval hit@1 | **19/20 (0.95)** | Gold document ranked first |
| Retrieval hit@3 | **20/20 (1.00)** | Gold document in the top 3 sent to the model |
| Citation accuracy | **20/20 (1.00)** | Answer cites at least one `[n]`, and every cited `[n]` is the gold document |
| Answer accuracy | **18/20 (0.90)** | Answer contains the expected key phrase |
| Latency p50 / p95 | **844 ms / 2028 ms** (n=20) | End-to-end `/api/ask`, measured from the eval client |

Known misses: in 2 of 20 cases the model replied with only the citation `[1]` and no answer text. The one hit@1 miss ("ranking passages") put the Embeddings doc second, behind the RAG doc. Models: `@cf/baai/bge-base-en-v1.5` and `@cf/meta/llama-3.3-70b-instruct-fp8-fast`. Run on Oct 27, 2026.

## Tests
Python: 8 tests; Rust: 5 test functions (Workers AI mocked). A scripted run of 5 questions gives identical answers, sources and scores for the old JavaScript, the Python and the Rust code. Covered: cosine math and edge cases, top-k ranking, embedding the index once per isolate, retrieval and citation in the prompt, source ref numbering, HTTP input validation, and a model failure returning 502.

```bash
python3 -m pytest
cd worker && cargo test
```

## Run it
```bash
python3 -m pytest                         # Python tests
cd worker && cargo test                   # Rust tests
cd worker && worker-build --release       # build the Worker (wasm)
wrangler dev                              # local; Workers AI runs via your Cloudflare login
wrangler deploy                           # free tier, no paid API keys
BASE=https://... python3 evals/run.py     # live eval (throttled); writes evals/results.json
```

The previous JavaScript version is on the `legacy-js` branch and the `pre-pycrust` tag.

## Stack
Cloudflare Workers · Workers AI (Llama 3.3 70B, BGE base embeddings) · Rust (workers-rs) · Python (pytest)

## License
MIT
