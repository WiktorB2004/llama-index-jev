# Benchmark

Smoke retrieval eval: **BM25 first-stage vs `JevRerank`**. No Cohere, ColBERT, or
zerank-2 in this pass — those need extra keys or a GPU. OpenRouter is enough
to compare Jev against the no-rerank baseline.

Jev scores one query–passage pair per HTTP call. A full BEIR split is thousands
of Decisions requests. This runner defaults to **5 queries**, **top_k=10**,
**top_n=5**, so a first run is about **50 calls**.

## Setup

```bash
uv sync --group benchmark
export OPENROUTER_API_KEY=...   # or TYPESAFE_API_KEY with --provider typesafe
```

## Run

```bash
uv run python -m benchmark.run_benchmark --provider openrouter --queries 5
```

Useful flags (the defaults above are the smoke):

| Flag | Default | Notes |
| --- | --- | --- |
| `--dataset` | `nfcorpus` | BEIR test split; also `scifact` |
| `--queries` | `5` | First N test queries that have qrels |
| `--top-k` | `10` | BM25 candidate depth |
| `--top-n` | `5` | Keep this many after rerank (headline metric is nDCG@5) |
| `--mode` | `noul` | Jev question type; `score` is a second system later |
| `--provider` | `openrouter` | `typesafe` uses `TYPESAFE_API_KEY` |
| `--timeout-s` | `15` | Default Jev timeout (2.5s) is too tight for a 10-hit burst |
| `--baseline-only` | off | BM25 only; no API key |
| `--no-cache` | off | Do not read/write `benchmark/cache/` |

Re-running the same flags hits the on-disk cache (first-stage hits and per
`(query_id, doc_id, mode, provider, model)` Jev scores), so a crash does not
re-bill finished passages.

## Metrics

Both systems return `top_n` docs from the same BM25 `top_k` pool:

- **bm25** — first-stage order, truncated to `top_n`
- **jev-noul** — Jev relevance probability, truncated to `top_n`

Reported: nDCG@5, nDCG@10, rerank latency p50/p95, OpenRouter `usage` tokens
and USD (falls back to ~$0.042/M input tokens if the response has no `cost`).
nDCG@10 with `--top-n 5` only has five ranked docs; pass `--top-n 10` if that
cutoff matters.

JSON is written to `benchmark/results/` (gitignored). Cache lives in
`benchmark/cache/` (also gitignored).

## What this is not

Not a full BEIR leaderboard. Do not raise `--queries` to the official split
until the smoke table looks sane. `mode=score`, more queries, and other
rerankers are follow-ups.
