# Benchmark (stub)

Compare Jev rerank against other rerankers on a retrieval benchmark.

This pass only scaffolds the runner. A live `TYPESAFE_API_KEY` is required
to produce numbers, and the user is still waitlisted.

Intended methodology (to be implemented in `run_benchmark.py`):

- **Dataset:** a BEIR subset (start with `scifact` or `nfcorpus` so a first
  run is cheap).
- **Retrievers / rerankers:**
  - Jev (`JevRerank`, `mode="score"` and `mode="noul"`)
  - Cohere Rerank
  - ColBERT (`ColbertRerank`)
  - zerank-2
- **Metrics:** nDCG@10, latency (p50 / p95), estimated cost per query.
- **Protocol:** retrieve a fixed `similarity_top_k` (e.g. 50) with the same
  first-stage retriever for every reranker, then rerank to 10.

See `run_benchmark.py` for the TODO surface.
