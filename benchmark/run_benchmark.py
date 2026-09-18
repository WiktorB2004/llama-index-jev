"""Jev vs Cohere Rerank vs ColBERT vs zerank-2 on a BEIR subset.

TODO: implement the run. It needs a live TYPESAFE_API_KEY (and keys for
the other vendors). This file is a methodology stub only.

Intended methodology
--------------------
Dataset
    A small BEIR subset first (`scifact` or `nfcorpus`) so a smoke run is
    cheap, then a larger split if numbers look real.

First-stage retrieval
    The same bi-encoder / vector index for every reranker, `top_k=50`.
    Rerankers only reorder; they do not retrieve.

Rerankers
    * JevRerank(mode="score") and JevRerank(mode="noul")
    * Cohere Rerank
    * ColbertRerank
    * zerank-2

Metrics
    nDCG@10, latency p50/p95, estimated USD per query.

Why this is stubbed
    Jev scoring is one HTTP call per passage. A real BEIR run is a paid
    API bill, and this repo is designed to build and test with zero live
    access.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "Benchmark runner is stubbed. See the module docstring for the "
        "intended methodology. A live TYPESAFE_API_KEY is required."
    )


if __name__ == "__main__":
    main()
