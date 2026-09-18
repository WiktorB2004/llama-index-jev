# LlamaIndex + TypeSafe Jev

Drop-in LlamaIndex **reranker** and **router** powered by [TypeSafe Jev](https://typesafe.ai): typed `Score` / `Choice` answers, cheap compared to LLM-as-judge — not a Cohere or FlagEmbedding cross-encoder.

[![PyPI - postprocessor](https://img.shields.io/pypi/v/llama-index-postprocessor-jev?label=postprocessor)](https://pypi.org/project/llama-index-postprocessor-jev/)
[![PyPI - selectors](https://img.shields.io/pypi/v/llama-index-selectors-jev?label=selectors)](https://pypi.org/project/llama-index-selectors-jev/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/llama-index-postprocessor-jev/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://wiktorb2004.github.io/llama-index-jev/)
[![CI](https://img.shields.io/github/actions/workflow/status/WiktorB2004/llama-index-jev/ci.yml?branch=main&label=CI)](https://github.com/WiktorB2004/llama-index-jev/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/github/license/WiktorB2004/llama-index-jev)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/llama-index-postprocessor-jev?label=postprocessor%20dl)](https://pypi.org/project/llama-index-postprocessor-jev/)

Independent community project. **Not** affiliated with TypeSafe or LlamaIndex.

## Install

```bash
pip install llama-index-postprocessor-jev   # JevRerank
pip install llama-index-selectors-jev       # JevSingleSelector, JevMultiSelector
export TYPESAFE_API_KEY=...                 # or OPENROUTER_API_KEY + provider="openrouter"
```

## Quickstart

**Rerank** — score each retrieved passage, keep the top `n`:

```python
from llama_index.postprocessor.jev import JevRerank

reranker = JevRerank(top_n=5, mode="score")
# OpenRouter: JevRerank(provider="openrouter", top_n=5, mode="score")
query_engine = index.as_query_engine(node_postprocessors=[reranker])
```

**Select** — pick which query engine / tool handles the query:

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.selectors.jev import JevSingleSelector

engine = RouterQueryEngine(
    selector=JevSingleSelector(),
    query_engine_tools=[weather_tool, docs_tool],
)
```

Docs: [wiktorb2004.github.io/llama-index-jev](https://wiktorb2004.github.io/llama-index-jev/). Paste-and-run walkthroughs (OpenRouter, mock embeddings / MockLLM so you do not need an OpenAI key): [`examples/`](examples/README.md).

Package docs: [`JevRerank`](packages/llama-index-postprocessor-jev/README.md) · [`JevSingleSelector` / `JevMultiSelector`](packages/llama-index-selectors-jev/README.md).

`mode="score"` is a **0–3** relevance rubric (off-topic → fully answers), not cosine similarity. Rerank **fails open** (keep retrieval order); select **fails closed** (raise, or a declared `default_index`).

## Results

BEIR **nfcorpus** test, 323 queries. Protocol: MiniLM dense top-10, then `JevRerank(mode="score", top_n=5)`, OpenRouter `jev-latest`.

| System | nDCG@5 |
| --- | ---: |
| BM25 | 0.298 |
| MiniLM | 0.340 |
| MiniLM + Jev | **0.396** |

Δ nDCG@5 vs MiniLM: **+0.056** (95% CI 0.042–0.072, excludes 0). Cost ≈ **$0.096** for the split (≈ **$0.0003/query**).

This is **not** a BEIR leaderboard vs Cohere: first-stage is MiniLM, not Pyserini. SciFact (same protocol, 300 queries): MiniLM 0.629 → MiniLM+Jev **0.715** (Δ **+0.086**, 95% CI 0.059–0.113). Details and BGE-small: [`benchmark/README.md`](benchmark/README.md).

## Why Jev instead of Cohere / FlagEmbedding / an LLM

Cross-encoders (Cohere, FlagEmbedding) are dedicated rerank models with a similarity score. An LLM-as-judge loop is flexible and expensive, and the answer is unstructured. Jev is a System One **decision** model: you send a small `state` and typed questions (`Score`, `Choice`, `Noul`) and get typed answers back. Same vendor covers both LlamaIndex hooks — rerank after retrieval, select before a tool call — at about $0.0003/query on this protocol.

## Design

### One call per passage

Rerank asks one relevance question **per passage** (`state = {query, passage}` only). Stuffing many passages into one prompt causes context rot, and Jev cannot see question ids. TypeSafe's own rerank cookbook scores one pair at a time. Parallelism is `max_concurrency` (default 8).

Select may batch: a router only has the query as state, so one `Choice` (single) or one `Noul` per option (multi) in a single request is the right shape.

### Fail open vs fail closed

Rerank **fails open**: if any Jev call in a pass errors, the original retrieval order is returned (truncated to `top_n`). A slightly-worse ranking is better than no context. Set `raise_on_error=True` to surface the error.

Select **fails closed**: API errors, an out-of-set choice, or low confidence **raise**, unless you set `default_index`. The wrong tool is worse than an error.

### Why two packages

LlamaIndex has two extension points, published as two families:

| Package | Class | Hook |
| --- | --- | --- |
| [`llama-index-postprocessor-jev`](packages/llama-index-postprocessor-jev/README.md) | `JevRerank` | `BaseNodePostprocessor` (after retrieval) |
| [`llama-index-selectors-jev`](packages/llama-index-selectors-jev/README.md) | `JevSingleSelector`, `JevMultiSelector` | `BaseSelector` (before a tool call) |

Install only the one you need.

## Tests

Tests mock `TypeSafeClient.system_one` / `AsyncTypeSafeClient.system_one`. No live API key is required.

```bash
uv sync

# Run each package separately so test module names do not collide.
uv run pytest --rootdir=packages/llama-index-postprocessor-jev \
    packages/llama-index-postprocessor-jev
uv run pytest --rootdir=packages/llama-index-selectors-jev \
    packages/llama-index-selectors-jev

uv run mypy
```

## Docs

```bash
uv sync --group docs
uv run mkdocs serve
```

Published at [wiktorb2004.github.io/llama-index-jev](https://wiktorb2004.github.io/llama-index-jev/).

## Benchmark

Retrieval eval lives in [`benchmark/`](benchmark/). Needs `OPENROUTER_API_KEY` (or `TYPESAFE_API_KEY`) and `uv sync --group benchmark`.

```bash
# Smoke (~50 Jev calls)
uv run python -m benchmark.run_benchmark --provider openrouter --queries 5

# Usage protocol (full test split, MiniLM top-10, Jev score, top_n=5)
uv run python -m benchmark.run_benchmark --preset usage --dataset nfcorpus
uv run python -m benchmark.run_benchmark --preset usage --dataset scifact
```

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md). To report a vulnerability, use [SECURITY.md](SECURITY.md).

## License

MIT. See [LICENSE](LICENSE). Changelog: [CHANGELOG.md](CHANGELOG.md). Cite this repo with [CITATION.cff](CITATION.cff).
