# Benchmark

Measures `JevRerank` the way this package is used: retrieve a short list, then
reorder with Jev. Labels come from BEIR (nfcorpus / scifact). This is **not** a
BEIR leaderboard (first-stage is MiniLM or `rank-bm25`, not Pyserini). There is
no Cohere / ColBERT column.

## Results

Measured 2026-09-18 with `JevRerank(provider="openrouter", mode="score")` and
`sentence-transformers/all-MiniLM-L6-v2`. Protocol: retrieve 10, keep 5.
nDCG@5 is the headline. Intervals are 95% percentile bootstrap over queries
(seed 0). Cost is OpenRouter `usage.cost` (not estimated).

| Dataset | N | BM25 | MiniLM | MiniLM + Jev | Δ vs MiniLM (95% CI) | USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [NFCorpus](https://ir-datasets.com/beir.html#beir/nfcorpus/test) test | 323 | 0.298 | 0.340 | **0.396** | **+0.056 [0.042, 0.072]** | 0.096 |
| [SciFact](https://ir-datasets.com/beir.html#beir/scifact/test) test | 300 | 0.535 | 0.629 | **0.715** | **+0.086 [0.059, 0.113]** | 0.092 |

On both MiniLM splits the Δ CI is above 0: Jev improved the MiniLM list, not a
five-query fluke. Extra cost is about **$0.0003 per query**.

**BGE-small** (stronger first-stage, same protocol, NFCorpus only):

| Dataset | N | BGE-small | BGE-small + Jev | Δ vs BGE (95% CI) | USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| NFCorpus test | 323 | 0.375 | **0.415** | **+0.040 [0.026, 0.055]** | 0.101 |

Jev still helps when retrieve is stronger, but the lift is smaller than vs
MiniLM (+0.056 → +0.040). About 1730/3230 passage scores were already in the
MiniLM Jev cache (1500 new API calls). SciFact + BGE is not measured.

nDCG@10 on SciFact equals nDCG@5 because `top_n=5`. NFCorpus nDCG@10 is lower
for every system (graded labels, truncated lists). Do not compare these
absolute BM25 numbers to published BEIR tables (`rank-bm25` + MiniLM/BGE ≠
Pyserini).

Raw JSON is gitignored (`benchmark/results/*.json`). Numbers above are from
`usage_nfcorpus_20260918T121829Z.json`,
`usage_scifact_20260918T123647Z.json`, and
`usage_nfcorpus_bge-small_20260918T130959Z.json`.

## Setup

```bash
uv sync --group benchmark
export OPENROUTER_API_KEY=...   # or TYPESAFE_API_KEY with --provider typesafe
```

`--preset usage` downloads the dense model on the first encode (MiniLM by
default; `bge-small` / `e5-base` if you pass `--embed-model`). Vectors are
cached under `benchmark/cache/`.

## Smoke

Cheap path (~50 Jev calls) to prove the runner:

```bash
uv run python -m benchmark.run_benchmark --provider openrouter --queries 5
```

BM25 top-10, Jev `noul`, first 5 nfcorpus queries.

## Usage protocol (run this)

LlamaIndex-shaped: MiniLM top-10, `JevRerank(mode="score", top_n=5)` (library
default), full test split, BM25 as an extra baseline, 95% bootstrap CIs on
nDCG@5.

```bash
uv run python -m benchmark.run_benchmark --preset usage --dataset nfcorpus
uv run python -m benchmark.run_benchmark --preset usage --dataset scifact
```

Rough cost from the smoke tariff (~$0.000028/call): nfcorpus is on the order
of **3k Jev calls / ~$0.09**; scifact is similar. First MiniLM encode is local.
A crash is safe: Jev scores are cached per `(query, doc, mode)`.

Headline comparison printed as:

`minilm → minilm+jev-score  Δ nDCG@5 = …  95% CI […]`

If the CI excludes 0, that is a presentable improvement **on this stack**, not
vs Cohere or vs published BEIR BM25 numbers.

## Stronger first-stage

Same usage protocol with `BAAI/bge-small-en-v1.5` (or `e5-base`). Jev scores are
cached per `(query, doc, mode)`, so overlapping MiniLM hits are not re-billed.
`--timeout-s 30` is safer than 15s on long runs.

```bash
uv run python -m benchmark.run_benchmark --preset usage --dataset nfcorpus \
  --embed-model bge-small --timeout-s 30
```

If that Δ CI is still above 0, run SciFact the same way (`--dataset scifact`).
SciFact + BGE has not been run.

## Flags

| Flag | Smoke default | Usage preset |
| --- | --- | --- |
| `--dataset` | `nfcorpus` | same; also `scifact` |
| `--queries` / `--all-queries` | 5 | all test queries with qrels |
| `--retriever` | `bm25` | `dense` |
| `--embed-model` | `minilm` | same unless you pass `bge-small` / `e5-base` |
| `--include-bm25` | off | on |
| `--top-k` / `--top-n` | 10 / 5 | 10 / 5 |
| `--mode` | `noul` | `score` |
| `--timeout-s` | 15 | 15 |
| `--baseline-only` | off | skip Jev |
| `--no-cache` | off | off |

JSON → `benchmark/results/` (gitignored). Cache → `benchmark/cache/` (gitignored).
