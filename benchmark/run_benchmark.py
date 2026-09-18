"""Smoke: BM25 first-stage vs JevRerank on a tiny BEIR split.

Default flags keep the run around 50 Jev calls (5 queries × top_k 10):

    uv sync --group benchmark
    export OPENROUTER_API_KEY=...
    uv run python -m benchmark.run_benchmark --provider openrouter --queries 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.postprocessor.jev import JevRerank
from llama_index.postprocessor.jev.utils import build_relevance_question, get_answer

from benchmark.cache import JsonCache, first_stage_key, jev_score_key
from benchmark.data import DATASET_IDS, BenchmarkSplit, load_split
from benchmark.metrics import add_usage, mean, ndcg_at_k, parse_usage, percentile
from benchmark.retrieve import BM25Index, Hit

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / "benchmark" / "cache"
DEFAULT_OUTPUT_DIR = ROOT / "benchmark" / "results"


def _hits_to_nodes(hits: list[Hit]) -> list[NodeWithScore]:
    nodes: list[NodeWithScore] = []
    for hit in hits:
        node = TextNode(text=hit.text, id_=hit.doc_id, metadata={"doc_id": hit.doc_id})
        nodes.append(NodeWithScore(node=node, score=hit.score))
    return nodes


def _node_id(node: NodeWithScore) -> str:
    metadata = node.node.metadata or {}
    doc_id = metadata.get("doc_id")
    if isinstance(doc_id, str) and doc_id:
        return doc_id
    return node.node.node_id


def _score_one(
    reranker: JevRerank, node: NodeWithScore, query_str: str
) -> tuple[float, dict[str, float]]:
    response = reranker._client.system_one(
        state=reranker._state_for(node, query_str),
        questions={"relevance": build_relevance_question(reranker.mode)},
        model=reranker.model,
    )
    usage = parse_usage(getattr(response, "usage", None))
    answer = get_answer(response, "relevance", reranker.mode)
    scored = reranker._apply_answer(node, answer)
    return float(scored.score or 0.0), usage


def _load_or_retrieve(
    index: BM25Index,
    split: BenchmarkSplit,
    *,
    top_k: int,
    cache: JsonCache | None,
) -> dict[str, list[Hit]]:
    hits_by_query: dict[str, list[Hit]] = {}
    for query in split.queries:
        cached = cache.get(first_stage_key(query.query_id)) if cache else None
        if isinstance(cached, dict) and cached.get("query") == query.text:
            raw_hits = cached.get("hits") or []
            if isinstance(raw_hits, list) and len(raw_hits) >= top_k:
                hits_by_query[query.query_id] = [
                    Hit(
                        doc_id=str(item["doc_id"]),
                        text=str(item["text"]),
                        score=float(item["score"]),
                    )
                    for item in raw_hits[:top_k]
                ]
                continue
        hits = index.search(query.text, top_k)
        hits_by_query[query.query_id] = hits
        if cache is not None:
            cache.set(
                first_stage_key(query.query_id),
                {
                    "query": query.text,
                    "hits": [
                        {"doc_id": hit.doc_id, "text": hit.text, "score": hit.score}
                        for hit in hits
                    ],
                },
            )
    return hits_by_query


def _rerank_query(
    reranker: JevRerank,
    query_text: str,
    query_id: str,
    hits: list[Hit],
    *,
    top_n: int,
    cache: JsonCache | None,
    provider: str,
    model: str,
    mode: str,
) -> tuple[list[str], list[dict[str, float]], int, int, float]:
    """Return ranked ids, per-call usage, api_calls, cache_hits, wall seconds."""
    nodes = _hits_to_nodes(hits)
    scores: list[float | None] = [None] * len(hits)
    usages: list[dict[str, float] | None] = [None] * len(hits)
    pending: list[tuple[int, NodeWithScore, Hit]] = []
    cache_hits = 0
    started = time.perf_counter()
    for index, (node, hit) in enumerate(zip(nodes, hits)):
        key = jev_score_key(
            query_id, hit.doc_id, mode=mode, provider=provider, model=model
        )
        cached = cache.get(key) if cache else None
        if isinstance(cached, dict) and "score" in cached:
            scores[index] = float(cached["score"])
            usages[index] = parse_usage(cached.get("usage"))
            cache_hits += 1
        else:
            pending.append((index, node, hit))

    api_calls = 0
    if pending:
        with ThreadPoolExecutor(max_workers=reranker.max_concurrency) as pool:
            future_map = {
                pool.submit(_score_one, reranker, node, query_text): (index, hit)
                for index, node, hit in pending
            }
            for future in as_completed(future_map):
                index, hit = future_map[future]
                score, usage = future.result()
                scores[index] = score
                usages[index] = usage
                api_calls += 1
                if cache is not None:
                    cache.set(
                        jev_score_key(
                            query_id,
                            hit.doc_id,
                            mode=mode,
                            provider=provider,
                            model=model,
                        ),
                        {
                            "score": score,
                            "usage": {
                                "input_tokens": usage["input_tokens"],
                                "output_tokens": usage["output_tokens"],
                                "cost_usd": usage["cost_usd"],
                            },
                        },
                    )

    scored = [
        NodeWithScore(node=node.node, score=score or 0.0)
        for node, score in zip(nodes, scores)
    ]
    ranked = sorted(scored, key=lambda item: item.score or 0.0, reverse=True)[:top_n]
    elapsed = time.perf_counter() - started
    recorded_usage = [usage for usage in usages if usage is not None]
    return (
        [_node_id(node) for node in ranked],
        recorded_usage,
        api_calls,
        cache_hits,
        elapsed,
    )


def _summarize(
    name: str,
    rankings: dict[str, list[str]],
    split: BenchmarkSplit,
    *,
    latencies: list[float] | None = None,
    usages: list[dict[str, float]] | None = None,
    api_calls: int = 0,
    cache_hits: int = 0,
) -> dict[str, Any]:
    qrels_by_id = {query.query_id: query.qrels for query in split.queries}
    ndcg5 = [
        ndcg_at_k(rankings[query.query_id], qrels_by_id[query.query_id], 5)
        for query in split.queries
    ]
    ndcg10 = [
        ndcg_at_k(rankings[query.query_id], qrels_by_id[query.query_id], 10)
        for query in split.queries
    ]
    summary: dict[str, Any] = {
        "system": name,
        "ndcg@5": mean(ndcg5),
        "ndcg@10": mean(ndcg10),
        "queries": len(split.queries),
    }
    if latencies is not None:
        summary["latency_p50_s"] = percentile(latencies, 50)
        summary["latency_p95_s"] = percentile(latencies, 95)
        summary["latency_mean_s"] = mean(latencies)
        summary["api_calls"] = api_calls
        summary["cache_hits"] = cache_hits
    if usages:
        total = {
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "cost_usd": 0.0,
            "cost_estimated": 0.0,
        }
        for usage in usages:
            add_usage(total, usage)
        summary["input_tokens"] = int(total["input_tokens"])
        summary["output_tokens"] = int(total["output_tokens"])
        summary["cost_usd"] = total["cost_usd"]
        summary["cost_estimated"] = bool(total["cost_estimated"])
    return summary


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "system",
        "ndcg@5",
        "ndcg@10",
        "p50_s",
        "p95_s",
        "calls",
        "cache",
        "tokens_in",
        "usd",
    ]
    print("  ".join(f"{header:>10}" for header in headers))
    for row in rows:
        values = [
            str(row.get("system", "")),
            f"{row.get('ndcg@5', 0.0):.4f}",
            f"{row.get('ndcg@10', 0.0):.4f}",
            _fmt_opt(row.get("latency_p50_s"), "{:.3f}"),
            _fmt_opt(row.get("latency_p95_s"), "{:.3f}"),
            _fmt_opt(row.get("api_calls"), "{}"),
            _fmt_opt(row.get("cache_hits"), "{}"),
            _fmt_opt(row.get("input_tokens"), "{}"),
            _fmt_cost(row.get("cost_usd"), bool(row.get("cost_estimated"))),
        ]
        print("  ".join(f"{value:>10}" for value in values))


def _fmt_opt(value: Any, fmt: str) -> str:
    if value is None:
        return "-"
    return fmt.format(value)


def _fmt_cost(value: Any, estimated: bool) -> str:
    if value is None:
        return "-"
    text = f"{float(value):.6f}"
    return f"~{text}" if estimated else text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke BM25 vs JevRerank on a BEIR subset."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_IDS),
        default="nfcorpus",
        help="BEIR test split (default: nfcorpus).",
    )
    parser.add_argument("--queries", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10, dest="top_k")
    parser.add_argument("--top-n", type=int, default=5, dest="top_n")
    parser.add_argument("--mode", choices=("noul", "score"), default="noul")
    parser.add_argument(
        "--provider",
        choices=("openrouter", "typesafe"),
        default="openrouter",
    )
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--timeout-s", type=float, default=15.0, dest="timeout_s")
    parser.add_argument(
        "--max-concurrency", type=int, default=8, dest="max_concurrency"
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read or write the on-disk score cache.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Skip Jev (no API key required).",
    )
    args = parser.parse_args(argv)
    if args.queries < 1:
        parser.error("--queries must be >= 1")
    if args.top_k < 1 or args.top_n < 1:
        parser.error("--top-k and --top-n must be >= 1")
    if args.top_n > args.top_k:
        parser.error("--top-n cannot exceed --top-k")
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    print(
        f"Loading {args.dataset} ({args.queries} queries, BM25 top_k={args.top_k})...",
        flush=True,
    )
    split = load_split(args.dataset, args.queries)
    index = BM25Index(split.docs)

    first_stage_cache: JsonCache | None = None
    jev_cache: JsonCache | None = None
    if not args.no_cache:
        cache_root = args.cache_dir / args.dataset
        first_stage_cache = JsonCache(cache_root / f"bm25_top{args.top_k}.json")
        jev_cache = JsonCache(
            cache_root / f"jev_{args.provider}_{args.mode}_{args.model}.json"
        )

    hits_by_query = _load_or_retrieve(
        index, split, top_k=args.top_k, cache=first_stage_cache
    )

    baseline_rankings = {
        query.query_id: [
            hit.doc_id for hit in hits_by_query[query.query_id][: args.top_n]
        ]
        for query in split.queries
    }
    rows = [_summarize("bm25", baseline_rankings, split)]

    jev_rankings: dict[str, list[str]] = {}
    if not args.baseline_only:
        reranker = JevRerank(
            provider=args.provider,
            model=args.model,
            top_n=args.top_n,
            mode=args.mode,
            timeout_s=args.timeout_s,
            max_concurrency=args.max_concurrency,
            raise_on_error=True,
        )
        latencies: list[float] = []
        usages: list[dict[str, float]] = []
        api_calls = 0
        cache_hits = 0
        print(
            f"Reranking with Jev ({args.provider}, mode={args.mode}, "
            f"timeout_s={args.timeout_s})...",
            flush=True,
        )
        for query in split.queries:
            ranked, query_usage, calls, hits, elapsed = _rerank_query(
                reranker,
                query.text,
                query.query_id,
                hits_by_query[query.query_id],
                top_n=args.top_n,
                cache=jev_cache,
                provider=args.provider,
                model=args.model,
                mode=args.mode,
            )
            jev_rankings[query.query_id] = ranked
            latencies.append(elapsed)
            usages.extend(query_usage)
            api_calls += calls
            cache_hits += hits
            print(
                f"  {query.query_id}: {calls} api / {hits} cache  {elapsed:.2f}s",
                flush=True,
            )
        rows.append(
            _summarize(
                f"jev-{args.mode}",
                jev_rankings,
                split,
                latencies=latencies,
                usages=usages,
                api_calls=api_calls,
                cache_hits=cache_hits,
            )
        )

    payload = {
        "dataset": args.dataset,
        "dataset_id": DATASET_IDS[args.dataset],
        "queries": args.queries,
        "top_k": args.top_k,
        "top_n": args.top_n,
        "mode": args.mode,
        "provider": args.provider,
        "model": args.model,
        "retriever": "bm25",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "systems": rows,
        "baseline_rankings": baseline_rankings,
        "jev_rankings": jev_rankings,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.output_dir / f"smoke_{args.dataset}_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    _print_table(rows)
    print(f"\nWrote {out_path}")
    return payload


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
