"""LlamaIndex-shaped retrieval eval: dense (or BM25) first-stage vs JevRerank.

Smoke (cheap, ~50 Jev calls):

    uv run python -m benchmark.run_benchmark --provider openrouter --queries 5

Usage protocol (full test split, MiniLM top-10, Jev default score, top_n=5):

    uv run python -m benchmark.run_benchmark --preset usage --dataset nfcorpus
    uv run python -m benchmark.run_benchmark --preset usage --dataset scifact
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
from benchmark.metrics import (
    add_usage,
    bootstrap_delta_ci,
    bootstrap_mean_ci,
    mean,
    ndcg_at_k,
    parse_usage,
    percentile,
)
from benchmark.retrieve import (
    BM25Index,
    DenseIndex,
    FirstStage,
    Hit,
    embed_label,
    embed_model_slug,
    resolve_embed_model,
)

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
    index: FirstStage,
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


def _truncate_hits(
    hits_by_query: dict[str, list[Hit]], top_n: int
) -> dict[str, list[str]]:
    return {
        query_id: [hit.doc_id for hit in hits[:top_n]]
        for query_id, hits in hits_by_query.items()
    }


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
    nodes = _hits_to_nodes(hits)
    scores: list[float | None] = [None] * len(hits)
    usages: list[dict[str, float] | None] = [None] * len(hits)
    pending: list[tuple[int, NodeWithScore, Hit]] = []
    cache_hits = 0
    started = time.perf_counter()
    for index, (node, hit) in enumerate(zip(nodes, hits, strict=True)):
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
        for node, score in zip(nodes, scores, strict=True)
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


def _ndcg_lists(
    rankings: dict[str, list[str]], split: BenchmarkSplit
) -> tuple[list[float], list[float]]:
    ndcg5 = [
        ndcg_at_k(rankings[query.query_id], query.qrels, 5) for query in split.queries
    ]
    ndcg10 = [
        ndcg_at_k(rankings[query.query_id], query.qrels, 10) for query in split.queries
    ]
    return ndcg5, ndcg10


def _summarize(
    name: str,
    rankings: dict[str, list[str]],
    split: BenchmarkSplit,
    *,
    seed: int,
    latencies: list[float] | None = None,
    usages: list[dict[str, float]] | None = None,
    api_calls: int = 0,
    cache_hits: int = 0,
) -> tuple[dict[str, Any], list[float]]:
    ndcg5, ndcg10 = _ndcg_lists(rankings, split)
    ci5 = bootstrap_mean_ci(ndcg5, seed=seed)
    ci10 = bootstrap_mean_ci(ndcg10, seed=seed)
    summary: dict[str, Any] = {
        "system": name,
        "ndcg@5": mean(ndcg5),
        "ndcg@5_ci95": list(ci5) if ci5[0] is not None else None,
        "ndcg@10": mean(ndcg10),
        "ndcg@10_ci95": list(ci10) if ci10[0] is not None else None,
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
        summary["cost_usd_per_query"] = total["cost_usd"] / max(len(split.queries), 1)
        summary["cost_estimated"] = bool(total["cost_estimated"])
    return summary, ndcg5


def _fmt_ndcg(value: float, ci: Any) -> str:
    if not ci or ci[0] is None:
        return f"{value:.4f}"
    return f"{value:.4f} [{ci[0]:.3f},{ci[1]:.3f}]"


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "system",
        "ndcg@5",
        "ndcg@10",
        "p50_s",
        "usd/q",
        "calls",
        "cache",
        "usd",
    ]
    print("  ".join(f"{header:>22}" for header in headers))
    for row in rows:
        values = [
            str(row.get("system", "")),
            _fmt_ndcg(float(row.get("ndcg@5", 0.0)), row.get("ndcg@5_ci95")),
            _fmt_ndcg(float(row.get("ndcg@10", 0.0)), row.get("ndcg@10_ci95")),
            _fmt_opt(row.get("latency_p50_s"), "{:.3f}"),
            _fmt_opt(row.get("cost_usd_per_query"), "{:.6f}"),
            _fmt_opt(row.get("api_calls"), "{}"),
            _fmt_opt(row.get("cache_hits"), "{}"),
            _fmt_cost(row.get("cost_usd"), bool(row.get("cost_estimated"))),
        ]
        print("  ".join(f"{value:>22}" for value in values))


def _fmt_opt(value: Any, fmt: str) -> str:
    if value is None:
        return "-"
    return fmt.format(value)


def _fmt_cost(value: Any, estimated: bool) -> str:
    if value is None:
        return "-"
    text = f"{float(value):.6f}"
    return f"~{text}" if estimated else text


def _first_stage_label(args: argparse.Namespace) -> str:
    if args.retriever == "dense":
        return embed_label(args.embed_model)
    return str(args.retriever)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense/BM25 first-stage vs JevRerank on a BEIR split."
    )
    parser.add_argument(
        "--preset",
        choices=("smoke", "usage"),
        default="smoke",
        help="smoke: 5 queries, BM25. usage: full split, MiniLM, Jev score.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_IDS),
        default="nfcorpus",
        help="BEIR test split (default: nfcorpus).",
    )
    parser.add_argument("--queries", type=int, default=5)
    parser.add_argument(
        "--all-queries",
        action="store_true",
        dest="all_queries",
        help="Use every test query that has qrels.",
    )
    parser.add_argument("--top-k", type=int, default=10, dest="top_k")
    parser.add_argument("--top-n", type=int, default=5, dest="top_n")
    parser.add_argument("--mode", choices=("noul", "score"), default="noul")
    parser.add_argument(
        "--retriever",
        choices=("bm25", "dense"),
        default="bm25",
        help="First-stage for the Jev candidate pool.",
    )
    parser.add_argument(
        "--embed-model",
        default="minilm",
        dest="embed_model",
        help="Dense first-stage: minilm, bge-small, e5-base, or a Hugging Face id.",
    )
    parser.add_argument(
        "--include-bm25",
        action="store_true",
        dest="include_bm25",
        help="Also report a BM25 top_n baseline (usage preset turns this on).",
    )
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
    parser.add_argument("--seed", type=int, default=0)
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
    args.embed_model = resolve_embed_model(args.embed_model)
    if args.preset == "usage":
        args.all_queries = True
        args.retriever = "dense"
        args.mode = "score"
        args.top_k = 10
        args.top_n = 5
        args.include_bm25 = True
    if args.queries < 1:
        parser.error("--queries must be >= 1")
    if args.top_k < 1 or args.top_n < 1:
        parser.error("--top-k and --top-n must be >= 1")
    if args.top_n > args.top_k:
        parser.error("--top-n cannot exceed --top-k")
    return args


def _run_jev(
    *,
    args: argparse.Namespace,
    split: BenchmarkSplit,
    hits_by_query: dict[str, list[Hit]],
    cache: JsonCache | None,
) -> tuple[dict[str, list[str]], dict[str, Any], list[float]]:
    reranker = JevRerank(
        provider=args.provider,
        model=args.model,
        top_n=args.top_n,
        mode=args.mode,
        timeout_s=args.timeout_s,
        max_concurrency=args.max_concurrency,
        raise_on_error=True,
    )
    rankings: dict[str, list[str]] = {}
    latencies: list[float] = []
    usages: list[dict[str, float]] = []
    api_calls = 0
    cache_hits = 0
    stage = _first_stage_label(args)
    print(
        f"Reranking with Jev ({args.provider}, mode={args.mode}, "
        f"timeout_s={args.timeout_s}) over {stage} top-{args.top_k}...",
        flush=True,
    )
    for query in split.queries:
        ranked, query_usage, calls, hits, elapsed = _rerank_query(
            reranker,
            query.text,
            query.query_id,
            hits_by_query[query.query_id],
            top_n=args.top_n,
            cache=cache,
            provider=args.provider,
            model=args.model,
            mode=args.mode,
        )
        rankings[query.query_id] = ranked
        latencies.append(elapsed)
        usages.extend(query_usage)
        api_calls += calls
        cache_hits += hits
        print(
            f"  {query.query_id}: {calls} api / {hits} cache  {elapsed:.2f}s",
            flush=True,
        )
    jev_name = f"{_first_stage_label(args)}+jev-{args.mode}"
    summary, ndcg5 = _summarize(
        jev_name,
        rankings,
        split,
        seed=args.seed,
        latencies=latencies,
        usages=usages,
        api_calls=api_calls,
        cache_hits=cache_hits,
    )
    return rankings, summary, ndcg5


def run(args: argparse.Namespace) -> dict[str, Any]:
    n_queries = None if args.all_queries else args.queries
    query_label = "all" if n_queries is None else str(n_queries)
    stage = _first_stage_label(args)
    print(
        f"Loading {args.dataset} ({query_label} queries, "
        f"{stage} top_k={args.top_k})...",
        flush=True,
    )
    split = load_split(args.dataset, n_queries)
    cache_root = None if args.no_cache else args.cache_dir / args.dataset
    if cache_root is not None:
        cache_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    ndcg5_by_system: dict[str, list[float]] = {}
    rankings_out: dict[str, dict[str, list[str]]] = {}

    jev_hits: dict[str, list[Hit]] | None = None

    if args.retriever == "bm25" or args.include_bm25:
        bm25_cache = (
            JsonCache(cache_root / f"bm25_top{args.top_k}.json")
            if cache_root is not None
            else None
        )
        print("Building BM25 index...", flush=True)
        bm25_hits = _load_or_retrieve(
            BM25Index(split.docs), split, top_k=args.top_k, cache=bm25_cache
        )
        bm25_rankings = _truncate_hits(bm25_hits, args.top_n)
        summary, ndcg5 = _summarize("bm25", bm25_rankings, split, seed=args.seed)
        rows.append(summary)
        ndcg5_by_system["bm25"] = ndcg5
        rankings_out["bm25"] = bm25_rankings
        if args.retriever == "bm25":
            jev_hits = bm25_hits

    if args.retriever == "dense":
        dense_cache = None
        if cache_root is not None:
            dense_cache = JsonCache(
                cache_root
                / f"dense_{embed_model_slug(args.embed_model)}_top{args.top_k}.json"
            )
        dense_index = DenseIndex(
            split.docs, model_name=args.embed_model, cache_dir=cache_root
        )
        dense_hits = _load_or_retrieve(
            dense_index, split, top_k=args.top_k, cache=dense_cache
        )
        dense_rankings = _truncate_hits(dense_hits, args.top_n)
        dense_name = embed_label(args.embed_model)
        summary, ndcg5 = _summarize(dense_name, dense_rankings, split, seed=args.seed)
        rows.append(summary)
        ndcg5_by_system[dense_name] = ndcg5
        rankings_out[dense_name] = dense_rankings
        jev_hits = dense_hits

    jev_rankings: dict[str, list[str]] = {}
    if not args.baseline_only:
        if jev_hits is None:
            raise RuntimeError("No first-stage hits for Jev")
        jev_cache = None
        if cache_root is not None:
            jev_cache = JsonCache(
                cache_root / f"jev_{args.provider}_{args.mode}_{args.model}.json"
            )
        jev_rankings, jev_summary, jev_ndcg5 = _run_jev(
            args=args, split=split, hits_by_query=jev_hits, cache=jev_cache
        )
        rows.append(jev_summary)
        ndcg5_by_system[str(jev_summary["system"])] = jev_ndcg5
        rankings_out[str(jev_summary["system"])] = jev_rankings

    comparisons: list[dict[str, Any]] = []
    baseline_name = _first_stage_label(args)
    jev_name = f"{baseline_name}+jev-{args.mode}"
    if baseline_name in ndcg5_by_system and jev_name in ndcg5_by_system:
        base = ndcg5_by_system[baseline_name]
        treat = ndcg5_by_system[jev_name]
        delta = mean(treat) - mean(base)
        ci = bootstrap_delta_ci(base, treat, seed=args.seed)
        comparisons.append(
            {
                "baseline": baseline_name,
                "treatment": jev_name,
                "delta_ndcg@5": delta,
                "delta_ndcg@5_ci95": list(ci) if ci[0] is not None else None,
            }
        )

    payload = {
        "preset": args.preset,
        "dataset": args.dataset,
        "dataset_id": DATASET_IDS[args.dataset],
        "queries": len(split.queries),
        "all_queries": args.all_queries,
        "top_k": args.top_k,
        "top_n": args.top_n,
        "mode": args.mode,
        "provider": args.provider,
        "model": args.model,
        "retriever": args.retriever,
        "embed_model": args.embed_model if args.retriever == "dense" else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "systems": rows,
        "comparisons": comparisons,
        "rankings": rankings_out,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    extra = f"_{_first_stage_label(args)}" if args.retriever == "dense" else ""
    out_path = args.output_dir / f"{args.preset}_{args.dataset}{extra}_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    _print_table(rows)
    for comparison in comparisons:
        ci = comparison["delta_ndcg@5_ci95"]
        ci_text = (
            f"  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci and ci[0] is not None else ""
        )
        print(
            f"\n{comparison['baseline']} → {comparison['treatment']}  "
            f"Δ nDCG@5 = {comparison['delta_ndcg@5']:+.4f}{ci_text}"
        )
    print(f"\nWrote {out_path}")
    return payload


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    main(sys.argv[1:])
