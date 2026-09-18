"""nDCG, latency percentiles, bootstrap CIs, and OpenRouter usage totals."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

JEV_INPUT_USD_PER_MILLION = 0.042


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    total = 0.0
    for rank, rel in enumerate(relevances[:k], start=1):
        total += (2.0**rel - 1.0) / math.log2(rank + 1)
    return total


def ndcg_at_k(ranked_ids: Sequence[str], qrels: Mapping[str, float], k: int) -> float:
    """nDCG@k for one query. Unjudged docs score 0. No relevant docs → 0."""
    gains = [float(qrels.get(doc_id, 0.0)) for doc_id in ranked_ids[:k]]
    ideal = sorted((float(rel) for rel in qrels.values() if rel > 0), reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(gains, k) / idcg


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def percentile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def parse_usage(raw: Any) -> dict[str, float]:
    """Pull input/output tokens and USD from an OpenRouter or SDK usage object."""
    if raw is None:
        return {
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "cost_usd": 0.0,
            "cost_estimated": 0.0,
        }
    if not isinstance(raw, Mapping):
        raw = {
            "input_tokens": getattr(raw, "input_tokens", None),
            "prompt_tokens": getattr(raw, "prompt_tokens", None),
            "output_tokens": getattr(raw, "output_tokens", None),
            "completion_tokens": getattr(raw, "completion_tokens", None),
            "cost": getattr(raw, "cost", None),
            "total_cost": getattr(raw, "total_cost", None),
        }
    input_tokens = float(raw.get("input_tokens") or raw.get("prompt_tokens") or 0.0)
    output_tokens = float(
        raw.get("output_tokens") or raw.get("completion_tokens") or 0.0
    )
    cost = raw.get("cost")
    if cost is None:
        cost = raw.get("total_cost")
    if cost is None:
        cost = raw.get("cost_usd")
    if cost is None:
        cost_usd = (input_tokens / 1_000_000.0) * JEV_INPUT_USD_PER_MILLION
        estimated = True
    else:
        cost_usd = float(cost)
        estimated = False
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "cost_estimated": float(estimated),
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    p: float = 0.95,
    seed: int = 0,
) -> tuple[float, float] | tuple[None, None]:
    """Percentile bootstrap CI for the mean. Needs at least two observations."""
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1.0 - p) / 2.0
    low_index = math.floor(alpha * (n_boot - 1))
    high_index = math.ceil((1.0 - alpha) * (n_boot - 1))
    high_index = min(high_index, n_boot - 1)
    return means[low_index], means[high_index]


def bootstrap_delta_ci(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    n_boot: int = 1000,
    p: float = 0.95,
    seed: int = 0,
) -> tuple[float, float] | tuple[None, None]:
    """CI for mean(treatment - baseline) with paired query resampling."""
    if len(baseline) != len(treatment):
        raise ValueError("baseline and treatment must have the same length")
    deltas = [t - b for t, b in zip(treatment, baseline)]
    return bootstrap_mean_ci(deltas, n_boot=n_boot, p=p, seed=seed)


def add_usage(total: dict[str, float], one: Mapping[str, float]) -> None:
    total["input_tokens"] = total.get("input_tokens", 0.0) + float(
        one.get("input_tokens", 0.0)
    )
    total["output_tokens"] = total.get("output_tokens", 0.0) + float(
        one.get("output_tokens", 0.0)
    )
    total["cost_usd"] = total.get("cost_usd", 0.0) + float(one.get("cost_usd", 0.0))
    if one.get("cost_estimated"):
        total["cost_estimated"] = 1.0
