from __future__ import annotations

from benchmark.metrics import (
    JEV_INPUT_USD_PER_MILLION,
    dcg_at_k,
    mean,
    ndcg_at_k,
    parse_usage,
    percentile,
)


def test_ndcg_perfect_ranking() -> None:
    qrels = {"a": 2, "b": 1, "c": 0}
    assert ndcg_at_k(["a", "b", "c"], qrels, k=3) == 1.0


def test_ndcg_swapped_pair_is_worse() -> None:
    qrels = {"a": 2, "b": 1}
    perfect = ndcg_at_k(["a", "b"], qrels, k=2)
    swapped = ndcg_at_k(["b", "a"], qrels, k=2)
    assert 0.0 < swapped < perfect == 1.0


def test_ndcg_unjudged_is_zero_gain() -> None:
    qrels = {"rel": 1}
    score = ndcg_at_k(["noise", "rel"], qrels, k=2)
    # First slot is unjudged; DCG is only the discounted second-place rel.
    expected_dcg = dcg_at_k([0.0, 1.0], 2)
    expected_idcg = dcg_at_k([1.0], 2)
    assert score == expected_dcg / expected_idcg


def test_ndcg_no_relevant_docs_is_zero() -> None:
    assert ndcg_at_k(["a"], {}, k=1) == 0.0


def test_percentile_empty_and_singleton() -> None:
    assert percentile([], 50) is None
    assert percentile([4.0], 95) == 4.0


def test_percentile_interpolates() -> None:
    values = [0.0, 10.0]
    assert percentile(values, 50) == 5.0


def test_mean_empty_is_zero() -> None:
    assert mean([]) == 0.0


def test_parse_usage_openrouter_shape() -> None:
    parsed = parse_usage({"prompt_tokens": 100, "completion_tokens": 2, "cost": 0.001})
    assert parsed["input_tokens"] == 100.0
    assert parsed["output_tokens"] == 2.0
    assert parsed["cost_usd"] == 0.001
    assert parsed["cost_estimated"] == 0.0


def test_parse_usage_estimates_from_jev_rate() -> None:
    parsed = parse_usage({"input_tokens": 1_000_000})
    assert parsed["cost_usd"] == JEV_INPUT_USD_PER_MILLION
    assert parsed["cost_estimated"] == 1.0


def test_parse_usage_reads_cached_cost_usd() -> None:
    parsed = parse_usage({"input_tokens": 10, "output_tokens": 1, "cost_usd": 0.0002})
    assert parsed["cost_usd"] == 0.0002
    assert parsed["cost_estimated"] == 0.0
