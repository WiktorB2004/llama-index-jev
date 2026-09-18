from __future__ import annotations

from benchmark.run_benchmark import parse_args


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.dataset == "nfcorpus"
    assert args.queries == 5
    assert args.top_k == 10
    assert args.top_n == 5
    assert args.mode == "noul"
    assert args.provider == "openrouter"
    assert args.timeout_s == 15.0
