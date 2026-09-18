from __future__ import annotations

from benchmark.run_benchmark import parse_args


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.preset == "smoke"
    assert args.dataset == "nfcorpus"
    assert args.queries == 5
    assert args.all_queries is False
    assert args.top_k == 10
    assert args.top_n == 5
    assert args.mode == "noul"
    assert args.retriever == "bm25"
    assert args.include_bm25 is False
    assert args.provider == "openrouter"
    assert args.timeout_s == 15.0
    assert args.embed_model == "sentence-transformers/all-MiniLM-L6-v2"


def test_parse_args_usage_preset() -> None:
    args = parse_args(["--preset", "usage", "--dataset", "scifact"])
    assert args.preset == "usage"
    assert args.dataset == "scifact"
    assert args.all_queries is True
    assert args.retriever == "dense"
    assert args.mode == "score"
    assert args.top_k == 10
    assert args.top_n == 5
    assert args.include_bm25 is True
    assert args.embed_model == "sentence-transformers/all-MiniLM-L6-v2"


def test_parse_args_bge_alias_with_usage_preset() -> None:
    args = parse_args(
        ["--preset", "usage", "--dataset", "nfcorpus", "--embed-model", "bge-small"]
    )
    assert args.retriever == "dense"
    assert args.embed_model == "BAAI/bge-small-en-v1.5"
    assert args.mode == "score"
