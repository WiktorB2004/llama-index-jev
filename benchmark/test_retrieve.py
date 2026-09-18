from __future__ import annotations

from benchmark.retrieve import (
    BGE_QUERY_INSTRUCTION,
    embed_label,
    prefixed_texts,
    resolve_embed_model,
)


def test_resolve_embed_aliases() -> None:
    assert resolve_embed_model("minilm") == "sentence-transformers/all-MiniLM-L6-v2"
    assert resolve_embed_model("bge-small") == "BAAI/bge-small-en-v1.5"
    assert resolve_embed_model("e5-base") == "intfloat/e5-base-v2"
    assert resolve_embed_model("BAAI/bge-small-en-v1.5") == "BAAI/bge-small-en-v1.5"


def test_embed_label() -> None:
    assert embed_label("minilm") == "minilm"
    assert embed_label("BAAI/bge-small-en-v1.5") == "bge-small"


def test_minilm_has_no_prefix() -> None:
    assert prefixed_texts("minilm", ["hello"], is_query=True) == ["hello"]
    assert prefixed_texts("minilm", ["hello"], is_query=False) == ["hello"]


def test_bge_prefixes_queries_only() -> None:
    assert prefixed_texts("bge-small", ["q"], is_query=True) == [
        BGE_QUERY_INSTRUCTION + "q"
    ]
    assert prefixed_texts("bge-small", ["d"], is_query=False) == ["d"]


def test_e5_prefixes_query_and_passage() -> None:
    assert prefixed_texts("e5-base", ["q"], is_query=True) == ["query: q"]
    assert prefixed_texts("e5-base", ["d"], is_query=False) == ["passage: d"]
