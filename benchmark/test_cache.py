from __future__ import annotations

from pathlib import Path

from benchmark.cache import JsonCache, jev_score_key


def test_json_cache_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    cache = JsonCache(path)
    cache.set("q1", {"score": 0.9})
    reloaded = JsonCache(path)
    assert reloaded.get("q1") == {"score": 0.9}


def test_json_cache_missing_key(tmp_path: Path) -> None:
    cache = JsonCache(tmp_path / "missing.json")
    assert cache.get("nope") is None


def test_jev_score_key_includes_mode_and_provider() -> None:
    left = jev_score_key(
        "q", "d", mode="noul", provider="openrouter", model="jev-latest"
    )
    right = jev_score_key(
        "q", "d", mode="score", provider="openrouter", model="jev-latest"
    )
    assert left != right
