"""JSON-on-disk cache so a crashed smoke run does not re-bill Jev calls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, Any] = {}
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self._data = loaded

    def get(self, key: str) -> Any | None:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8"
        )
        tmp.replace(self.path)


def first_stage_key(query_id: str) -> str:
    return query_id


def jev_score_key(
    query_id: str,
    doc_id: str,
    *,
    mode: str,
    provider: str,
    model: str,
) -> str:
    return f"{query_id}\t{doc_id}\t{mode}\t{provider}\t{model}"
