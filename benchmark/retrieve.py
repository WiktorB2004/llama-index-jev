"""BM25 first-stage retrieval over the loaded corpus."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hit:
    doc_id: str
    text: str
    score: float


class BM25Index:
    def __init__(self, docs: dict[str, str]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "The benchmark extra is missing. Install with "
                "`uv sync --group benchmark`."
            ) from exc

        self.doc_ids = list(docs)
        self.texts = [docs[doc_id] for doc_id in self.doc_ids]
        self._bm25 = BM25Okapi([text.lower().split() for text in self.texts])

    def search(self, query: str, top_k: int) -> list[Hit]:
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(
            range(len(scores)), key=lambda index: float(scores[index]), reverse=True
        )[:top_k]
        return [
            Hit(
                doc_id=self.doc_ids[index],
                text=self.texts[index],
                score=float(scores[index]),
            )
            for index in ranked
        ]
