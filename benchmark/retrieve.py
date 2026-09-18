"""First-stage retrieval: BM25 and local sentence-transformer dense kNN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Hit:
    doc_id: str
    text: str
    score: float


class FirstStage(Protocol):
    def search(self, query: str, top_k: int) -> list[Hit]: ...


def _missing_extra() -> ImportError:
    return ImportError(
        "The benchmark extra is missing. Install with `uv sync --group benchmark`."
    )


EMBED_ALIASES = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "e5-base": "intfloat/e5-base-v2",
}

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def resolve_embed_model(name: str) -> str:
    """Map aliases such as ``bge-small`` to Hugging Face ids."""
    stripped = name.strip()
    return EMBED_ALIASES.get(stripped.lower(), stripped)


def embed_model_slug(model_name: str) -> str:
    return resolve_embed_model(model_name).replace("/", "_")


def embed_label(model_name: str) -> str:
    """Short id for tables and result filenames (``minilm``, ``bge-small``)."""
    resolved = resolve_embed_model(model_name)
    for alias, full in EMBED_ALIASES.items():
        if resolved == full:
            return alias
    return embed_model_slug(resolved).rsplit("_", 1)[-1]


def prefixed_texts(model_name: str, texts: list[str], *, is_query: bool) -> list[str]:
    """Apply retrieval prefixes required by BGE / E5. MiniLM is unchanged."""
    resolved = resolve_embed_model(model_name)
    lowered = resolved.lower()
    if "bge-" in lowered:
        if is_query:
            return [BGE_QUERY_INSTRUCTION + text for text in texts]
        return list(texts)
    if "e5-" in lowered:
        prefix = "query: " if is_query else "passage: "
        return [prefix + text for text in texts]
    return list(texts)


class BM25Index:
    def __init__(self, docs: dict[str, str]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise _missing_extra() from exc

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


class DenseIndex:
    """Cosine kNN over L2-normalized sentence-transformer embeddings."""

    def __init__(
        self,
        docs: dict[str, str],
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Path | None = None,
    ) -> None:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise _missing_extra() from exc

        self.doc_ids = list(docs)
        self.texts = [docs[doc_id] for doc_id in self.doc_ids]
        self.model_name = resolve_embed_model(model_name)
        self._np = np
        self._model = SentenceTransformer(self.model_name)
        self.emb = self._load_or_encode(cache_dir)

    def _encode(
        self, texts: list[str], *, is_query: bool, show_progress: bool = False
    ) -> object:
        return self._model.encode(
            prefixed_texts(self.model_name, texts, is_query=is_query),
            batch_size=64,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def _load_or_encode(self, cache_dir: Path | None) -> object:
        np = self._np
        path = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            path = cache_dir / f"embeddings_{embed_model_slug(self.model_name)}.npz"
            if path.exists():
                loaded = np.load(path, allow_pickle=True)
                cached_ids = [str(doc_id) for doc_id in loaded["doc_ids"].tolist()]
                if cached_ids == self.doc_ids:
                    print(f"Loaded cached embeddings from {path}", flush=True)
                    return loaded["emb"]
        print(
            f"Encoding {len(self.texts)} docs with {self.model_name}...",
            flush=True,
        )
        emb = self._encode(self.texts, is_query=False, show_progress=True)
        if path is not None:
            np.savez_compressed(
                path,
                emb=emb,
                doc_ids=np.array(self.doc_ids, dtype=object),
            )
        return emb

    def search(self, query: str, top_k: int) -> list[Hit]:
        np = self._np
        query_vec = self._encode([query], is_query=True)[0]
        scores = self.emb @ query_vec
        ranked = np.argsort(scores)[::-1][:top_k]
        return [
            Hit(
                doc_id=self.doc_ids[int(index)],
                text=self.texts[int(index)],
                score=float(scores[int(index)]),
            )
            for index in ranked
        ]
