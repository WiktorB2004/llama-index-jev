"""Load a small BEIR split from ir_datasets."""

from __future__ import annotations

from dataclasses import dataclass

DATASET_IDS = {
    "nfcorpus": "beir/nfcorpus/test",
    "scifact": "beir/scifact/test",
}


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    qrels: dict[str, int]


@dataclass(frozen=True)
class BenchmarkSplit:
    name: str
    docs: dict[str, str]
    queries: list[Query]


def _doc_text(doc: object) -> str:
    title = getattr(doc, "title", None) or ""
    body = getattr(doc, "text", None) or getattr(doc, "body", None) or ""
    return f"{title} {body}".strip()


def _query_text(query: object) -> str:
    for attr in ("text", "query", "title"):
        value = getattr(query, attr, None)
        if value:
            return str(value).strip()
    raise ValueError(f"Query {query!r} has no text field")


def load_split(name: str, n_queries: int | None) -> BenchmarkSplit:
    try:
        import ir_datasets
    except ImportError as exc:
        raise ImportError(
            "The benchmark extra is missing. Install with `uv sync --group benchmark`."
        ) from exc

    if name not in DATASET_IDS:
        known = ", ".join(sorted(DATASET_IDS))
        raise ValueError(f"Unknown dataset {name!r}; expected one of: {known}")

    dataset_id = DATASET_IDS[name]
    dataset = ir_datasets.load(dataset_id)
    docs_source = dataset
    if not dataset.has_docs():
        parent_id = dataset_id.rsplit("/", 1)[0]
        docs_source = ir_datasets.load(parent_id)

    docs = {str(doc.doc_id): _doc_text(doc) for doc in docs_source.docs_iter()}
    qrels_dict = {
        str(qid): {str(did): int(rel) for did, rel in rels.items() if int(rel) > 0}
        for qid, rels in dataset.qrels_dict().items()
    }

    selected: list[Query] = []
    for query in dataset.queries_iter():
        query_id = str(query.query_id)
        qrels = qrels_dict.get(query_id) or {}
        if not qrels:
            continue
        selected.append(Query(query_id=query_id, text=_query_text(query), qrels=qrels))
        if n_queries is not None and len(selected) >= n_queries:
            break

    if n_queries is not None and len(selected) < n_queries:
        raise RuntimeError(
            f"{dataset_id} only has {len(selected)} queries with qrels; "
            f"requested {n_queries}"
        )
    if not selected:
        raise RuntimeError(f"{dataset_id} has no queries with qrels")
    return BenchmarkSplit(name=name, docs=docs, queries=selected)
