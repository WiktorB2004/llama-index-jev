# llama-index-postprocessor-jev

Drop-in `JevRerank` for [LlamaIndex](https://www.llamaindex.ai): score each retrieved passage with [TypeSafe Jev](https://typesafe.ai), keep the top `n`. Typed 0–3 scores, not a cross-encoder and not an LLM-as-judge loop.

Full story, selector package, and benchmark numbers: [docs](https://wiktorb2004.github.io/llama-index-jev/) · [repo](https://github.com/WiktorB2004/llama-index-jev#readme).

```bash
pip install llama-index-postprocessor-jev
export TYPESAFE_API_KEY=...   # or pass api_key= to JevRerank
# OpenRouter (optional): export OPENROUTER_API_KEY=... and
# JevRerank(provider="openrouter")
# Vercel (optional): export AI_GATEWAY_API_KEY=... and
# JevRerank(provider="vercel")
# Cloudflare (optional): export CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID
# JevRerank(provider="cloudflare")
```

## Usage

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.postprocessor.jev import JevRerank

index = VectorStoreIndex.from_documents(
    [
        Document(text="The Eiffel Tower is in Paris."),
        Document(text="Python is a programming language."),
        Document(text="The Louvre is a museum in Paris."),
    ]
)

reranker = JevRerank(top_n=2, mode="score")
# reranker = JevRerank(provider="openrouter", top_n=2, mode="score")
query_engine = index.as_query_engine(
    similarity_top_k=5,
    node_postprocessors=[reranker],
)
print(query_engine.query("Where is the Eiffel Tower?"))
```

A standalone walkthrough is in
[`examples/basic_rerank.py`](https://github.com/WiktorB2004/llama-index-jev/blob/main/examples/basic_rerank.py)
([examples README](https://github.com/WiktorB2004/llama-index-jev/blob/main/examples/README.md)).
The checked-in walkthrough uses `provider="openrouter"` and needs `OPENROUTER_API_KEY`.

## Design

### One call per passage

Each retrieved node becomes its own `system_one` call:

```python
state = {"query": query_str, "passage": passage_text}
questions = {"relevance": Score(...)}  # or Noul(...)
```

Jev's question ids are **invisible** to the model. If you stuff 50 passages
into one `state` and ask 50 questions, the model cannot tell which question
goes with which passage, and accuracy falls as the context grows ("context
rot"). TypeSafe's own rerank cookbook scores one query–passage pair at a
time. Parallelism is bounded with `max_concurrency` (default 8), not by
chunking nodes into groups of 255.

### Fail open

If any Jev call in a pass fails (timeout, 429, 5xx), the whole pass is
abandoned and the original retrieval order is returned, truncated to
`top_n`. A slightly-worse ranking is better than dropping the user's
context. Set `raise_on_error=True` to surface the error instead. The pass
is all-or-nothing: we never mix a partial Jev ranking with original
retrieval scores.

### `confidence_threshold` flags, it does not drop

In `mode="score"`, Jev also returns `confidence`. If you set
`confidence_threshold`, nodes below it get
`metadata["jev_low_confidence"] = True`. They still participate in ranking.
Filtering on confidence would silently delete evidence; flagging lets the
caller decide.

Noul mode has no confidence field. `jev_confidence` /
`jev_low_confidence` are not written.

### Score scale is 0–3, not cosine similarity

`mode="score"` (the default) uses a 4-level rubric:

| Score | Meaning |
| --- | --- |
| 0 | Off-topic |
| 1 | Tangential |
| 2 | Partial answer |
| 3 | Fully answers |

The returned `.score` can land between levels (e.g. `2.4`). LlamaIndex only
**sorts** by this number. Do not feed it to `SimilarityPostprocessor` or
any cutoff that expects a 0–1 cosine.

`mode="noul"` is a yes/no relevance probability in 0–1.

Original retrieval scores are stored on the node as
`metadata["retrieval_score"]`.
