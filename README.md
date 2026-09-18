# LlamaIndex + TypeSafe Jev

Community integrations that plug [TypeSafe AI](https://typesafe.ai)'s Jev
decision model into [LlamaIndex](https://www.llamaindex.ai) as:

1. a **reranker** (`llama-index-postprocessor-jev`) - score each retrieved
   passage independently, then keep the top `n`
2. a **selector** (`llama-index-selectors-jev`) - pick which query engine /
   tool should handle a query

This is an independent community project. It is **not** officially affiliated
with TypeSafe or LlamaIndex.

## Why two packages

LlamaIndex has two different extension points:

- `BaseNodePostprocessor` sits _after retrieval_. It reorders `NodeWithScore`
  objects.
- `BaseSelector` sits _before calling a tool_. It returns one or more
  `SingleSelection` indices.

Those are separately publishable integration families
(`llama-index-postprocessor-*` and `llama-index-selectors-*`), so this repo
ships two independently installable packages that happen to share a vendor.

## Quickstart - rerank

```bash
pip install llama-index-postprocessor-jev
export TYPESAFE_API_KEY=...
```

```python
from llama_index.postprocessor.jev import JevRerank

reranker = JevRerank(top_n=5, mode="score")
# OpenRouter: JevRerank(provider="openrouter", top_n=5, mode="score")
query_engine = index.as_query_engine(node_postprocessors=[reranker])
```

See [`packages/llama-index-postprocessor-jev/README.md`](packages/llama-index-postprocessor-jev/README.md)
and [`examples/basic_rerank.py`](examples/basic_rerank.py).

## Quickstart - select

```bash
pip install llama-index-selectors-jev
export TYPESAFE_API_KEY=...
```

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.selectors.jev import JevSingleSelector

engine = RouterQueryEngine(
    selector=JevSingleSelector(),
    query_engine_tools=[weather_tool, docs_tool],
)
```

See [`packages/llama-index-selectors-jev/README.md`](packages/llama-index-selectors-jev/README.md),
[`examples/basic_selector.py`](examples/basic_selector.py), and
[`examples/router_query_engine.py`](examples/router_query_engine.py).

## Design in one paragraph

Jev is a System One model: you send a small `state` and a map of typed
questions (`Noul` / `Choice` / `Score`) and get typed answers back. Rerank
asks one relevance question **per passage** (state is `{query, passage}`
only) because stuffing many passages into one prompt causes context rot and
Jev cannot see question ids. Select may batch: a router only has the query
as state, so one `Choice` (single) or one `Noul` per option (multi) in a
single request is the right shape. Rerank **fails open** (keep original
retrieval order) because a slightly-worse ranking is better than no context.
Select **fails closed** (raise or a declared `default_index`) because the
wrong tool is worse than an error.

## Packages

| Package                         | Class                                   | LlamaIndex hook         |
| ------------------------------- | --------------------------------------- | ----------------------- |
| `llama-index-postprocessor-jev` | `JevRerank`                             | `BaseNodePostprocessor` |
| `llama-index-selectors-jev`     | `JevSingleSelector`, `JevMultiSelector` | `BaseSelector`          |

## Tests

Tests mock `TypeSafeClient.system_one` / `AsyncTypeSafeClient.system_one`.
No live API key is required.

```bash
uv sync

# Run each package separately so test module names do not collide.
uv run pytest --rootdir=packages/llama-index-postprocessor-jev \
    packages/llama-index-postprocessor-jev
uv run pytest --rootdir=packages/llama-index-selectors-jev \
    packages/llama-index-selectors-jev

uv run mypy
```

## Benchmark

A methodology stub lives in [`benchmark/`](benchmark/). Results writeup:
_coming soon_ (needs a live Jev key).

## License

MIT. See [LICENSE](LICENSE).
