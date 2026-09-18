# llama-index-selectors-jev

Drop-in `JevSingleSelector` / `JevMultiSelector` for [LlamaIndex](https://www.llamaindex.ai) routers: typed tool / query-engine choice with [TypeSafe Jev](https://typesafe.ai), without an LLM-as-judge.

Full story, reranker package, and benchmark numbers: [docs](https://wiktorb2004.github.io/llama-index-jev/) · [repo](https://github.com/WiktorB2004/llama-index-jev#readme).

```bash
pip install llama-index-selectors-jev
export TYPESAFE_API_KEY=...   # or pass api_key= to the selector
# OpenRouter (optional): export OPENROUTER_API_KEY=... and
# JevSingleSelector(provider="openrouter")
```

## Usage

```python
from llama_index.core.tools import ToolMetadata
from llama_index.selectors.jev import JevSingleSelector

selector = JevSingleSelector()
result = selector.select(
    [
        ToolMetadata(name="weather", description="Forecasts and current conditions"),
        ToolMetadata(name="docs", description="Product documentation lookup"),
    ],
    "Will it rain in Paris tomorrow?",
)
print(result.ind, result.reason)
```

`JevSingleSelector` plugs into `RouterQueryEngine`:

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.selectors.jev import JevSingleSelector

engine = RouterQueryEngine(
    selector=JevSingleSelector(),
    query_engine_tools=[weather_tool, docs_tool],
)
```

See [`examples/basic_selector.py`](https://github.com/WiktorB2004/llama-index-jev/blob/main/examples/basic_selector.py)
and [`examples/router_query_engine.py`](https://github.com/WiktorB2004/llama-index-jev/blob/main/examples/router_query_engine.py)
([examples README](https://github.com/WiktorB2004/llama-index-jev/blob/main/examples/README.md)).
The checked-in walkthroughs use `provider="openrouter"` and need `OPENROUTER_API_KEY`.

## Design

### Fail closed

Wrong routing is worse than slightly-worse ranking. If the API errors, if
Jev returns a choice that is not in the option set, or if confidence is
below `confidence_threshold`, `JevSingleSelector` **raises** — unless you
set `default_index`, in which case that one selection is returned with a
reason that names both the fallback and the error.

This is TypeSafe's "don't act when uncertain." A low-confidence but
in-schema choice is treated as failure, not as success.

### Single vs multi

- **`JevSingleSelector`** asks one `Choice` question whose option names
  are the tool names (when those names are unique). The model sees names
  and descriptions, not opaque `option_0` labels.
- **`JevMultiSelector`** asks one `Noul` per option in a **single**
  `system_one` call (the query is the whole state, so batching questions is
  correct). Options with `noul > threshold` are kept, in original order.

If there are more than 255 options, the multi-selector splits the question
map across calls of at most 255 keys. That is the only 255-chunking in this
repo; it is a safety cap on question-map size, not a rerank batch size.

### Multi-select never returns an empty list

If zero Nouls clear `threshold`, `JevMultiSelector` returns exactly the
**highest-noul** choice. An empty selection would leave a router with
nothing to call. `default_index` is **not** used for this case — it is
API-error fallback only.
