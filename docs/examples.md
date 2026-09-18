# Examples

Live walkthroughs of the Jev reranker and selector against OpenRouter's Decisions API. Retrieval and routing quality is the point; embeddings and the unused router summarizer are local mocks so you do not also need an OpenAI key.

```bash
export OPENROUTER_API_KEY=...
uv run python examples/basic_rerank.py
uv run python examples/basic_selector.py
uv run python examples/router_query_engine.py
```

Outputs below were captured with `provider="openrouter"` and `timeout_s=30`. Scores and choices can move slightly between runs.

## `basic_rerank.py`

Five in-memory passages, mock embeddings (every cosine is `1.000`), then `JevRerank(top_n=3, mode="score")` on `Where is the Eiffel Tower?`.

Score mode is a 0–3 rubric (0 off-topic … 3 fully answers), not cosine similarity. Jev promotes the Eiffel Tower sentence to the top and drops the photosynthesis / Python noise.

```text
Before JevRerank:
  1.000  Paris is the capital of France.
  1.000  Photosynthesis converts light energy into chemical energy.
  1.000  The Louvre is the world's largest art museum, in Paris.
  1.000  The Eiffel Tower is a wrought-iron tower in Paris, France.
  1.000  Python is a high-level programming language.

After JevRerank:
  3.000  The Eiffel Tower is a wrought-iron tower in Paris, France.
  0.910  Paris is the capital of France.
  0.690  The Louvre is the world's largest art museum, in Paris.
```

Source: [`examples/basic_rerank.py`](https://github.com/WiktorB2004/llama-index-jev/blob/main/examples/basic_rerank.py).

## `basic_selector.py`

`JevSingleSelector` over two toy tools (`weather` vs `docs`) with no `RouterQueryEngine`. Query: `Will it rain in Paris tomorrow?`

```text
query:      Will it rain in Paris tomorrow?
index:      0
reason:     Jev selected 'weather' (confidence=1.00)
chose:      weather
```

Source: [`examples/basic_selector.py`](https://github.com/WiktorB2004/llama-index-jev/blob/main/examples/basic_selector.py).

## `router_query_engine.py`

Same two tools, wired through `RouterQueryEngine`. Jev picks a route, then the chosen engine returns a fixed string.

`RouterQueryEngine` always constructs an LLM for its multi-route summarizer even when the selector is single-choice. This file passes `MockLLM()` so that constructor does not try to import `llama-index-llms-openai`. `verbose=True` prints the selection before the answer.

```text
Selecting query engine 0: Jev selected 'weather' (confidence=1.00).
Forecast: light rain in Paris tomorrow.
```

Source: [`examples/router_query_engine.py`](https://github.com/WiktorB2004/llama-index-jev/blob/main/examples/router_query_engine.py).
