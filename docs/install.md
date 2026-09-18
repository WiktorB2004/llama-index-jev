# Install

Python **3.10+**. Two independently installable packages — pick the LlamaIndex hook you need.

=== "Rerank"

    ```bash
    pip install llama-index-postprocessor-jev
    ```

    ```python
    from llama_index.postprocessor.jev import JevRerank
    ```

=== "Select"

    ```bash
    pip install llama-index-selectors-jev
    ```

    ```python
    from llama_index.selectors.jev import JevSingleSelector, JevMultiSelector
    ```

=== "Both (uv workspace)"

    From a clone of this repo:

    ```bash
    uv sync
    ```

    The workspace root depends on both packages.

## API keys

Default provider is TypeSafe. Pass a key, or set the matching env var.

=== "TypeSafe"

    ```bash
    export TYPESAFE_API_KEY=...
    ```

    ```python
    JevRerank()                       # reads TYPESAFE_API_KEY
    JevRerank(api_key="...")          # explicit
    ```

=== "OpenRouter"

    ```bash
    export OPENROUTER_API_KEY=...
    ```

    ```python
    JevRerank(provider="openrouter")
    JevSingleSelector(provider="openrouter", api_key="...")
    ```

Missing keys raise `ValueError` at construction, not at query time. See [providers](guides/providers.md).

## LlamaIndex

Both packages require `llama-index-core>=0.13,<0.15` (pulled in as a dependency). You still need whatever you use for indexes, embeddings, and query engines — for example `llama-index` or `llama-index-embeddings-*`.

The [examples](examples.md) use mock embeddings / `MockLLM` so they only need an OpenRouter key.
