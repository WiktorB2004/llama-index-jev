# Development

```bash
uv sync
```

## Tests

Tests mock `TypeSafeClient.system_one` / `AsyncTypeSafeClient.system_one`. No live API key is required.

```bash
# Run each package separately so test module names do not collide.
uv run pytest --rootdir=packages/llama-index-postprocessor-jev \
    packages/llama-index-postprocessor-jev
uv run pytest --rootdir=packages/llama-index-selectors-jev \
    packages/llama-index-selectors-jev

uv run mypy
```

## Docs

This site is MkDocs Material. API pages are generated from docstrings with mkdocstrings.

```bash
uv sync --group docs
uv run mkdocs serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). `mkdocs build --strict` is what CI and GitHub Pages run.

Narrative pages live in `docs/`. Constructor and field docs live on the public classes in the two packages — edit those when the API changes, not the generated HTML.

GitHub Pages deploys from `.github/workflows/docs.yml` on every push to `main`. In the repo settings, set **Pages → Source** to **GitHub Actions** (one-time). The site is `https://wiktorb2004.github.io/llama-index-jev/`.

## Benchmark

Retrieval eval lives in `benchmark/`. Needs `OPENROUTER_API_KEY` (or `TYPESAFE_API_KEY`) and `uv sync --group benchmark`. See [benchmark](benchmark.md).

## License

MIT. See [LICENSE](https://github.com/WiktorB2004/llama-index-jev/blob/main/LICENSE).
