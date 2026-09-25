# Contributing

Thanks for looking at the repo. This is an independent community project — **not** affiliated with TypeSafe or LlamaIndex.

Please follow the [Code of Conduct](https://github.com/WiktorB2004/llama-index-jev/blob/main/CODE_OF_CONDUCT.md).

## Scope

In scope:

- `llama-index-postprocessor-jev` (`JevRerank`)
- `llama-index-selectors-jev` (`JevSingleSelector`, `JevMultiSelector`)
- Docs, examples, and the retrieval benchmark harness

Out of scope (file those upstream):

- TypeSafe / Jev API behaviour
- LlamaIndex core
- OpenRouter

## Issues

Use the GitHub issue templates. Search existing issues first.

Do **not** open a public issue for a vulnerability or a leaked API key. See [SECURITY.md](https://github.com/WiktorB2004/llama-index-jev/blob/main/SECURITY.md).

## Setup

Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pre-commit install
```

`pre-commit` runs Ruff and mypy on commit. CI installs from `uv.lock` (`uv sync --frozen`) and runs the same checks without the hook.

## Dependencies

[Renovate](https://docs.renovatebot.com/) opens weekly grouped PRs for Python packages, GitHub Actions, and pre-commit pins. Do not add Dependabot version updates — the two bots would conflict. Dependabot **alerts** (GitHub Security) are fine.

## Tests

Tests mock `TypeSafeClient.system_one` / `AsyncTypeSafeClient.system_one`. No live API key is required.

Run each package separately so test module names do not collide:

```bash
uv run pytest --rootdir=packages/llama-index-postprocessor-jev \
    packages/llama-index-postprocessor-jev
uv run pytest --rootdir=packages/llama-index-selectors-jev \
    packages/llama-index-selectors-jev
```

Benchmark unit tests (no live retrieval):

```bash
uv run pytest benchmark
```

## Lint and types

```bash
uv run ruff check packages examples benchmark
uv run ruff format --check packages examples benchmark
uv run mypy
```

## Docs

```bash
uv sync --group docs
uv run mkdocs serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). CI runs `mkdocs build --strict`.

Narrative pages live in `docs/`. Constructor and field docs live on the public classes — edit those when the API changes, not generated HTML.

## Changelog

User-facing changes get a bullet under `## Unreleased` in `CHANGELOG.md` (docs site includes that file). Skip packaging-only or internal refactors if they do not affect callers.

## Pull requests

1. One concern per PR.
2. Tests for behaviour changes. Keep the mock-client pattern; do not call TypeSafe, OpenRouter, Vercel, or Cloudflare from unit tests.
3. Docs and changelog when the public API or documented protocol changes.
4. Fill in the PR template.

Do not commit `.env`, `TYPESAFE_API_KEY`, `OPENROUTER_API_KEY`, `AI_GATEWAY_API_KEY`, `CLOUDFLARE_API_TOKEN`, or `CLOUDFLARE_ACCOUNT_ID`.
