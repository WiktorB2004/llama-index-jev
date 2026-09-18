# Changelog

## Unreleased

- GitHub community files: contributing guide, code of conduct, security policy, issue and PR templates

## 0.1.1

Packaging and PyPI metadata only — no API changes.

- Keywords, classifiers, and project URLs on both packages
- Hatch wheel `packages` so installs ship a wheel, not sdist-only
- Product README (install, snippets, nfcorpus / SciFact numbers) on GitHub and PyPI
- `contains_example = true` for LlamaHub metadata

## 0.1.0

Initial release of the two independently installable LlamaIndex integrations:

- `llama-index-postprocessor-jev` — `JevRerank` (`BaseNodePostprocessor`)
- `llama-index-selectors-jev` — `JevSingleSelector`, `JevMultiSelector` (`BaseSelector`)

TypeSafe and OpenRouter providers, score/noul rerank modes, fail-open rerank and fail-closed select.
