# Selectors

LlamaIndex `BaseSelector`s. Single-route uses one `Choice`; multi-route uses one `Noul` per option.

## JevSingleSelector

| Argument | Default | Notes |
| --- | --- | --- |
| `provider` | `"typesafe"` | `"typesafe"` or `"openrouter"` |
| `model` | `"jev-latest"` | Remapped to `~typesafe/...` on OpenRouter |
| `default_index` | `None` | Fallback index if Jev fails; otherwise raise |
| `confidence_threshold` | `None` | Low / missing confidence is failure |
| `timeout_s` | `2.5` | HTTP timeout |
| `api_key` | env var | Constructor-only; see [providers](../guides/providers.md) |

::: llama_index.selectors.jev.JevSingleSelector

## JevMultiSelector

| Argument | Default | Notes |
| --- | --- | --- |
| `provider` | `"typesafe"` | `"typesafe"` or `"openrouter"` |
| `model` | `"jev-latest"` | Remapped to `~typesafe/...` on OpenRouter |
| `threshold` | `0.5` | Keep options with `noul > threshold` |
| `default_index` | `None` | API-error fallback only, not empty-Noul |
| `timeout_s` | `2.5` | HTTP timeout |
| `api_key` | env var | Constructor-only; see [providers](../guides/providers.md) |

::: llama_index.selectors.jev.JevMultiSelector
