# JevRerank

LlamaIndex `BaseNodePostprocessor`. Score each retrieved node with one Jev call, sort, keep `top_n`.

| Field | Default | Notes |
| --- | --- | --- |
| `provider` | `"typesafe"` | `"typesafe"`, `"openrouter"`, or `"vercel"` |
| `model` | `"jev-latest"` | OpenRouter: `~typesafe/...`. Vercel: bare ids become `typesafe-ai/jev` |
| `top_n` | `5` | Nodes returned after sorting |
| `mode` | `"score"` | `"score"` (0–3) or `"noul"` (0–1) |
| `confidence_threshold` | `None` | Score mode: flags `jev_low_confidence`, does not drop |
| `timeout_s` | `2.5` | HTTP timeout |
| `raise_on_error` | `False` | `False` = fail open |
| `max_concurrency` | `8` | In-flight `system_one` calls |
| `api_key` | env var | Constructor-only; see [providers](../guides/providers.md) |

::: llama_index.postprocessor.jev.JevRerank
