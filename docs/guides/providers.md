# Providers

Both packages talk to Jev through the same `system_one(state, questions, model)` shape. The `provider` argument picks the transport.

| `provider` | Client | Env var | Default |
| --- | --- | --- | --- |
| `"typesafe"` | TypeSafe System One SDK | `TYPESAFE_API_KEY` | yes |
| `"openrouter"` | OpenRouter Decisions API | `OPENROUTER_API_KEY` | |
| `"vercel"` | Vercel AI Gateway (TypeSafe API) | `AI_GATEWAY_API_KEY` | |

```python
JevRerank()                                 # TypeSafe
JevRerank(provider="openrouter")            # OpenRouter
JevRerank(provider="vercel")                # Vercel AI Gateway
JevSingleSelector(provider="openrouter", timeout_s=30)
```

Unknown `provider` values raise `ValueError` at construction.

## Model ids

The default model is `jev-latest`. On OpenRouter that id is remapped to `~typesafe/jev-latest`. Pass a slug that already contains `/` or starts with `~` to skip remapping (`~typesafe/jev-latest`, `typesafe/jev-latest`, …).

On Vercel, an id with no `/` is sent as `typesafe-ai/jev`. A slug that already contains `/` is sent unchanged.

## Timeouts

`timeout_s` (default **2.5**) is forwarded to the HTTP client. The [examples](../examples.md) and longer [benchmark](../benchmark.md) runs use `timeout_s=30` because a 2.5s budget is tight for live OpenRouter calls.

## Vercel AI Gateway

`provider="vercel"` uses the TypeSafe SDK against `https://ai-gateway.vercel.sh/typesafe` with `AI_GATEWAY_API_KEY`. Question types stay `noul`, `choice`, and `score`. Vercel's AI SDK `evaluate` path names the yes/no type `boolean`; this provider does not use that path.

## OpenRouter vs TypeSafe

OpenRouter is **not** TypeSafe `/v1/systemone`. It posts to `https://openrouter.ai/api/alpha/decisions` and wraps the JSON so callers still read `.answers[key]`. Cost on the documented nfcorpus protocol is OpenRouter `usage.cost`, not an estimate.

The examples and benchmark in this repo use `provider="openrouter"` so a single `OPENROUTER_API_KEY` is enough.
