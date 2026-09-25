# Providers

Both packages talk to Jev through the same `system_one(state, questions, model)` shape. The `provider` argument picks the transport.

| `provider` | Client | Env var | Default |
| --- | --- | --- | --- |
| `"typesafe"` | TypeSafe System One SDK | `TYPESAFE_API_KEY` | yes |
| `"openrouter"` | OpenRouter Decisions API | `OPENROUTER_API_KEY` | |
| `"vercel"` | Vercel AI Gateway (TypeSafe API) | `AI_GATEWAY_API_KEY` | |
| `"cloudflare"` | Cloudflare Workers AI | `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` | |

```python
JevRerank()                                 # TypeSafe
JevRerank(provider="openrouter")            # OpenRouter
JevRerank(provider="vercel")                # Vercel AI Gateway
JevRerank(provider="cloudflare")            # Cloudflare Workers AI
JevSingleSelector(provider="openrouter", timeout_s=30)
```

Unknown `provider` values raise `ValueError` at construction.

## Model ids

The default model is `jev-latest`. On OpenRouter that id is remapped to `~typesafe/jev-latest`. Pass a slug that already contains `/` or starts with `~` to skip remapping (`~typesafe/jev-latest`, `typesafe/jev-latest`, …).

On Vercel, an id with no `/` is sent as `typesafe-ai/jev`. A slug that already contains `/` is sent unchanged.

On Cloudflare, `jev-latest` and `typesafe/jev` are sent as `typesafe/jev`. Any other id raises `ValueError`.

## Timeouts

`timeout_s` (default **2.5**) is forwarded to the HTTP client. The [examples](../examples.md) and longer [benchmark](../benchmark.md) runs use `timeout_s=30` because a 2.5s budget is tight for live OpenRouter calls.

## Vercel AI Gateway

`provider="vercel"` uses the TypeSafe SDK against `https://ai-gateway.vercel.sh/typesafe` with `AI_GATEWAY_API_KEY`. Question types stay `noul`, `choice`, and `score`. Vercel's AI SDK `evaluate` path names the yes/no type `boolean`; this provider does not use that path.

## Cloudflare Workers AI

`provider="cloudflare"` posts to `https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run`. The body is `{"model": "typesafe/jev", "input": {"state", "questions"}}`. The token is `CLOUDFLARE_API_TOKEN` (or `api_key=`). The account id is `CLOUDFLARE_ACCOUNT_ID`. Question types stay `noul`, `choice`, and `score`. A `result` envelope is unwrapped when Cloudflare sends one; otherwise `answers` is read from the top-level JSON.

This is Workers AI, the model host. Cloudflare AI Gateway, the proxy, does not serve Jev on its own.

## OpenRouter vs TypeSafe

OpenRouter is **not** TypeSafe `/v1/systemone`. It posts to `https://openrouter.ai/api/alpha/decisions` and wraps the JSON so callers still read `.answers[key]`. Cost on the documented nfcorpus protocol is OpenRouter `usage.cost`, not an estimate.

The examples and benchmark in this repo use `provider="openrouter"` so a single `OPENROUTER_API_KEY` is enough.
