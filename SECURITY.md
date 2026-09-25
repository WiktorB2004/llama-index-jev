# Security Policy

## Supported versions

The `0.1.x` line on `main` is supported. Older tags are not.

## Reporting a vulnerability

Do **not** open a public GitHub issue for a security problem or a leaked API key.

Report privately with
[GitHub security advisories](https://github.com/WiktorB2004/llama-index-jev/security/advisories/new).

Include:

- Package (`llama-index-postprocessor-jev`, `llama-index-selectors-jev`, or the workspace)
- Version or commit
- Impact (what an attacker can do)
- A minimal reproduction if you have one

You should hear back within a few days. If the report is accepted, a fix will land on `main` and a patched release will follow when the packages need a version bump.

## Secrets

`.env` is gitignored. Never commit `TYPESAFE_API_KEY`, `OPENROUTER_API_KEY`, `AI_GATEWAY_API_KEY`, or other credentials. If a key was pushed, rotate it at the provider and report the leak privately as above.
