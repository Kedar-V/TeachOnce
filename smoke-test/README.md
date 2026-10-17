# Phase 0 smoke tests

Integration checks against **local vLLM** (Brev via tunnel). NVIDIA-only — no OpenAI.

## Before the notebook

```bash
# terminal 1 — keep running
brev port-forward teachonce-gpu -p 8000:8000

# terminal 2
cd /path/to/NvidiaHack
cp -n .env.example .env   # if needed
jupyter notebook smoke-test/phase0_integration.ipynb
# or: jupyter lab smoke-test/phase0_integration.ipynb
```

## Kernel (one-time)

Already set up as **TeachOnce Phase0 Smoke** (`teachonce-smoke`):

```bash
# recreate if needed
uv venv --python 3.12 smoke-test/.venv
source smoke-test/.venv/bin/activate
uv pip install ipykernel jupyter
python -m ipykernel install --user --name=teachonce-smoke --display-name="TeachOnce Phase0 Smoke"
```

In Cursor/VS Code: select kernel → **TeachOnce Phase0 Smoke**.

Stdlib HTTP only — no AI SDKs.

## What it checks

1. `GET /v1/models` — model registered  
2. `POST /v1/chat/completions` — one completion  
3. Env policy — base URL is loopback / self-hosted (not `api.openai.com`)
