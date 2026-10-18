#!/usr/bin/env python3
"""Pre-SFT baseline: Nemotron VL via vLLM + guided_json → dag_reward metrics.

Same model as SFT: nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16.
Input = real video.mp4 via video_url (file://), not text-only / not JPEG stubs.
Structured output: guided_json (vLLM 0.8.x / Red Hat OpenAI-compatible API).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.nemo_rl_gt import dag_reward  # noqa: E402
from scripts.tool_registry_adapter import MVP_OPS  # noqa: E402


def _action_variants() -> list[dict]:
    """One guided choice per MVP registry tool — model picks (op, tool) as a locked pair."""
    out = []
    for op, meta in sorted(MVP_OPS.items()):
        out.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "type", "op", "tool"],
                "properties": {
                    "id": {"type": "string"},
                    "type": {"const": "ACTION"},
                    "op": {"const": op},
                    "tool": {"const": meta["service"]},
                    "label": {"type": "string"},
                    "inputs": {"type": "object"},
                },
            }
        )
    return out


def guided_schema() -> dict:
    """JSON schema: ACTION nodes must be one of the predefined MVP_OPS (op+tool locked)."""
    start = {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "type"],
        "properties": {"id": {"type": "string"}, "type": {"const": "START"}, "label": {"type": "string"}},
    }
    end = {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "type"],
        "properties": {
            "id": {"type": "string"},
            "type": {"const": "END"},
            "outcome": {"type": "string"},
            "label": {"type": "string"},
        },
    }
    decision = {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "type"],
        "properties": {
            "id": {"type": "string"},
            "type": {"const": "DECISION"},
            "label": {"type": "string"},
            "branches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {"name": {"type": "string"}},
                },
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["id", "name", "version", "nodes", "edges"],
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string"},
            "version": {"type": "string"},
            "nodes": {
                "type": "array",
                "minItems": 2,
                "items": {"oneOf": [start, decision, end, *_action_variants()]},
            },
            "edges": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["from", "to"],
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "when": {"type": ["string", "null"]},
                    },
                },
            },
        },
    }


def resolve_video(gt: dict, video_root: Path) -> Path:
    raw = Path(gt.get("video_path") or "")
    if raw.exists():
        return raw
    vid = gt.get("video_id") or raw.parent.name
    alt = video_root / vid / "video.mp4"
    if alt.exists():
        return alt
    raise FileNotFoundError(f"video missing: {gt.get('id')}")


def call_vllm(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    video_path: Path,
    schema: dict,
    max_tokens: int,
    timeout: float,
) -> tuple[str, float]:
    # Real mp4 — vLLM loads via --allowed-local-media-path
    video_url = video_path.resolve().as_uri()  # file:///...
    content: list[dict] = [
        {"type": "text", "text": prompt},
        {"type": "video_url", "video_url": {"url": video_url}},
    ]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"{base_url.rstrip('/')}/chat/completions"
    # vLLM 0.8.x: guided_json; 0.11+ still accepts it as extra; some builds want structured_outputs
    payloads = [
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "guided_json": schema,
        },
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "structured_outputs": {"json": schema},
        },
    ]
    t0 = time.perf_counter()
    last_err: Exception | None = None
    data = None
    for payload in payloads:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if r.status_code < 400:
            data = r.json()
            break
        last_err = RuntimeError(f"{r.status_code}: {r.text[:400]}")
    lat = time.perf_counter() - t0
    if data is None:
        raise last_err or RuntimeError("chat.completions failed")
    text = data["choices"][0]["message"]["content"] or ""
    return text, lat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=str(ROOT / "data/generated/nemo_rl/gt.jsonl"))
    ap.add_argument("--video-root", default=str(ROOT / "data/generated/demonstrations/videos"))
    ap.add_argument("--base-url", default=os.environ.get("NVIDIA_VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("LOCAL_VLLM_API_KEY", "teachonce-local-key"))
    ap.add_argument("--model", default=os.environ.get("MODEL_ID", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"))
    ap.add_argument("--max-samples", type=int, default=25)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", default=str(ROOT / "results/baseline_vllm_dag.json"))
    args = ap.parse_args()
    if "Nemotron" not in args.model and "nemotron" not in args.model.lower():
        raise SystemExit(f"baseline must use Nemotron VL, got {args.model!r}")

    rows = [json.loads(l) for l in Path(args.gt).read_text().splitlines() if l.strip()]
    rows = rows[: args.max_samples]
    video_root = Path(args.video_root)
    schema = guided_schema()

    # wait for /v1/models
    for _ in range(90):
        try:
            h = requests.get(
                f"{args.base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {args.api_key}"},
                timeout=5,
            )
            if h.status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(5)
    else:
        raise SystemExit(f"vLLM not ready at {args.base_url}")

    per: list[dict] = []
    sums = {"reward": 0.0, "schema_ok": 0.0, "critical_ops": 0.0, "taken_branch": 0.0, "latency_s": 0.0}
    n_ok = 0
    for gt in tqdm(rows, desc="baseline"):
        try:
            path = resolve_video(gt, video_root)
            if path.suffix.lower() not in (".mp4", ".webm", ".mkv", ".mov"):
                raise RuntimeError(f"not a video file: {path}")
            text, lat = call_vllm(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                prompt=gt.get("prompt")
                or (
                    "Emit executable DAG JSON. For each ACTION node pick one tool from the "
                    "predefined registry (op+tool locked pairs in the schema)."
                ),
                video_path=path,
                schema=schema,
                max_tokens=args.max_tokens,
                timeout=300,
            )
            scores = dag_reward(text, gt)
            scores["id"] = gt["id"]
            scores["video"] = str(path)
            scores["latency_s"] = round(lat, 3)
            scores["raw_preview"] = text[:240]
            n_ok += 1
        except Exception as e:
            scores = {
                "id": gt.get("id"),
                "reward": 0.0,
                "schema_ok": 0.0,
                "critical_ops": 0.0,
                "taken_branch": 0.0,
                "latency_s": 0.0,
                "error": str(e)[:300],
            }
        per.append(scores)
        for k in ("reward", "schema_ok", "critical_ops", "taken_branch", "latency_s"):
            sums[k] += float(scores.get(k) or 0)

    n = max(len(per), 1)
    summary = {
        "n": len(per),
        "n_success": n_ok,
        "model": args.model,
        "input": "video_url/file://mp4",
        "guided_json": True,
        "registry_oneof": True,
        "mean_reward": round(sums["reward"] / n, 4),
        "mean_schema_ok": round(sums["schema_ok"] / n, 4),
        "mean_critical_ops": round(sums["critical_ops"] / n, 4),
        "mean_taken_branch": round(sums["taken_branch"] / n, 4),
        "mean_latency_s": round(sums["latency_s"] / n, 3),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_sample": per}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
