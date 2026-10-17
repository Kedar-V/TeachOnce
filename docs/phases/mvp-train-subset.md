# MVP train subset (hackathon)

**Schema (non-negotiable):** [`../dataset-ir.md`](../dataset-ir.md) · `data/schema/*.schema.json`

## Stage A — continuous video only

| Source | Media | Samples (local) | Path |
|--------|-------|----------------:|------|
| **OSTask-demo** | `.mkv` | 5 | `data/raw/ostask-demo/` |
| **VideoCUA** (Conky zip) | `.mp4` | 3 | `data/raw/videocua/Conky/` |

**Excluded:** AgentNet (PNG packs), VideoGUI on HF (screenshots; full video on Drive).

```bash
python3 scripts/download_video_sources.py   # OSTask + VideoCUA only
python3 scripts/ingest_stage_a.py           # OSTask → schema (quarantine if ops don’t map)
python3 scripts/validate_train_jsonl.py
python3 scripts/build_mvp_split.py          # Stage B warranty gold
```

Adapter fail-closed → `data/generated/quarantine/`. Do not invent registry ops.

## Stage B — product gold (still required)

Warranty self/synthetic videos → same schema. Min **20** / full **50** — [`../prompts/synthetic-finetune-videos.md`](../prompts/synthetic-finetune-videos.md).

| Split | Role |
|-------|------|
| `data/generated/splits/mvp_train.jsonl` | Stage B gold |
| `data/generated/splits/stage_a_train.jsonl` | Schema-valid Stage A only |
