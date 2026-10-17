# TeachOnce — frozen train / DAG schema

**Source of truth:** [`lld.md`](lld.md) §3.3 + §4 + §5.2  
**Machine schemas:** [`../data/schema/train_row.schema.json`](../data/schema/train_row.schema.json) · [`../data/schema/dag.schema.json`](../data/schema/dag.schema.json)  
**Adapter:** [`../scripts/tool_registry_adapter.py`](../scripts/tool_registry_adapter.py)

## Non-negotiable

1. Every train/eval JSONL row **must** validate against `train_row.schema.json`.  
2. `target_dag` **must** validate against `dag.schema.json` **and** `validate_registry_dag`.  
3. `family` is always `warranty_resolution`.  
4. `ACTION.op` ∈ MVP 8 ops only — no click/`x`/`y`/`selector` nodes.  
5. Public corpora → same schema via adapter; `ok: false` → `data/generated/quarantine/`, **not** train.

## Train row (LLD §3.3)

```json
{
  "id": "train-warranty_01",
  "family": "warranty_resolution",
  "split": "train",
  "path_type": "active",
  "source_dataset": "teachonce_self_synthetic",
  "source_record_id": null,
  "input": {
    "goal": "...",
    "video_id": "warranty_01",
    "video_path": ".../video.mp4",
    "sop": "...",
    "policies": [],
    "demonstrations": [
      {
        "video_id": "warranty_01",
        "events": [
          {
            "t_start": 0.0,
            "t_end": 5.0,
            "op": "LOOKUP_CUSTOMER",
            "label": "Search customer",
            "observed": {}
          }
        ]
      }
    ],
    "tools_ref": "mvp_registry_v1"
  },
  "target_dag": { "id": "WF-...", "name": "...", "version": "1.0.0", "nodes": [], "edges": [] }
}
```

## MVP ops (LLD §5.2)

`LOOKUP_CUSTOMER` · `GET_TICKET` · `UPDATE_TICKET` · `GET_DEVICE` · `CHECK_WARRANTY` · `GET_DIAGNOSTICS` · `CREATE_REPAIR` · `RECORD_PAID_OPTIONS`

## One-shot Stage A download

See [`phases/mvp-train-subset.md`](phases/mvp-train-subset.md) — **OSTask-demo** (~66 MB, 5 tasks, &lt;10 min). Not full VideoCUA.
