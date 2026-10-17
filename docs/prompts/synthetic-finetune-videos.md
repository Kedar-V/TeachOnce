# Prompt recipe — Stage-B finetune videos (3 procedures)

**Run on a separate machine** (or agent-drive Northstar while screen-recording).  
**Model:** `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`  
**Console:** [`../../northstar-console/`](../../northstar-console/) · **Agent protocol:** [`../../northstar-console/AGENT.md`](../../northstar-console/AGENT.md)  
**Adapter:** [`../../scripts/tool_registry_adapter.py`](../../scripts/tool_registry_adapter.py)

---

## Record on Northstar (deterministic)

```bash
cd northstar-console && python3 -m http.server 8765
```

| Procedure | Path | URL |
|-----------|------|-----|
| Warranty | active | `http://127.0.0.1:8765/?procedure=warranty&path=active&hud=0` |
| Warranty | expired | `...?procedure=warranty&path=expired&hud=0` |
| Parts | in_stock | `...?procedure=parts&path=in_stock&hud=0` |
| Parts | backorder | `...?procedure=parts&path=backorder&hud=0` |
| Escalate | p1 | `...?procedure=escalate&path=p1&hud=0` |
| Escalate | standard | `...?procedure=escalate&path=standard&hud=0` |

Controlling agent: `Northstar.observe()` → click/type `data-testid` → until `flags.done`.  
Optional canned cursor: `&autoplay=1` (prefer agent-driven).

---

## One-shot rule

No regen later. Shoot **55–60**, keep **50**. Val + one demo per family **first**.

**Minimum if time dies:** ≥8 warranty + ≥6 parts + ≥6 escalate (all with valid meta), `manifest.incomplete=true`.

---

## Quota (exactly 50 keepers)

| Procedure | `family` | Paths | Count | video_id prefix |
|-----------|----------|-------|-------|-----------------|
| **Warranty** | `warranty_resolution` | 14× `active` + 8× `expired` | **22** | `warranty_01`…`22` |
| **Parts** | `parts_fulfillment` | 9× `in_stock` + 7× `backorder` | **16** | `parts_01`…`16` |
| **Escalate** | `severity_escalation` | 7× `p1` + 5× `standard` | **12** | `escalate_01`…`12` |
| **Total** | | | **50** | |

**Val (shoot first):**

| video_id | procedure | path | split |
|----------|-----------|------|-------|
| `warranty_21` | warranty | active | val |
| `warranty_22` | warranty | expired | val |
| `parts_15` | parts | in_stock | val |
| `parts_16` | parts | backorder | val |
| `escalate_11` | escalate | p1 | val |
| `escalate_12` | escalate | standard | val |

**Demo polish (shoot second):** `warranty_01` (active), `parts_01` (in_stock), `escalate_01` (p1).

---

## Shared rules (all procedures)

### Tool registry — allowlist only

```text
# shared
LOOKUP_CUSTOMER, GET_TICKET, UPDATE_TICKET, GET_DEVICE

# warranty
CHECK_WARRANTY, GET_DIAGNOSTICS, CREATE_REPAIR, RECORD_PAID_OPTIONS

# parts
CHECK_PARTS_STOCK, CREATE_PARTS_SHIPMENT, CREATE_BACKORDER

# escalate
ASSESS_SEVERITY, PAGE_ONCALL, ASSIGN_L2_QUEUE
```

Never invent op names. Never put snake_case ops as on-screen hero text — **meta only**.

### Realism

Drive **Northstar Support Console** (real tabs/forms/tables/cursor/typing).  
Reject toy `Step: OP_NAME` flashcards. Target ≥2 Mbps H.264, 1280×720 or 1920×1080, 35–75s, hold ≥1.5s after each action.

### IDs

| Kind | Prefix | Examples |
|------|--------|----------|
| Customer | `C-` | C-1042, C-2109, C-0881, C-5010 |
| Device | `D-` | D-2021 |
| Ticket | `CS-` | CS-48213 |
| Warranty | `W-` | W-9001 |
| Repair | `SR-` | SR-88412 |
| Shipment | `SH-` | SH-22011 |
| Backorder | `BO-` | BO-9014 |

Use console defaults per path (see AGENT.md / `js/data.js`).

### Output layout

```text
data/generated/demonstrations/videos/
  warranty_01/{video.mp4, meta.json}
  ...
  parts_01/...
  escalate_01/...
  manifest.json
```

### meta.json (all procedures)

```json
{
  "video_id": "parts_03",
  "source_dataset": "teachonce_self_synthetic",
  "generation_method": "agent_driven_northstar_console",
  "procedure": "parts",
  "family": "parts_fulfillment",
  "split": "train",
  "path_type": "in_stock",
  "duration_sec": 48,
  "fps": 30,
  "resolution": "1280x720",
  "goal": "Fulfill spare part for failed device when in stock",
  "sop": null,
  "case": { "customer_id": "C-1042", "device_id": "D-2021", "ticket_id": "CS-48213" },
  "ops": ["LOOKUP_CUSTOMER", "GET_TICKET", "GET_DEVICE", "CHECK_PARTS_STOCK", "CREATE_PARTS_SHIPMENT", "UPDATE_TICKET"],
  "events": [
    { "t_start": 0.0, "t_end": 6.0, "op": "LOOKUP_CUSTOMER", "label": "Search customer", "observed": {} }
  ],
  "tools_ref": "mvp_registry_v1",
  "console_url": "?procedure=parts&path=in_stock&hud=0",
  "prompt_variant": "light",
  "notes": ""
}
```

**Required fields:** `procedure`, `family`, `path_type`, `ops`, `events`, `case`, `split`, `video_id`.  
`path_type` must match console `path` (`active`|`expired`|`in_stock`|`backorder`|`p1`|`standard`).

---

# Procedure 1 — Warranty resolution

**family:** `warranty_resolution`  
**Nav:** Customers · Tickets · Devices · Warranty · Diagnostics · Repair

## Master prompt (warranty)

```text
You are recording Stage-B TeachOnce training video on Northstar Support Console
(?procedure=warranty&path={active|expired}&hud=0).

GOAL: Expert support agent walks the warranty resolution procedure.
Model target: video → tool-registry DAG (Nemotron Nano 12B VL).

PATH active:
  LOOKUP_CUSTOMER → GET_TICKET → GET_DEVICE → CHECK_WARRANTY (show ACTIVE chip)
  → GET_DIAGNOSTICS → CREATE_REPAIR → UPDATE_TICKET
  meta.path_type=active; must include CREATE_REPAIR; must NOT include RECORD_PAID_OPTIONS

PATH expired:
  … → CHECK_WARRANTY (show EXPIRED chip + past end date)
  → GET_DIAGNOSTICS → RECORD_PAID_OPTIONS → UPDATE_TICKET
  meta.path_type=expired; must include RECORD_PAID_OPTIONS; must NOT include CREATE_REPAIR

UI: realistic SaaS — search/type customer, open ticket row, select device, warranty
panel chip, diagnostics table, repair form OR paid-options panel, save ticket + toast.
NO Step:OP_NAME burn-in. Registry ops only in meta.json.

Write meta.json with family=warranty_resolution, procedure=warranty.
```

## Per-op (warranty)

| Op | On screen |
|----|-----------|
| LOOKUP_CUSTOMER | Type `C-####`, Search, results |
| GET_TICKET | Open ticket from queue |
| GET_DEVICE | Select device row |
| CHECK_WARRANTY | ACTIVE/EXPIRED chip in warranty panel |
| GET_DIAGNOSTICS | Results table / fault code |
| CREATE_REPAIR | Repair form → SR-##### |
| RECORD_PAID_OPTIONS | Paid options → confirm |
| UPDATE_TICKET | Status + note → Saved |

## Shot list — warranty (22)

| video_id | path | split | Variant |
|----------|------|-------|---------|
| warranty_01 | active | train | **Demo** clean |
| warranty_02–13 | active | train | pace/theme/IDs |
| warranty_14 | active | train | captions off, slower |
| warranty_15–20 | expired | train | pace/theme/IDs |
| warranty_21 | active | **val** | clean |
| warranty_22 | expired | **val** | clean |

---

# Procedure 2 — Spare parts fulfillment

**family:** `parts_fulfillment`  
**Nav:** Customers · Tickets · Devices · Inventory · Shipping

## Master prompt (parts)

```text
You are recording Stage-B TeachOnce training video on Northstar Support Console
(?procedure=parts&path={in_stock|backorder}&hud=0).

GOAL: Expert fulfills a spare part for a failed device. Different DAG from warranty
(inventory decision, not warranty decision).

PATH in_stock:
  LOOKUP_CUSTOMER → GET_TICKET → GET_DEVICE → CHECK_PARTS_STOCK (show IN_STOCK)
  → CREATE_PARTS_SHIPMENT → UPDATE_TICKET
  meta.path_type=in_stock; must include CREATE_PARTS_SHIPMENT; must NOT include CREATE_BACKORDER

PATH backorder:
  … → CHECK_PARTS_STOCK (show OUT_OF_STOCK)
  → CREATE_BACKORDER → UPDATE_TICKET
  meta.path_type=backorder; must include CREATE_BACKORDER; must NOT include CREATE_PARTS_SHIPMENT

UI: after device select, Inventory stock panel with clear IN_STOCK or OUT_OF_STOCK chip;
Shipping form for shipment SH-##### or backorder BO-#####; then update ticket.
Do NOT open Warranty/Diagnostics/Repair tabs for this procedure.
family=parts_fulfillment; procedure=parts.
```

## Per-op (parts)

| Op | On screen |
|----|-----------|
| LOOKUP_CUSTOMER / GET_TICKET / GET_DEVICE | Same CRM pattern as warranty |
| CHECK_PARTS_STOCK | Inventory: SKU + IN_STOCK / OUT_OF_STOCK chip |
| CREATE_PARTS_SHIPMENT | Shipping form → SH-##### |
| CREATE_BACKORDER | Backorder form → BO-##### |
| UPDATE_TICKET | Status e.g. “Parts shipped” / “Parts backordered” |

## Shot list — parts (16)

| video_id | path | split | Variant |
|----------|------|-------|---------|
| parts_01 | in_stock | train | **Demo** clean |
| parts_02–09 | in_stock | train | pace/IDs |
| parts_10–14 | backorder | train | pace/IDs |
| parts_15 | in_stock | **val** | clean |
| parts_16 | backorder | **val** | clean |

---

# Procedure 3 — Severity escalation

**family:** `severity_escalation`  
**Nav:** Tickets · Customers · Escalation

## Master prompt (escalate)

```text
You are recording Stage-B TeachOnce training video on Northstar Support Console
(?procedure=escalate&path={p1|standard}&hud=0).

GOAL: Agent triages ticket severity and routes — third DAG family (no warranty/parts).

PATH p1 (default case C-5010):
  GET_TICKET → LOOKUP_CUSTOMER → ASSESS_SEVERITY (show P1)
  → PAGE_ONCALL → UPDATE_TICKET
  meta.path_type=p1; must include PAGE_ONCALL; must NOT include ASSIGN_L2_QUEUE

PATH standard (default case C-0881):
  GET_TICKET → LOOKUP_CUSTOMER → ASSESS_SEVERITY (show P2/P3)
  → ASSIGN_L2_QUEUE → UPDATE_TICKET
  meta.path_type=standard; must include ASSIGN_L2_QUEUE; must NOT include PAGE_ONCALL

UI: start on Tickets queue; open case; look up customer; Escalation desk assess;
then either "Page on-call" or "Assign L2 queue"; update ticket status.
Do NOT use Warranty/Inventory/Repair flows.
family=severity_escalation; procedure=escalate.
```

## Per-op (escalate)

| Op | On screen |
|----|-----------|
| GET_TICKET | Open ticket first (queue) |
| LOOKUP_CUSTOMER | Customers search |
| ASSESS_SEVERITY | Escalation desk → P1 or P2/P3 badge |
| PAGE_ONCALL | Page on-call button / confirmation |
| ASSIGN_L2_QUEUE | Assign L2 queue button / confirmation |
| UPDATE_TICKET | “Escalated to on-call” / “Assigned L2” |

## Shot list — escalate (12)

| video_id | path | split | Variant |
|----------|------|-------|---------|
| escalate_01 | p1 | train | **Demo** clean |
| escalate_02–07 | p1 | train | pace/IDs |
| escalate_08–10 | standard | train | pace/IDs |
| escalate_11 | p1 | **val** | clean |
| escalate_12 | standard | **val** | clean |

---

## Per-video operator template

```text
Record 1280x720 screen of Northstar Support Console.
URL: {CONSOLE_URL}
video_id: {VIDEO_ID}
procedure: {PROCEDURE}
path_type: {PATH}
family: {FAMILY}
split: {SPLIT}

Follow playbook ops in order (human labels on screen; registry ops in meta only):
{OPS_LIST}

Settle ≥1.5s after each action. Visible cursor + typing where relevant.
Write meta.json (schema above) in the same folder as video.mp4 before next take.
```

---

## Agent one-liner per procedure

```js
// Warranty active
Northstar.act("set_procedure", { procedure: "warranty", path: "active" });
for (const op of Northstar.playbook().steps.map(s => s.op)) {
  Northstar.act("playbook_step", { op }); /* wait 1.5s+ between for recording */
}

// Parts backorder
Northstar.act("set_procedure", { procedure: "parts", path: "backorder" });

// Escalate P1
Northstar.act("set_procedure", { procedure: "escalate", path: "p1" });
```

Or click through using `Northstar.observe().availableActions`.

---

## QA gate (before leaving machine)

- [ ] 22 warranty + 16 parts + 12 escalate folders (or documented incomplete minimum)
- [ ] Val six exist and play
- [ ] Demo three exist (`warranty_01`, `parts_01`, `escalate_01`)
- [ ] Every folder has `video.mp4` + `meta.json`
- [ ] `meta.procedure` / `family` / `path_type` / `ops` consistent (no cross-family ops)
- [ ] Warranty expired ∌ CREATE_REPAIR; parts backorder ∌ CREATE_PARTS_SHIPMENT; escalate standard ∌ PAGE_ONCALL
- [ ] No toy Step:OP UI; sniff-test “real console”
- [ ] Adapter: `adapt_record(..., [{"op": o} for o in meta["ops"]])` → `ok` for each
- [ ] Zip tree + `manifest.json`

---

## Manifest stub

```json
{
  "version": "2",
  "stage": "B",
  "count": 50,
  "incomplete": false,
  "procedures": {
    "warranty": { "family": "warranty_resolution", "active": 14, "expired": 8 },
    "parts": { "family": "parts_fulfillment", "in_stock": 9, "backorder": 7 },
    "escalate": { "family": "severity_escalation", "p1": 7, "standard": 5 }
  },
  "val": ["warranty_21", "warranty_22", "parts_15", "parts_16", "escalate_11", "escalate_12"],
  "demo": ["warranty_01", "parts_01", "escalate_01"],
  "model_target": "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",
  "console": "northstar-console",
  "adapter": "scripts/tool_registry_adapter.py",
  "one_shot": true
}
```

---

## What NOT to generate

- Mixing procedures in one video  
- Warranty ops on a parts URL (or vice versa)  
- Toy step-trainer UIs / cloning old `gen_one_synthetic_sample` flashcards  
- Relabeling VideoCUA as `teachonce_self_synthetic`  
- Only one family (goal is **several different DAGs** for the demo)
