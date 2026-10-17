# TeachOnce — Low-Level Design (LLD)

**Status:** Implementation-level design  
**Companion docs:** [`plan.md`](plan.md), [`hld.md`](hld.md)  
**Audience:** Engineers implementing backend, ML, and frontend modules  

This document specifies schemas, API contracts, module boundaries, execution algorithms, and the primary sequence for the **warranty A–C MVP**. Values marked *(initial)* may be tuned during Milestone 0 without changing the overall design.

**MVP lens:** Implement only what Flows A–C need. Skip multi-family splits, Test Lab, Review Queue, and edge-case recovery.

---

## 0. Mandatory NVIDIA-only AI policy (enforced in code)

Full policy: [`plan.md`](plan.md) § Mandatory NVIDIA-only AI infrastructure policy.

**Rules for implementers:**

1. **Sole AI outbound module:** `backend/app/services/nvidia_inference.py`  
2. **Required env:** `NVIDIA_VLLM_BASE_URL` — if missing/empty → raise configuration error (HTTP 503); **never** fall back to a public LLM API  
3. **Allowlists:** `NVIDIA_ALLOWED_MODEL_IDS`, `NVIDIA_ALLOWED_INFERENCE_HOSTS` (or equivalent constants)  
4. **Protocol:** OpenAI-compatible client OK only with `base_url=NVIDIA_VLLM_BASE_URL` and disabled provider defaults  
5. **No** `openai.api_key` / Anthropic / Gemini / Groq / OpenRouter / Together / Fireworks / Cursor model SDKs as AI backends  
6. **Frontend:** zero direct AI provider calls — only FastAPI  
7. **Dataset / eval / augmentation:** deterministic templates preferred; any NL rewrite uses the same NVIDIA gateway on Brev  
8. **Cursor IDE** is not an application dependency  

---

## 1. Scope

| In scope | Out of scope |
|----------|--------------|
| DAG JSON schema and validator rules | Arbitrary code execution in nodes |
| FastAPI route contracts | Production auth / multi-tenancy |
| Tool adapter I/O | Real Salesforce / ServiceNow |
| Executor traversal algorithm | Parallel nodes, loops, rich retry engines |
| Runtime SQLite tables | Distributed orchestration |
| Generate → Brev Nemotron via NVIDIA gateway | Second generative model; any non-NVIDIA hosted LLM |
| Frontend page → API mapping (5 MVP screens) | Test Lab, Review Queue, multi-SOP UI; browser → LLM provider |
| Two warranty demo cases + one decision | Exotic predicates, loops, parallel nodes, rich retries |

---

## 2. Repository module map

```text
NvidiaHack/
  docs/                 plan.md, hld.md, lld.md, brev.md
  data/generated/       frozen canonical world
  scripts/
    generate_dataset.py
    validate_dataset.py
    reset_runtime.py
  ml/
    recipes/nemotron_nano_9b_teachonce_peft.yaml
    prepare_automodel_dataset.py
    merge_lora.py
    smoke_infer.py
    eval_compiler.py
  backend/
    app/
      main.py                 # FastAPI app factory + routers
      config.py               # NVIDIA_VLLM_BASE_URL, allowlists, DATA_DIR, DB_PATH
      schema/
        dag.py                # Pydantic DAG models
        procedure.py          # Procedure Package models
        api.py                # request/response DTOs
      dataset/
        loader.py             # read immutable generated files
      services/
        nvidia_inference.py   # SOLE AI outbound path (Brev Nemotron)
      generate/
        prompts.py            # system + user template
        service.py            # package → nvidia_inference → parse → validate → store
      validate/
        dag_validator.py
      tools/
        registry.py           # op → handler map
        crm.py
        device.py
        warranty.py
        diagnostics.py
        repair.py
      executor/
        engine.py             # traverse DAG
        bindings.py           # $case / $nodes resolution
        predicates.py         # eq/neq (MVP)
        verify.py             # postconditions
      db.py                   # SQLite schema + seed + reset
      routers/
        dataset.py
        workflows.py
        cases.py
        executions.py
        metrics.py
        admin.py
  frontend/
    src/
      api.ts                  # FastAPI only — no AI provider SDKs
      pages/
        Overview.tsx
        TeachStudio.tsx
        WorkflowEditor.tsx
        Executions.tsx
        Integrations.tsx
```

---

## 3. Canonical dataset contracts

### 3.1 Manifest

```json
{
  "version": "1.0.0",
  "seed": 42,
  "generated_at": "ISO-8601",
  "checksums": {
    "entities/customers.json": "sha256:...",
    "workflows/ground_truth.json": "sha256:..."
  }
}
```

### 3.2 Entity ID conventions

| Entity | Prefix | Example |
|--------|--------|---------|
| Customer | `C-` | `C-1042` |
| Device | `D-` | `D-2021` |
| Ticket / case | `CS-` | `CS-48213` |
| Warranty | `W-` | `W-9001` |
| Diagnostic | `DG-` | `DG-331` |
| Repair order | `SR-` | `SR-88412` (runtime-created) |
| Workflow | `WF-` | `WF-1042` |
| Execution | `EX-` | `EX-8849` |

### 3.3 Training JSONL record *(logical)*

```json
{
  "id": "train-warranty-001",
  "family": "warranty_resolution",
  "split": "train",
  "input": {
    "goal": "...",
    "sop": "...",
    "policies": ["..."],
    "demonstrations": [{ "events": [] }],
    "tools": [{ "name": "CHECK_WARRANTY", "input_schema": {}, "output_schema": {} }]
  },
  "target_dag": { "nodes": [], "edges": [] }
}
```

**MVP:** `family` is always `warranty_resolution`. Train/val are paraphrases of the same topology — not other process families. Do not build a multi-family split.

---

## 4. DAG schema (executable)

### 4.1 Top-level

```json
{
  "id": "WF-1042",
  "name": "Laptop Warranty & Service Resolution",
  "version": "1.0.0",
  "nodes": [],
  "edges": []
}
```

### 4.2 Node types

```text
START
  id, type="START"

ACTION
  id, type="ACTION"
  label
  tool          # registry service key, e.g. "warranty"
  op            # e.g. "CHECK_WARRANTY"
  inputs        # map: param → Binding
  # optional: description, evidence refs (UI)

DECISION
  id, type="DECISION"
  label
  branches[]    # MVP: exactly two — Active + Expired

END
  id, type="END"
  outcome       # e.g. "SUCCEEDED"
```

### 4.3 Binding

```text
Literal:   { "const": <json> }
Case ref:  { "ref": "$case.device_id" }
Node ref:  { "ref": "$nodes.<node_id>.<field>" }
```

### 4.4 Predicate *(MVP)*

```json
{
  "op": "eq",
  "left": { "ref": "$nodes.check_warranty.warranty_status" },
  "right": { "const": "ACTIVE" }
}
```

MVP ops: `eq`, `neq` (add `exists` only if ground truth needs it). No arbitrary expressions, Python, or JS.

### 4.5 Edge

```json
{
  "from": "check_warranty",
  "to": "decide_warranty",
  "when": null
}
```

For decision exits, `when` equals a branch `name` (MVP: `"Active"` / `"Expired"`).

### 4.6 Validator rules

1. Unique node IDs; valid types only.  
2. Exactly one `START`; at least one `END`.  
3. Every edge endpoint exists.  
4. Graph is a DAG (no cycles).  
5. Every node reachable from `START`; every path can reach an `END`.  
6. Each `DECISION` has outgoing edges covering every `branches[]` name (MVP: Active + Expired).  
7. Every `ACTION.op` exists in the tool registry; required inputs bound.  
8. `ref` bindings syntactically valid.  
9. Predicates use only allowed ops.  

Invalid → HTTP 422 with structured error list; do not persist as active. Do not over-invest in static path analysis beyond what the warranty DAG needs.

---

## 5. Tool registry and adapters

### 5.1 Registry entry

```json
{
  "id": "warranty",
  "name": "Warranty Service",
  "ops": {
    "CHECK_WARRANTY": {
      "inputs": { "device_id": "string" },
      "outputs": {
        "warranty_status": "ACTIVE|EXPIRED|UNKNOWN",
        "warranty_end": "date",
        "crm_warranty_flag": "ACTIVE|EXPIRED|UNKNOWN"
      }
    }
  }
}
```

### 5.2 MVP operations

| Op | Service | Side effects |
|----|---------|--------------|
| `LOOKUP_CUSTOMER` | CRM | Read |
| `GET_TICKET` | CRM | Read |
| `UPDATE_TICKET` | CRM | Write ticket status / notes |
| `GET_DEVICE` | Device | Read |
| `CHECK_WARRANTY` | Warranty | Read |
| `GET_DIAGNOSTICS` | Diagnostics | Read |
| `CREATE_REPAIR` | Repair | Insert repair order **(active path)** |
| `RECORD_PAID_OPTIONS` | Repair | Write paid-path artifact **(expired path)** |

Handlers query/mutate SQLite only. Never invent rows that are not seeded or created by prior ops in the run.

**Canonical Flow C outcome:** expired path calls `RECORD_PAID_OPTIONS` + `UPDATE_TICKET`; must **not** create a repair order.

---

## 6. Runtime database (SQLite)

### 6.1 Tables *(MVP)*

```text
customers(id, tier, email, ...)
devices(id, customer_id, serial, model, ...)
warranties(id, device_id, status, end_date, crm_flag, ...)
tickets(id, customer_id, device_id, status, summary, ...)
diagnostics(id, device_id, result_code, detail, ...)
repair_orders(id, ticket_id, device_id, status, created_at, ...)  -- mutable
workflows(id, name, version, status, dag_json, created_at, ...)
executions(id, workflow_id, case_id, status, started_at, ended_at, ...)
execution_events(id, execution_id, seq, type, node_id, payload_json, ts)
```

### 6.2 Seed / reset

- On startup (or admin reset): wipe mutable tables as needed → load entities from `data/generated/`.  
- Preserve frozen files on disk.  
- Optional: clear `executions` / keep audit — default **clear executions on reset** for clean demos.

---

## 7. FastAPI contracts

Base URL: `http://<host>:8000`  
Content-Type: `application/json`

### 7.1 Dataset

`GET /dataset/procedure-package/warranty`

**Response:** Procedure Package for Teach Studio (goal, sop, policies, demonstrations, tools, source ids).

### 7.2 Workflows

`POST /workflows/generate`

```json
{
  "procedure_package_id": "warranty",
  "overrides": {}
}
```

**Behavior:**

1. Load package from canonical dataset.  
2. Call `services.nvidia_inference.complete(...)` → Red Hat vLLM on Brev (`NVIDIA_VLLM_BASE_URL`) with compile prompt.  
3. Parse JSON DAG (strip fences if needed).  
4. Validate; on failure optionally **one** retry via the **same** NVIDIA gateway (validator errors in prompt) — never another provider.  
5. Persist workflow `status=draft`.  
6. Return `{ workflow_id, dag, validation, latency_ms }`.  

**Demo rules:** no ground-truth substitution; no third-party LLM keys. UI may keep the last validated model DAG if a later generate flakes.

`POST /workflows/{id}/activate` → `status=active`  
`GET /workflows/{id}` → metadata + DAG  

### 7.3 Cases

`GET /cases?scenario=active_warranty|expired_warranty|all`  
`GET /cases/{case_id}`

### 7.4 Executions

`POST /executions`

```json
{
  "workflow_id": "WF-...",
  "case_id": "CS-..."
}
```

**Response:** `{ execution_id, status, events[], artifacts{}, duration_ms }`

`GET /executions`  
`GET /executions/{id}`

### 7.5 Metrics

`GET /metrics/overview`

```json
{
  "seeded": { "total_executions": 12865, "note": "illustrative" },
  "live": { "executions": 2, "succeeded": 2, "failed": 0 },
  "recent_live": []
}
```

### 7.6 Admin

`POST /admin/reset` → reseed SQLite from canonical dataset.

### 7.7 Config env

```text
DATA_DIR=./data/generated
DB_PATH=./backend/runtime.db

# Required — self-hosted NVIDIA stack on Brev (no public LLM defaults)
NVIDIA_VLLM_BASE_URL=https://<brev-host>/v1
NVIDIA_VLLM_MODEL=nvidia/NVIDIA-Nemotron-Nano-9B-v2   # or merged TeachOnce checkpoint id
NVIDIA_VLLM_API_KEY=optional                          # only if our gateway requires it
NVIDIA_ALLOWED_MODEL_IDS=nvidia/NVIDIA-Nemotron-Nano-9B-v2,teachonce-nemotron-nano-9b
NVIDIA_ALLOWED_INFERENCE_HOSTS=<brev-host>

# Forbidden to set for AI: OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
ALLOW_GROUND_TRUTH_GENERATE=false   # true only for local executor tests (not an LLM fallback)
```

Startup must validate `NVIDIA_VLLM_BASE_URL` host ∈ allowlist and model id ∈ allowlist.

---

## 8. Generate path (compiler)

### 8.1 Prompt shape *(MVP)*

- **System:** Compile into executable DAG JSON matching schema X. **Output JSON only** (no markdown, no reasoning).  
- **User:** Warranty Procedure Package + schema summary + tool list.  

Prefer JSON / constrained decoding if Red Hat vLLM supports it; else prompt + strip fences + validate. Pin thinking/budget off or minimal when serving Nano 9B v2 for generate.

### 8.2 NVIDIA inference gateway (`services/nvidia_inference.py`)

Centralize **all** generative calls here (compile, optional NL augmentation, eval helpers).

```text
complete(messages, *, model=None, response_format=None) -> InferenceResult
  - Resolve base_url from NVIDIA_VLLM_BASE_URL (required)
  - Reject if host not allowlisted
  - Resolve model from NVIDIA_VLLM_MODEL / arg; reject if not allowlisted
  - Call OpenAI-compatible chat.completions with explicit base_url
    (do not use SDK default api.openai.com)
  - Return text + latency_ms + raw usage metadata
  - On misconfig → ConfigurationError (no fallback provider)
  - On transport/model failure → propagate; API layer maps to HTTP 502/503
```

Do not import or call OpenAI/Anthropic/Gemini/Groq/OpenRouter/Together/Fireworks/Cursor model clients elsewhere.

### 8.3 Serving artifact

1. Train LoRA with NeMo AutoModel on Brev.  
2. **Merge** adapter into base weights (`vllm_merge_lora` requirement for Mamba).  
3. Point Red Hat vLLM at merged directory on Brev.  
4. Smoke via `nvidia_inference` → warranty package → valid DAG before wiring UI.  
5. Record checkpoint id + URL in `docs/brev.md`.  

---

## 9. Executor algorithm

```text
function execute(workflow, case_id):
  ctx = { case: load_case(case_id), nodes: {}, branch: null }
  events = []
  node = start_node(workflow)

  while node.type != END:
    if node.type == ACTION:
      args = resolve_bindings(node.inputs, ctx)
      result = call_tool(node.tool, node.op, args)
      ctx.nodes[node.id] = result
      events.append(ToolEvent(...))
      node = unique_successor(workflow, node)

    else if node.type == DECISION:
      # MVP: two branches (Active / Expired); demos always match one
      name = first_matching_branch(node.branches, ctx)
      if name is null: fail execution  # no fancy default/else
      events.append(DecisionEvent(name))
      node = successor_for_branch(workflow, node, name)

    else if node.type == START:
      node = unique_successor(workflow, node)

  artifacts = verify_outcome(workflow, ctx)   # read ticket/repair state
  status = SUCCEEDED if artifacts.ok else FAILED
  persist_execution(status, events, artifacts)
  return status, events, artifacts
```

**Critical:** Branching and tool calls follow the stored DAG only — never `if case_id == ...` outcome shortcuts.

### 9.1 Event types

`NODE_STARTED`, `TOOL_CALLED`, `TOOL_RESULT`, `BRANCH_SELECTED`, `POSTCONDITION_CHECKED`, `EXECUTION_COMPLETED`, `EXECUTION_FAILED`

---

## 10. End-to-end sequence (Flows A–B–C)

```mermaid
sequenceDiagram
    autonumber
    actor User as Support Operations Manager
    participant UI as TeachOnce UI
    participant API as FastAPI Backend
    participant DB as Canonical Dataset
    participant AI as Brev Nemotron via Red Hat vLLM
    participant DAG as DAG Executor
    participant Tools as Mock Enterprise APIs
    participant State as Runtime SQLite

    Note over UI,AI: UI never calls LLM providers; API uses nvidia_inference only

    User->>UI: Open Teach Studio
    UI->>API: Load available procedure package
    API->>DB: Read SOP, policy, demo, tool schemas
    DB-->>UI: Display process evidence

    User->>UI: Generate Workflow
    UI->>API: POST /workflows/generate
    API->>AI: nvidia_inference → compile procedure into DAG JSON
    AI-->>API: Generated executable DAG
    API->>API: Validate DAG structure and bindings
    API-->>UI: Validated workflow
    UI-->>User: Display interactive DAG

    User->>UI: Deploy and select customer case
    UI->>API: POST /executions
    API->>DAG: Start workflow with case ID

    DAG->>Tools: LOOKUP_CUSTOMER
    Tools->>State: Query customer record
    State-->>DAG: Customer details

    DAG->>Tools: GET_DEVICE / CHECK_WARRANTY
    Tools->>State: Read device and warranty
    State-->>DAG: Warranty status

    alt Active warranty
        DAG->>Tools: GET_DIAGNOSTICS
        Tools-->>DAG: Hardware failure
        DAG->>Tools: CREATE_REPAIR
        Tools->>State: Insert repair order
        State-->>DAG: Repair order ID
        DAG->>Tools: UPDATE_TICKET
        Tools->>State: Persist updated status
    else Expired warranty
        DAG->>Tools: RECORD_PAID_OPTIONS
        Tools->>State: Write paid-path artifact
        DAG->>Tools: UPDATE_TICKET
        Tools->>State: Persist paid-service ticket status
    end

    DAG->>State: Verify resulting business state
    DAG-->>API: Completion status and event log
    API-->>UI: Execution results
    UI-->>User: Show executed path and business outcome

    User->>UI: Open Overview
    UI->>API: Get execution metrics
    API-->>UI: Real run history and aggregated metrics
```

---

## 11. Frontend ↔ API mapping

| UI surface | Primary calls | Notes |
|------------|---------------|-------|
| Teach Studio | `GET .../procedure-package/warranty`, `POST /workflows/generate` | Show latency; surface validation errors; no browser LLM SDK |
| Workflow Editor | `GET /workflows/{id}`, `POST .../activate` | React Flow from `dag.nodes/edges` |
| Executions | `GET /cases`, `POST /executions`, `GET /executions/{id}` | Timeline from `events` |
| Overview | `GET /metrics/overview` | Separate seeded vs live |
| Integrations | static from tool registry (+ optional ping) | No fake “connected” without registry |

Visual reference: TeachOnce UI mock / `mock.mp4` (visual only) — do not redesign IA. **No Test Lab in MVP.**

---

## 12. ML pipeline procedures

### 12.1 Milestone 0 smoke

1. Provision Brev 1× A100 80GB; record $/hr in `docs/brev.md`.  
2. Load base Nano 9B; one completion.  
3. Short LoRA via Automodel recipe (tiny steps).  
4. Save adapter → merge → Red Hat vLLM serve.  
5. One OpenAI-compatible request returns text.  
6. Stop instance if idle.  

### 12.2 Full TeachOnce train

1. `prepare_automodel_dataset.py` converts `training/train.jsonl` to recipe format.  
2. Fork YAML: point dataset paths; keep LoRA dim=8, alpha=32, `exclude_modules: ["*.out_proj"]`.  
3. `automodel ... --nproc-per-node 1` *(adjust if recipe requires otherwise)*.  
4. `eval_compiler.py`: parse rate, schema pass rate, tool-op validity on val split.  
5. Merge → redeploy vLLM → warranty package golden smoke.  

---

## 13. Error handling *(MVP — keep dumb)*

| Failure | Behavior |
|---------|----------|
| `NVIDIA_VLLM_BASE_URL` missing / host not allowlisted | 503; **no** external fallback |
| vLLM / Nemotron unreachable on Brev | 502; retry / check Brev |
| Invalid DAG JSON | 422; **one** regenerate via same NVIDIA gateway |
| Unknown tool / missing entity | Fail that node; log event |
| No decision branch matched | Fail execution (demo cases always match) |
| Postcondition fail | Execution `FAILED` with verify event |

No autonomous repair, no `NEEDS_REVIEW` UI, no third-party LLM fallback.

---

## 14. Testing strategy *(MVP minimum)*

| Layer | What |
|-------|------|
| Dataset | `validate_dataset.py` integrity + checksums |
| Tools + executor | Active → repair row; expired → paid-options row, no repair |
| Generate | Live Brev call via `nvidia_inference` → schema-valid DAG |
| AI policy | No third-party LLM SDKs/keys; gateway is sole outbound AI path |
| E2E | generate → activate → two cases → live metrics ≥ 2 |

Skip exhaustive validator unit matrices until A–C work.

---

## 15. Definition of done (LLD checklist)

- [ ] Frozen warranty-only `data/generated` with manifest  
- [ ] Pydantic DAG + validator green on ground-truth warranty DAG  
- [ ] MVP tools read/write SQLite (including `RECORD_PAID_OPTIONS`)  
- [ ] Executor runs active + expired from **stored** DAG  
- [ ] `/workflows/generate` → `nvidia_inference` → Brev; validated DAG  
- [ ] `NVIDIA_VLLM_BASE_URL` + allowlists enforced; no third-party LLM keys  
- [ ] Frontend has no AI provider SDKs  
- [ ] UI: Teach Studio → Editor → Executions → Overview wired  
- [ ] `ALLOW_GROUND_TRUTH_GENERATE=false` for demo  
- [ ] `docs/brev.md` documents instance, train, serve, stop, checkpoint IDs  

---

## Document history

| Date | Change |
|------|--------|
| 2026-10-17 | Initial LLD: modules, DAG schema, APIs, executor algorithm, sequence diagram, ML procedures |
| 2026-10-17 | Enforced NVIDIA-only gateway (`nvidia_inference.py`, `NVIDIA_VLLM_BASE_URL`, allowlists, no third-party LLM fallbacks) |
| 2026-10-17 | MVP cut: warranty A–C only; unify expired path; drop Test Lab; simplify predicates/errors/tests |
