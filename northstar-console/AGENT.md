# Agent control — Northstar (3 procedures)

Drive this UI like a human. Prefer `observe → click/type → observe` over `startDemo()`.

```bash
cd northstar-console && python3 -m http.server 8765
```

## Procedures (distinct DAG families)

| `procedure` | Paths (`path=`) | Family | DAG shape |
|-------------|-----------------|--------|-----------|
| `warranty` | `active` \| `expired` | `warranty_resolution` | … → CHECK_WARRANTY → **Active/Expired** branch |
| `parts` | `in_stock` \| `backorder` | `parts_fulfillment` | … → CHECK_PARTS_STOCK → **Ship/Backorder** |
| `escalate` | `p1` \| `standard` | `severity_escalation` | … → ASSESS_SEVERITY → **On-call/L2** |

### URLs

```
/?procedure=warranty&path=active&hud=0
/?procedure=warranty&path=expired&hud=0
/?procedure=parts&path=in_stock&hud=0
/?procedure=parts&path=backorder&hud=0
/?procedure=escalate&path=p1&hud=0
/?procedure=escalate&path=standard&hud=0
```

(`scenario=active|expired` still works as alias for warranty `path`.)

## API

```js
Northstar.observe()
Northstar.act("click"|"type"|..., { testid, text })
Northstar.act("set_procedure", { procedure: "parts", path: "in_stock" })
Northstar.act("playbook_step", { op: "CHECK_PARTS_STOCK" })
Northstar.playbook()      // ordered ops for current path
Northstar.procedures()    // list all procedures + paths
```

## Playbooks

**warranty / active:**  
LOOKUP_CUSTOMER → GET_TICKET → GET_DEVICE → CHECK_WARRANTY → GET_DIAGNOSTICS → CREATE_REPAIR → UPDATE_TICKET

**warranty / expired:**  
… → RECORD_PAID_OPTIONS → UPDATE_TICKET

**parts / in_stock:**  
LOOKUP_CUSTOMER → GET_TICKET → GET_DEVICE → CHECK_PARTS_STOCK → CREATE_PARTS_SHIPMENT → UPDATE_TICKET

**parts / backorder:**  
… → CREATE_BACKORDER → UPDATE_TICKET

**escalate / p1:**  
GET_TICKET → LOOKUP_CUSTOMER → ASSESS_SEVERITY → PAGE_ONCALL → UPDATE_TICKET

**escalate / standard:**  
… → ASSIGN_L2_QUEUE → UPDATE_TICKET

## Stable testids (common)

`customer-query`, `btn-search-customer`, `row-ticket`, `row-device`, `btn-save-ticket`,  
`btn-check-warranty`, `btn-run-diag-page`, `btn-submit-repair`, `btn-record-paid`,  
`btn-check-stock`, `btn-create-shipment`, `btn-create-backorder`,  
`btn-assess-severity`, `btn-page-oncall`, `btn-assign-l2`,  
`nav-*` per procedure sidebar.

Full discoverability: `Northstar.targets()` after each render.
