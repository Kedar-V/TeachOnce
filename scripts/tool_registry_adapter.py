"""
Tool-registry adapter: every train/eval DAG must speak registry ops, never RPA clicks.

Any source (VideoCUA action log, Mind2Web, self-shot trace) → normalize → validate
→ canonical TeachOnce DAG JSON. Quarantine rows that cannot map.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Locked MVP ops — keep in sync with docs/lld.md §5.2 / docs/plan.md §9
MVP_OPS: dict[str, dict[str, Any]] = {
    # warranty
    "LOOKUP_CUSTOMER": {"service": "crm", "inputs": ("query",)},
    "GET_TICKET": {"service": "crm", "inputs": ("ticket_id",)},
    "UPDATE_TICKET": {"service": "crm", "inputs": ("ticket_id",)},
    "GET_DEVICE": {"service": "device", "inputs": ("device_id",)},
    "CHECK_WARRANTY": {"service": "warranty", "inputs": ("device_id",)},
    "GET_DIAGNOSTICS": {"service": "diagnostics", "inputs": ("device_id",)},
    "CREATE_REPAIR": {"service": "repair", "inputs": ("device_id", "ticket_id")},
    "RECORD_PAID_OPTIONS": {"service": "repair", "inputs": ("device_id", "ticket_id")},
    # spare parts fulfillment
    "CHECK_PARTS_STOCK": {"service": "inventory", "inputs": ("part_sku",)},
    "CREATE_PARTS_SHIPMENT": {"service": "shipping", "inputs": ("part_sku", "ticket_id")},
    "CREATE_BACKORDER": {"service": "shipping", "inputs": ("part_sku", "ticket_id")},
    # severity escalation
    "ASSESS_SEVERITY": {"service": "crm", "inputs": ("ticket_id",)},
    "PAGE_ONCALL": {"service": "crm", "inputs": ("ticket_id",)},
    "ASSIGN_L2_QUEUE": {"service": "crm", "inputs": ("ticket_id",)},
}

_ALIAS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"check\s*warrant|warrant(y)?\s*status|coverage", re.I), "CHECK_WARRANTY"),
    (re.compile(r"look\s*up\s*customer|find\s*customer|search\s*customer", re.I), "LOOKUP_CUSTOMER"),
    (re.compile(r"get\s*ticket|open\s*ticket|load\s*case|find\s*case", re.I), "GET_TICKET"),
    (re.compile(r"update\s*ticket|close\s*ticket|add\s*note", re.I), "UPDATE_TICKET"),
    (re.compile(r"get\s*device|device\s*registry|serial", re.I), "GET_DEVICE"),
    (re.compile(r"diagnostic|run\s*diag", re.I), "GET_DIAGNOSTICS"),
    (re.compile(r"create\s*repair|repair\s*order|rma", re.I), "CREATE_REPAIR"),
    (re.compile(r"paid\s*(service|option)|out.?of.?warrant|expired\s*path", re.I), "RECORD_PAID_OPTIONS"),
    (re.compile(r"parts?\s*stock|check\s*stock|inventory", re.I), "CHECK_PARTS_STOCK"),
    (re.compile(r"create\s*shipment|ship\s*parts?|parts?\s*shipment", re.I), "CREATE_PARTS_SHIPMENT"),
    (re.compile(r"back.?order", re.I), "CREATE_BACKORDER"),
    (re.compile(r"assess\s*severity|severity", re.I), "ASSESS_SEVERITY"),
    (re.compile(r"page\s*on.?call|on.?call", re.I), "PAGE_ONCALL"),
    (re.compile(r"assign\s*l2|l2\s*queue", re.I), "ASSIGN_L2_QUEUE"),
]

_RPA_KEYS = frozenset(
    {"x", "y", "click", "mousedown", "mouseup", "mousemove", "scroll", "key", "keycode", "selector", "xpath", "bbox"}
)


def map_step_to_op(step: dict[str, Any] | str) -> str | None:
    """Map one source step (text or action dict) to a registry op, or None if unmapped."""
    if isinstance(step, str):
        text = step
    else:
        if step.get("op") in MVP_OPS:
            return step["op"]
        text = " ".join(
            str(step.get(k, ""))
            for k in ("op", "action", "label", "name", "description", "text", "tool_hint", "type")
        )
        # Reject pure RPA payloads with no semantic label
        if _RPA_KEYS & set(step.keys()) and not any(
            step.get(k) for k in ("label", "name", "description", "tool_hint", "op", "action")
        ):
            return None
    for pat, op in _ALIAS:
        if pat.search(text):
            return op
    return None


def steps_to_ops(steps: list[Any]) -> list[str]:
    ops: list[str] = []
    for s in steps:
        op = map_step_to_op(s)
        if op and (not ops or ops[-1] != op):
            ops.append(op)
    return ops


_BRANCH_TAILS = frozenset(
    {
        "CREATE_REPAIR",
        "RECORD_PAID_OPTIONS",
        "CREATE_PARTS_SHIPMENT",
        "CREATE_BACKORDER",
        "PAGE_ONCALL",
        "ASSIGN_L2_QUEUE",
    }
)


def _append_action(nodes, edges, prev, op, nid):
    meta = MVP_OPS[op]
    nodes.append(
        {
            "id": nid,
            "type": "ACTION",
            "label": op.replace("_", " ").title(),
            "tool": meta["service"],
            "op": op,
            "inputs": {k: {"ref": f"$case.{k}"} for k in meta["inputs"]},
        }
    )
    edges.append({"from": prev, "to": nid, "when": None})
    return nid


def _branch_decision(nodes, edges, prev, decide_id, label, branches):
    """branches: list[(branch_name, op)]"""
    nodes.append(
        {
            "id": decide_id,
            "type": "DECISION",
            "label": label,
            "branches": [{"name": b} for b, _ in branches],
        }
    )
    edges.append({"from": prev, "to": decide_id, "when": None})
    for branch, op in branches:
        if op not in MVP_OPS:
            continue
        nid = f"branch_{branch.lower().replace(' ', '_')}"
        meta = MVP_OPS[op]
        nodes.append(
            {
                "id": nid,
                "type": "ACTION",
                "label": op.replace("_", " ").title(),
                "tool": meta["service"],
                "op": op,
                "inputs": {k: {"ref": f"$case.{k}"} for k in meta["inputs"]},
            }
        )
        edges.append({"from": decide_id, "to": nid, "when": branch})
        end_id = f"end_{branch.lower().replace(' ', '_')}"
        nodes.append({"id": end_id, "type": "END", "outcome": "SUCCEEDED"})
        edges.append({"from": nid, "to": end_id, "when": None})


def build_dag(
    ops: list[str],
    *,
    workflow_id: str = "WF-adapted",
    name: str = "Adapted workflow",
    include_warranty_branch: bool = True,
) -> dict[str, Any]:
    """Build START→ACTIONs→optional DECISION→END. All ACTION.op ∈ MVP_OPS."""
    unknown = [o for o in ops if o not in MVP_OPS]
    if unknown:
        raise ValueError(f"ops not in tool registry: {unknown}")

    nodes: list[dict[str, Any]] = [{"id": "start", "type": "START"}]
    edges: list[dict[str, Any]] = []
    prev = "start"
    linear = [o for o in ops if o not in _BRANCH_TAILS]
    for i, op in enumerate(linear):
        prev = _append_action(nodes, edges, prev, op, f"a{i}_{op.lower()}")

    branched = False
    if include_warranty_branch and "CHECK_WARRANTY" in ops:
        _branch_decision(
            nodes,
            edges,
            prev,
            "decide_warranty",
            "Warranty status",
            [("Active", "CREATE_REPAIR"), ("Expired", "RECORD_PAID_OPTIONS")],
        )
        branched = True
    elif "CHECK_PARTS_STOCK" in ops:
        _branch_decision(
            nodes,
            edges,
            prev,
            "decide_stock",
            "Parts stock",
            [("InStock", "CREATE_PARTS_SHIPMENT"), ("Backorder", "CREATE_BACKORDER")],
        )
        branched = True
    elif "ASSESS_SEVERITY" in ops:
        _branch_decision(
            nodes,
            edges,
            prev,
            "decide_severity",
            "Severity",
            [("P1", "PAGE_ONCALL"), ("Standard", "ASSIGN_L2_QUEUE")],
        )
        branched = True

    if not branched:
        # path-specific single tail from ops
        for op in ops:
            if op in _BRANCH_TAILS:
                prev = _append_action(nodes, edges, prev, op, f"tail_{op.lower()}")
        nodes.append({"id": "end", "type": "END", "outcome": "SUCCEEDED"})
        edges.append({"from": prev, "to": "end", "when": None})

    return {"id": workflow_id, "name": name, "version": "1.0.0", "nodes": nodes, "edges": edges}


def repair_registry_dag(dag: dict[str, Any]) -> dict[str, Any]:
    """Make guided_json outputs pass validate_registry_dag when possible.

    guided_schema cannot require an END node (xgrammar has no reliable `contains`),
    so models often emit START/ACTION/DECISION only, or edges to missing ids.
    """
    if not isinstance(dag, dict):
        return dag
    nodes_in = dag.get("nodes") or []
    edges_in = dag.get("edges") or []
    if not isinstance(nodes_in, list) or not isinstance(edges_in, list):
        return dag
    nodes = [dict(n) for n in nodes_in if isinstance(n, dict) and n.get("id")]
    # lock tool from registry when op is known (guided oneOf should already; repair drift)
    for n in nodes:
        if n.get("type") == "ACTION" and n.get("op") in MVP_OPS:
            n["tool"] = MVP_OPS[n["op"]]["service"]
    ids = {n["id"] for n in nodes}
    edges = [
        dict(e)
        for e in edges_in
        if isinstance(e, dict) and e.get("from") in ids and e.get("to") in ids
    ]
    types = {n["id"]: n.get("type") for n in nodes}
    if nodes and not any(t == "END" for t in types.values()):
        end_id = "end"
        i = 0
        while end_id in ids:
            i += 1
            end_id = f"end_{i}"
        nodes.append({"id": end_id, "type": "END", "outcome": "SUCCEEDED"})
        ids.add(end_id)
        outgoing = {e.get("from") for e in edges}
        dangling = [
            n["id"]
            for n in nodes
            if n.get("type") in ("ACTION", "DECISION", "START") and n["id"] not in outgoing
        ]
        if not dangling:
            dangling = [n["id"] for n in nodes if n.get("type") != "END"][-1:]
        for did in dangling:
            edges.append({"from": did, "to": end_id, "when": None})
    out = dict(dag)
    out["nodes"] = nodes
    out["edges"] = edges
    return out


def validate_registry_dag(dag: dict[str, Any]) -> list[str]:
    """Return error strings; empty == ok. Enforces tool-registry philosophy."""
    errs: list[str] = []
    nodes = dag.get("nodes") or []
    edges = dag.get("edges") or []
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return ["nodes/edges must be lists"]
    nodes = [n for n in nodes if isinstance(n, dict)]
    if len(nodes) != len(dag.get("nodes") or []):
        errs.append("non-dict node entries")
    ids = [n.get("id") for n in nodes]
    if len(ids) != len(set(ids)):
        errs.append("duplicate node ids")
    types = {n.get("id"): n.get("type") for n in nodes}
    if sum(1 for t in types.values() if t == "START") != 1:
        errs.append("need exactly one START")
    if not any(t == "END" for t in types.values()):
        errs.append("need at least one END")
    for n in nodes:
        if n.get("type") == "ACTION":
            op = n.get("op")
            if op not in MVP_OPS:
                errs.append(f"ACTION {n.get('id')}: op {op!r} not in tool registry")
            else:
                if n.get("tool") != MVP_OPS[op]["service"]:
                    errs.append(f"ACTION {n.get('id')}: tool service mismatch for {op}")
            for k in n.keys():
                if k in _RPA_KEYS:
                    errs.append(f"ACTION {n.get('id')}: RPA field {k!r} forbidden")
        if n.get("type") not in ("START", "ACTION", "DECISION", "END", None):
            # allow only MVP node types in adapted DAGs
            if n.get("type") not in ("START", "ACTION", "DECISION", "END"):
                errs.append(f"node {n.get('id')}: type {n.get('type')!r} not in MVP set")
    for e in edges:
        if not isinstance(e, dict):
            errs.append(f"non-dict edge: {e!r}")
            continue
        if e.get("from") not in types or e.get("to") not in types:
            errs.append(f"edge endpoint missing: {e}")
    return errs


def adapt_record(
    source: str,
    steps: list[Any],
    *,
    workflow_id: str | None = None,
    name: str | None = None,
    require_min_ops: int = 1,
) -> dict[str, Any]:
    """
    Full adapter path for one dataset row.
    Returns {ok, dag?, ops, errors, source} — quarantine when ok is False.
    """
    ops = steps_to_ops(steps)
    wid = workflow_id or f"WF-{source}-adapted"
    if len(ops) < require_min_ops:
        return {
            "ok": False,
            "source": source,
            "ops": ops,
            "errors": [f"mapped only {len(ops)} ops (need ≥{require_min_ops}); quarantine"],
            "dag": None,
        }
    branch = "CHECK_WARRANTY" in ops
    dag = build_dag(ops, workflow_id=wid, name=name or wid, include_warranty_branch=branch)
    errs = validate_registry_dag(dag)
    return {"ok": not errs, "source": source, "ops": ops, "errors": errs, "dag": dag if not errs else None}


def demo() -> None:
    publicish = [
        {"type": "click", "x": 10, "y": 20, "label": "Search customer"},
        {"action": "open ticket CS-48213"},
        {"description": "check warranty status"},
        {"label": "create repair order"},
    ]
    r = adapt_record("videocua", publicish, workflow_id="WF-demo-public")
    assert r["ok"], r["errors"]

    bad = adapt_record("raw_rpa", [{"click": True, "x": 1, "y": 2}])
    assert not bad["ok"]

    parts = adapt_record(
        "parts",
        [
            {"op": "LOOKUP_CUSTOMER"},
            {"op": "CHECK_PARTS_STOCK"},
            {"op": "CREATE_PARTS_SHIPMENT"},
        ],
        workflow_id="WF-parts",
    )
    assert parts["ok"], parts["errors"]
    assert any(n.get("type") == "DECISION" for n in parts["dag"]["nodes"])

    esc = adapt_record(
        "escalate",
        [{"op": "GET_TICKET"}, {"op": "ASSESS_SEVERITY"}, {"op": "PAGE_ONCALL"}],
        workflow_id="WF-esc",
    )
    assert esc["ok"], esc["errors"]

    print(
        json.dumps(
            {
                "warranty_ops": r["ops"],
                "parts_decision": True,
                "escalate_ops": esc["ops"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    demo()
