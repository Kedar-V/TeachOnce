# Phase 3 — FastAPI + TeachOnce UI

**Status:** `not_started`  
**Depends on:** Phase 1 APIs/data; Phase 2 generate preferred  
**Tracker:** [`../implementation.md`](../implementation.md)

## Goal

Ship MVP UI matching the TeachOnce mock: Teach Studio (**expert video** + aligned trace) → Generate (Nemotron) → React Flow IR/DAG → Executions → Overview, all against real FastAPI + local/Brev inference.

## Checklist (high level)

- [ ] FastAPI: procedure package (incl. `video_id` / URI), generate, activate, cases, executions, metrics, reset  
- [ ] React: Overview, Teach Studio (play/scrub source video), Editor, Executions, Integrations  
- [ ] Render **model** IR/DAG (not hardcoded)  
- [ ] No frontend AI provider SDKs  
- [ ] Flows A–B clickable  

## Detail

Expand when Phase 2 generate is live. See [`../hld.md`](../hld.md) · [`../lld.md`](../lld.md).
