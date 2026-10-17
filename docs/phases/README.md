# TeachOnce — Phase Plans

Living phase documents. Track day-to-day status in [`../implementation.md`](../implementation.md). Put deep how-to and acceptance criteria here.

| Phase | Doc | Status |
|-------|-----|--------|
| 0 | [`phase-0-infra.md`](phase-0-infra.md) | `done` |
| 1 | [`phase-1-dataset.md`](phase-1-dataset.md) | `not_started` ← **current** |
| 2 | [`phase-2-nemotron.md`](phase-2-nemotron.md) | `not_started` |
| 3 | [`phase-3-app.md`](phase-3-app.md) | `not_started` |
| 4 | [`phase-4-demo.md`](phase-4-demo.md) | `not_started` |

```text
Phase 0  Infra (Brev + vLLM + NeMo Gym)     done
Phase 1  Video-aligned IR dataset (video→DAG) ← NOW
Phase 2  Nemotron fine-tune + serve
Phase 3  FastAPI + TeachOnce UI
Phase 4  E2E demo polish (Flows A–C)
```

**Product north star:** expert screen-recording **video → executable DAG** via **Nemotron Nano 12B v2 VL** (12.6B, video in).  
**Product / policy:** [`../plan.md`](../plan.md) · **HLD/LLD:** [`../hld.md`](../hld.md) · [`../lld.md`](../lld.md)  
**Bounties (kickoff):** Nemotron Lightning · NeMo Gym · Red Hat vLLM — see [`../Small Models Hack Raleigh_Kickoff Pres.pdf`](../Small%20Models%20Hack%20Raleigh_Kickoff%20Pres.pdf)

**AI rule:** NVIDIA-only for all generative work. Phase 0 may have used a non-Nemotron smoke model for infra; product compile/serve and bounty narrative use **Nemotron**.
