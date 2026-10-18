# TeachOnce — Demo Video Script

**Status:** draft — iterate here  
**Format:** 3–5 min · screen + camera (solo) · [Notion instructions](https://concrete-panther-c83.notion.site/Demo-Video-Instructions-087f567d17cc8239839b011a10c5c0d3)  
**Core loop on camera:** SOP / expert video → Nemotron generate DAG → execute active → execute expired → dashboard  

**Target total:** ~3.5 min (solo — no team intro block)

---

## Cold open (~5s)

> I’m Kedar. This is TeachOnce — teach a support process once, get an executable workflow.

**On screen:** TeachOnce UI / logo

---

## Elevator pitch (~35s)

> Enterprises bury process knowledge in SOPs and expert walkthroughs. Engineers then hand-translate that into automation — slow and brittle.
>
> TeachOnce takes a warranty SOP plus an expert demo, and a fine-tuned NVIDIA Nemotron on Brev compiles it into a typed workflow DAG. The platform runs that DAG against tools — no third-party LLMs, inference only on our Brev endpoint.
>
> Same DAG, two cases: active warranty gets a repair; expired takes the paid path. That’s the product.

**On screen:** one slide or Overview — problem → DAG → outcomes

---

## Live demo (~75s)

### A — Teach → Generate (~30s)

> Here’s our laptop warranty SOP and an expert demo video. I hit Generate.

**On screen:** Teach Studio — [`SOP-CS-214_Laptop_Warranty.md`](SOP-CS-214_Laptop_Warranty.md) + demo video · Generate · spinner

> Nemotron returns a validated DAG — look up case, device, warranty, then branch.

**On screen:** Workflow Editor / React Flow — point at DECISION node

### B — Execute active (~20s)

> Active-warranty case. Run.

**On screen:** Executions — nodes light up → repair created · ticket updated

> Covered repair path. Done.

### C — Execute expired (~25s)

> Same workflow, expired warranty.

**On screen:** second run — paid-service branch · no repair order

> Conditional reasoning on one compiled DAG — not a hardcoded if-statement we wrote by hand.

**On screen:** dashboard / metrics showing both runs

---

## How we built it (~70s)

> Stack: fine-tuned Nemotron Nano on NVIDIA Brev, served with Red Hat vLLM. FastAPI gateway — NVIDIA only, hard-fail if the Brev URL is missing.
>
> Training data is one coherent warranty world: SOP, policies, demo traces, ground-truth DAG. Model compiles procedure → DAG JSON; a deterministic executor runs tools and branches.
>
> Hardware shaped the design — single A100, LoRA then merge for serve, keep the GPU off the execution hot path so the demo stays snappy.
>
> Biggest tradeoff: one process family, happy path only. Depth on warranty beats a shallow multi-SOP demo.

<!-- OPTIONAL side card (~20s) — cut if over time
> Side note: we can dogfood the same loop on our own Brev setup — record the steps, compile a setup DAG, render it back to a how-to Notion doc. Same teach → DAG → doc idea.
-->

**On screen:** HLD snippet or Brev + vLLM terminal briefly · back to UI

---

## So what? (~25s)

> So what: support orgs stop waiting on engineers to encode every SOP. A small specialized model turns procedure knowledge into inspectable automation — and we proved it end-to-end on NVIDIA infra in a weekend.

**On screen:** final DAG + two execution outcomes side by side · freeze

---

## Take / edit notes

| Take | Date | Notes |
|------|------|-------|
| | | |

## Open tweaks (edit freely)

- [ ] Exact Generate latency line (say real number after rehearsal)
- [ ] Name the model checkpoint / instance once in “how built”
- [ ] Confirm which demo video clip is on screen (warranty_01 vs CS-214)
- [ ] Keep or cut Brev dogfood side card
- [ ] Closing line — punchier?
