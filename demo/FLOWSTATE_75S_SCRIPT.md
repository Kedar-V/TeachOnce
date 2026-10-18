# FLOWSTATE — 75s Cinematic Elevator Pitch (Script Draft)

**Status:** v1 rendered (silent + burned-in captions) — iterate here  
**Render:** `motion-graphics-studio/out/flowstate-75sec-elevator-pitch.mp4` · source `motion-graphics-studio/src/flowstate/` · `npm run render:flowstate`  
**Subtitles:** [`flowstate-75s.srt`](flowstate-75s.srt) (timings also live in `CAPTIONS` in `src/flowstate/kit.tsx` — edit both)  
**Runtime:** exactly 75s · 1920×1080 · 30fps (2250 frames). Scene timing lives in `SCENES` in `src/flowstate/kit.tsx` as `[from, to, stretch]`; scenes 1–5 share one stretch (`PACE`) so pacing is even, extra duration is a hold. Every scene cross-dissolves into the next (`xf` in `FlowStatePitch.tsx`) — no hard cuts or dips to black.  
**Tone:** product reveal film (not tech explainer)  
**Palette:** `#0B0F14` · `#76B900` · `#45D9F5` · `#F5A45D` · `#F8FAFC`  
**Arc:** Frustration → cost → silence → reveal → demo → automation → $ payoff → close  

*Narration is the spoken track. Visual / SFX are production cues. Brand name “FlowState” must not appear before 0:22.*

---

## SCENE 1 — THE PROBLEM (0:00–0:12)

**Narration**

> Every company runs on procedures. But the instructions? “Check eligibility. Take appropriate action. Escalate when necessary.”

**Visual**

- Overhead night office; push into one desk buried in docs
- Document title: STANDARD OPERATING PROCEDURE
- Three lines appear in sequence, then critical words dissolve into `?`
  - Verify eligibility
  - Take appropriate action
  - Escalate if necessary
- SOP expands into a maze of unclear decisions (minimal text)

**SFX**

Quiet office ambience · paper · soft mouse clicks · low tension bed

---

## SCENE 2 — THE COST (0:12–0:22)

**Narration**

> But the real process lives in people’s heads. Turning it into automation takes days of mapping, coding, and testing.

**Visual**

- Split screen
  - Left: experienced agent sails through a support case
  - Right: engineers drowning in flowcharts, code, conflicting docs, calendar flips, test loops
- End: tangled overlapping connections → zoom to center → **freeze**

**SFX**

Rhythm rises · typing stacks · notifications · cut to brief silence on freeze

---

## SCENE 3 — REVEAL (0:22–0:29.2)

**Narration**

> Introducing… FlowState.

*(pause after “Introducing…” — let the impact land)*

**Visual**

- Freeze cluttered frame → room goes dark
- Tangled lines collapse into one green point → bass hit
- Point → luminous node → minimal DAG → original geometric FlowState mark
- Wordmark: **FLOWSTATE**
- Tagline: *From demonstration to automation.*
- Controlled green light sweep — memorable, not flashy

**SFX**

Silence → low-frequency impact → clean synth → optimistic pulse begins

---

## SCENE 4 — SOLUTION (0:29.2–0:39.8)

**Narration**

> Record a workflow once. Our AI learns the actions, tools, and decisions, then turns that demonstration into an executable workflow.

**Visual**

- Logo resolves into product UI
- Expert runs: open case → check warranty → verify eligibility → approve replacement **or** escalate
- Recording pulse → clip compresses → abstract AI pass → structured DAG assembles
- Nodes: Fetch Customer · Verify Warranty · Check Eligibility · Create Replacement · Escalate Case
- Branch condition visible; editable tool calls / edges
- Do **not** imply the model invents unsupported decisions with certainty

**SFX**

Soft UI clicks · light graph connection ticks

---

## SCENE 5 — AUTOMATION (0:39.8–0:50) — grid of completed cases holds ~1.5s

**Narration**

> Now, that workflow can run automatically. No manual scripting. No rebuilding the process from scratch.

**Visual**

- Zoom into DAG → **Run Workflow** → click
- Nodes fire in dependency order: customer → warranty → eligibility → replacement → done
- Success branch glows green
- One case → many cases on the same reusable workflow
- If real app UI available, use it; else label as conceptual product demo

**SFX**

Satisfying node-complete cues · steady optimistic pulse

---

## SCENE 6 — FINANCIAL IMPACT & BUSINESS VALUE (0:50–1:10)

**Narration**

> Ten thousand requests a month. Save just three minutes each, and that’s two hundred and forty thousand dollars a year in recovered labor capacity.

**Part A — hero calculation (0:50–1:01)** — each number holds before the next appears

- 10,000-cell request grid sweeps in; counter **10,000 REQUESTS / MONTH** (on “Ten thousand requests”)
- **3 MIN SAVED PER REQUEST** highlights; grid washes orange (on “three minutes”)
- Cells collapse into one glowing green time bar → **500 HRS RECOVERED / MONTH**
- Formula tokens stagger in: `500 hours × $40/hour × 12 months`
- Counter lands **$240,000 / YEAR** (on “two hundred and forty thousand”) — *Potential annual labor capacity value* · *Illustrative scenario — not actual customer savings*

**Part B — ripple (1:01–1:10)** — cards ~0.7s apart, full set holds ~2s before collapsing

> Plus faster service, fewer errors, and lower development effort.  *(added line — covers the card reveal; cut if you prefer silence)*

- $240K shrinks to center; label **POTENTIAL BENEFITS** · *projected outcomes, not measured results*
- Six cards spring in one at a time, left/right columns, gentle orbit drift:
  - Lower automation development costs (gold · code brackets ↓)
  - Faster customer resolutions (cyan · clock + speed lines)
  - Improved customer satisfaction (cyan · smile)
  - Fewer operational errors (green · shield ✓)
  - Faster employee onboarding (green · people ↑)
  - Preserved institutional knowledge (gold · connected nodes)
- No invented percentages on any card
- Lower-third: *Illustrative SaaS model: $2K/month/customer* · *$24K potential ARR/customer — hypothetical pricing, not actual revenue*
- Cards collapse into a central 4-node DAG glyph

**SFX**

Number ticks on each counter · restrained rise into $240K · soft pops per card · whoosh on collapse

---

## SCENE 7 — CLOSE (1:10–1:15)

**Narration**

> FlowState. Show it once. Automate it forever.

**Visual**

- DAG glyph from Scene 6 lifts and morphs into the FlowState mark
- **FLOWSTATE** · **SHOW IT ONCE.** · **AUTOMATE IT FOREVER.**
- Green pulse, then ~1s quiet hold

**SFX**

Signature sting · resolve to silence

---

## Timestamped narration (for VO / captions)

| Time | Line |
|------|------|
| 0:00–0:12 | Every company runs on procedures. But the instructions? “Check eligibility. Take appropriate action. Escalate when necessary.” |
| 0:12–0:22 | But the real process lives in people’s heads. Turning it into automation takes days of mapping, coding, and testing. |
| 0:22–0:29.2 | Introducing… FlowState. |
| 0:29.2–0:39.8 | Record a workflow once. Our AI learns the actions, tools, and decisions, then turns that demonstration into an executable workflow. |
| 0:39.8–0:50 | Now, that workflow can run automatically. No manual scripting. No rebuilding the process from scratch. |
| 0:50–1:10 | Ten thousand requests a month. Save just three minutes each, and that’s two hundred and forty thousand dollars a year in recovered labor capacity. Plus faster service, fewer errors, and lower development effort. |
| 1:10–1:15 | FlowState. Show it once. Automate it forever. |

**VO note:** ~95–110 words. If rushed in rehearsal, shorten Scene 1 second sentence or Scene 4 middle clause — never rush the 0:22 reveal pause.

---

## Financial assumptions (script truth)

| Input | Value |
|-------|-------|
| Requests / month | 10,000 |
| Minutes saved / request | 3 |
| Fully loaded labor | $40 / hour |
| Months | 12 |
| **Capacity value** | **500 hr/mo · $20k/mo · $240k/yr** |
| Illustrative SaaS | $2,000/mo → $24k ARR / customer |

Label capacity as *potential labor capacity value*, not cash in bank.

---

## Open tweaks

- [ ] Keep brand **FlowState** vs align naming with TeachOnce for hackathon?
- [ ] Scene 4 warranty steps vs CS-214 SOP wording — match demo assets?
- [ ] Cut or keep $24k subscription card (risk of looking salesy)?
- [ ] VO talent: self-record vs ElevenLabs / silent + captions only?
- [x] Remotion build v1 (visuals + captions, no audio)
- [ ] Record VO and lay SFX per cues above, then re-render with `showCaptions: false` if VO carries it
- [x] 75s cut: uniform 1.5× pacing on scenes 1–5, cross-dissolves between all scenes

## Take notes

| Take | Date | Notes |
|------|------|-------|
| | | |
