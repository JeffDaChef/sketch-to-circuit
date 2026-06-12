# Roadmap & Strategy

This file captures *why* we're building things in a certain order, the college-application
strategy behind the project, and the "cool feature" ideas parked for later so they don't
get lost. The authoritative build order is still `sketch_to_circuit_brief.md` §7; this file
is the living layer on top of it.

---

## The core strategy: moats beat features

By the time these applications go out, "I built a cool thing with AI" is table stakes — lots
of students will have one. What survives that is the stuff an AI *can't* produce on your
behalf. The brief already contains most of it; these are the real differentiators, ranked
above any feature:

1. **Original data in your own hand.** The domain-adaptation step — *"X% on strangers'
   drawings → Y% after fine-tuning on ~35 of my own"* — is the single most AI-proof thing in
   the project. An AI cannot draw circuits in your handwriting, photograph them with your
   camera, and annotate them. That before/after number is uniquely yours. (Phase 5.)
2. **You can defend it out loud.** If a judge/interviewer asks "what is Modified Nodal
   Analysis, and why *modified*?" and you can whiteboard it, the project is real no matter who
   typed the code. `EXPLAINED.md` exists for exactly this — budget real time to read it.
3. **Honest measurement, including failure.** Reporting where it breaks ("fails on crossing
   wires ~18% of the time, here's why") is the most credible engineering move there is, and the
   opposite of hidden AI slop. This is the self-defined extraction metric + ngspice validation
   (Phase 4). Judges score rigor, not polish.

**The move that quietly beats a feature:** write it up as a 2–4 page mini research paper —
problem, why the input was constrained, method, a results table (extraction accuracy, ngspice
agreement, domain-adaptation delta), and a limitations section. Plus a named competition result
(Congressional App Challenge / county→state science fair). That's what turns "cool AI project"
into "this person does real work."

---

## Parked feature ideas (Phase 4+ — do NOT build until the spine works end-to-end)

The whole point above: finish the working pipeline (detect → trace wires → netlist → solve →
overlay on a real photo, Phases 1–3) **before** adding any of these. A half-working project
with a flashy feature reads *worse* than a simple one that fully works. These are flourishes
for once the core is solid.

- ⭐ **Transient simulation (top pick).** The solver currently does DC steady-state only.
  Extend it to the time domain — watch a capacitor's voltage rise on a live curve as it
  charges (RC). Real EE depth, builds directly on the existing solver, and a graph animating
  live is a far stronger demo beat than static numbers. Needs numerical integration
  (backward-Euler / trapezoidal) and companion models for C/L. ~80% leverages what exists.
- **Critique that computes consequences.** The planned LLM "explain/critique" mode should go
  past "LED with no resistor" to the actual number: "this LED sees ~45 mA vs its 20 mA rating
  → it burns out; add ~220 Ω." Genuinely useful homework tool = a better essay story than
  "it detects components."
- **Export to a real netlist (SPICE/KiCad).** Let the extracted circuit open in real EDA
  software. Bridges the toy to the actual engineering ecosystem.
- **Confidence-aware detection.** Surface "60% sure this is a capacitor" and let the user
  correct it. Calibration/uncertainty is a research-grade idea and makes the UI honest.

**Skip** pure-flash additions (fancier overlay animations, more component types) — surface area
without a defensible idea.

> Sequencing note: "trust" comes from the ngspice validation + honest metric, which are nearly
> *free* given the solver already works. "Wow" comes from transient sim. Build trust first.

---

## College-application strengtheners (rigor moves, not features)

These cost little code but read as *research maturity* — the thing that separates "cool AI
project" from "this person does real work." Layer them on once the spine works:

- **Accuracy-vs-difficulty curve.** Don't report one number — report extraction accuracy
  *broken down by circuit complexity* (2 components vs 5 vs 8, with/without junctions). "It's
  98% on simple circuits and degrades to 70% past 6 components" is a real analysis a judge can
  probe, and it's honest.
- **Ablation study.** Show a number *with and without* a design choice — e.g. accuracy with the
  drafter split vs a naive random split (demonstrates the leakage you avoided), or with vs
  without a preprocessing step. Ablations are the single most "I did science" signal there is.
- **Formal error taxonomy.** Turn the honest-limitations section into a labelled table: each
  failure mode (crossing wires, misread junction, terminal-match miss), how often it happens,
  and a worked example image. This is the credibility centerpiece of the writeup.
- **Baseline comparison.** Compare your from-scratch result against something off-the-shelf
  (e.g. a generic detector with no class remap, or the published CGHD mAP) to show your method
  adds value, not just that it exists.
- **Hosted / shareable demo.** A tiny web page or recorded pipeline that a stranger can try or
  watch end-to-end — shows you can *ship*, not just prototype. (Low priority vs the above.)
- **Reproducibility.** Seeds fixed, `requirements.txt` pinned, a one-command "regenerate all my
  numbers" script. Quietly signals engineering discipline; also makes the writeup bulletproof.

> The metric infrastructure for most of these already exists: `solver/equivalence.py` +
> `metrics/` give the accuracy number; the difficulty/ablation cuts are just *grouping* the
> same runs differently. Cheap rigor.

---

## What's buildable RIGHT NOW without the CGHD dataset

The dataset only blocks *training the detector*. Everything below is testable against the
synthetic generator (which gives us images + ground-truth boxes + the known-correct netlist),
so we can build it and *verify it against a known answer* today:

- **Wire extraction** (`vision/`) — the hardest module; built and passing on synthetic images
  (200/200 recover the correct netlist on the two existing templates). ✅ first version done.
  **Known limitations to revisit in Phase 2 (honest failure modes):**
  (a) the ground "x-column overlap" fallback and (b) the junction top/bottom-half rule both
  assume the synthetic layout (vertical components, ground directly below). They pass all
  synthetic tests but will need genuine terminal-stub tracing to survive real hand-drawn
  photos. This is exactly the kind of limitation the writeup should report, not hide.

  **FINDING (attempted adding a horizontal "series-loop" template):** the extractor does NOT
  yet generalize to new layouts, and the three heuristics are *coupled* — fixing one breaks
  another (classic whack-a-mole):
    * The voltage-source symbol has a large bbox whose leads attach far outside it (~50–90px),
      so its terminals don't find a wire by close-range probing → they fall through to the
      ground hack, which then wrongly grounds BOTH source terminals in a horizontal layout.
    * "Follow the lead outward along its axis" probing fixes the source but breaks the
      junction-free series resistor chains (and vice-versa).
  **Conclusion:** generalizing wire extraction needs a *principled redesign*, not more patches
  — most likely: build a proper skeleton GRAPH (nodes = wire endpoints/junctions, edges =
  traced paths) and snap component terminals onto it, rather than the current
  blob-proximity + special-case-fallback approach. This is its own focused task and the top
  Phase-2 hardening priority. (Good news: the `metrics/` harness + `circuit_equivalent` make
  it instantly measurable, so a redesign can be driven by the accuracy number.)
- **Preprocessing** (`vision/`) — grayscale → adaptive threshold → morphology → clean binary.
- **End-to-end extraction metric** — graph-isomorphism (networkx) check that an extracted
  netlist is electrically equivalent to ground truth. (Elevation layer #2.)
- **ngspice validation harness** — compare our MNA solver to ngspice on a suite of circuits.
  (Elevation layer #1; needs ngspice installed at run time, but the harness can be written now.)

## What genuinely needs you (a human)

- Downloading CGHD + running training on Colab (Phase 1 finish).
- Drawing/photographing/annotating your own ~35 circuits (Phase 5 — the biggest moat).
- Mounting the camera and recording the demo (Phase 3 / Phase 6).
