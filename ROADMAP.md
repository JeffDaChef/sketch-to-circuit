# Roadmap & Strategy

This file captures *why* we're building things in a certain order, the college-application
strategy behind the project, and the "cool feature" ideas parked for later so they don't
get lost.

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

- ⭐ **Transient simulation (top pick).** ✅ **Built** (`solver/transient.py`). Extends the DC
  solver to the time domain via backward-Euler companion models — each capacitor becomes a
  resistor ∥ current source per step, so every instant is a plain R/V/I circuit handed to the
  existing (ngspice-validated) `solve()`. No second solver written. Matches the analytic RC
  curve (3.151 V vs theory's 3.16 V at one time constant); `python -m solver.transient` saves
  the charging curve. 18 tests incl. the analytic-curve anchor. **Now also built:** time-varying
  sources (sine), inductors ('L' kind, DC short + transient companion), series-RLC ringing, and
  trapezoidal integration (~20× more accurate than backward-Euler on the RC curve — a numerical
  ablation). **Remaining follow-up:** a *live* animated curve for the demo (currently saved
  PNGs: rc_charging, rlc_ringing, rectifier). Companion-model approach means transient results are
  themselves ngspice-validatable.
- ⭐ **AC / frequency-domain analysis.** ✅ **Built** (`solver/ac.py`). The same MNA core run with
  COMPLEX admittances (phasors): `Z_C=1/jωC`, `Z_L=jωL`. Sweeps frequency → transfer function
  H(f) → Bode plots. Verified against closed-form filters: RC low-pass hits −3 dB / −45° at its
  cutoff with −20 dB/decade rolloff; series-RLC band-pass peaks at f₀=1/(2π√LC). 8 analytic-anchor
  tests; `python -m solver.ac` saves `bode_lowpass.png` + `bode_bandpass.png`. Makes the solver a
  four-mode engine (DC + transient + non-linear + AC). ✅ **Small-signal AC also built:** linearises
  each diode at its DC operating point (r_d = n·V_t/I_D, the "26 mV/I" rule), so a diode is a
  bias-tunable resistor — demo builds a diode+C low-pass whose cutoff tracks the bias current 10:1
  (`bode_diode_tunable.png`). Remaining refinement: diode junction capacitance.
- **Critique that computes consequences.** The planned LLM "explain/critique" mode should go
  past "LED with no resistor" to the actual number: "this LED sees ~45 mA vs its 20 mA rating
  → it burns out; add ~220 Ω." Genuinely useful homework tool = a better essay story than
  "it detects components."
- **Export to a real netlist (SPICE/KiCad).** ✅ **Built** (`solver/spice_export.py`). Writes a
  complete, runnable SPICE `.cir` — proper diode `.model` lines (deduped), engineering-notation
  values (`10k`/`100u`, correct `MEG`-vs-`m`), and an optional analysis directive (`.op`/`.tran`/
  `.dc`). Opens unchanged in ngspice/LTspice/KiCad. `parse_value(format_value(x))==x` is pinned by
  test. 17 tests. Bridges the toy to the actual engineering ecosystem.
- **Confidence-aware detection.** Surface "60% sure this is a capacitor" and let the user
  correct it. Calibration/uncertainty is a research-grade idea and makes the UI honest.

**Skip** pure-flash additions (fancier overlay animations, more component types) — surface area
without a defensible idea.

---

## Difficulty levers (potential — bank these, build AFTER the spine works)

The synthetic-only pipeline came together fast because it was *designed* to be the easy 60%
(clean images we control). The hard part — real hand-drawn photos — is already on the
roadmap. These levers add *genuine* depth beyond that, ranked by hardness-per-impressiveness:

1. **Lift the no-crossing-wires constraint (research-grade).** ✅ **Built**
   (`vision/wire_extraction.py: thread_crossovers`). A plain '+' crossing fuses two wires into a
   degree-4 skeleton node; given a `crossover` box (detector stand-in; CGHD has the class) we
   split it, pairing the two most-opposite edges as each straight wire, so the nets stay
   distinct. Before/after proof: on a divider where the n1 rail crosses the n2 wire, *with* the
   box it extracts correctly, *without* it R1 comes out shorted. Hop-drawn crossovers (wires
   already apart in the skeleton) are left alone — graceful. 10 tests (unit + end-to-end +
   jitter). Story upgraded from "I constrained the problem" to "…and then I lifted the
   constraint." (Still open: have the *trained detector* emit crossover boxes on real photos.)
2. **Transient simulation + true nonlinear diodes (deep math).** ✅ **Both built.** Time-domain
   solving (`solver/transient.py`, backward-Euler companion models) AND Newton-Raphson on the
   exponential Shockley diode equation (`solver/nonlinear.py`) instead of the planned fixed-2V
   shortcut. Both reuse the validated linear `solve()` via companion models — the solver core is
   now a real numerical engine (DC + transient + non-linear) on one piece of linear algebra.
   ✅ **Combined too:** transient now runs Newton-Raphson *inside* each time step when diodes
   are present, plus time-varying sources (`sine()` helper) — so a **half-wave rectifier with a
   smoothing capacitor** simulates end-to-end (output 4.28 V, sub-volt ripple; `rectifier.png`).
   That one demo exercises all three solvers at once. ✅ **Also now done:** inductors ('L' kind,
   DC short + transient companion model), series-RLC ringing, and trapezoidal integration
   (~20× more accurate than backward-Euler on RC — a numerical-methods ablation). **Still open:**
   a *live* animated curve for the demo (currently saved PNGs).
3. **Noise-robustness study (rigor + Phase-2 prep).** ✅ **Built** (`metrics/noise_robustness.py`).
   Corrupts synthetic images (Gaussian blur, Gaussian noise, salt-pepper speckle) at rising
   severity and publishes the accuracy-vs-severity curve (`noise_robustness.png`). Findings (80
   circuits/level): blur tolerated (floors ~72%), noise has a cliff past σ≈45 (100%→20%), and
   **speckle is the weak point — even 0.4% halves accuracy**. Phase-2 to-do it directly implies:
   despeckle/denoise preprocessing + adaptive (local) thresholding instead of global Otsu. Per-
   extraction timeout added (heavy corruption explodes the skeleton graph). 7 tests. **Still
   open from the original idea:** elastic/wobble distortion and lighting-gradient curves (the
   global-Otsu lighting cliff is noted but not yet a clean published curve).
4. **The human moat (unchanged, still #1 overall):** ~35 own drawings + domain-adaptation
   delta, whiteboard-level command of MNA, the mini-paper with ablations + error taxonomy,
   a competition entry.

Ordering: finish the spine first (detector → real photos → live camera). Then #2 (mostly
leveraged off the existing solver), then #1 as the headline stretch, #3 woven in along the
way. Difficulty added to an unfinished project reads as scattered.

> Sequencing note: "trust" comes from the ngspice validation + honest metric, which are nearly
> *free* given the solver already works. "Wow" comes from transient sim. Build trust first.

---

## College-application strengtheners (rigor moves, not features)

These cost little code but read as *research maturity* — the thing that separates "cool AI
project" from "this person does real work." Layer them on once the spine works:

- **Accuracy-vs-difficulty curve.** ✅ **Built** (`metrics/difficulty.py`). Forces circuit size
  via the templates' new `k` param and charts accuracy vs component count: **100% up to ~10–12
  components** across all three families, then a fall-off the study honestly attributes to the
  fixed render resolution (components shrink below the skeletoniser's limit), not a logic failure.
  Building it caught — and fixed — a benchmark artifact (value labels crowding the wiring at high
  density were getting erased and cutting connections; `loc="right"` fixed it), which is itself a
  strong "I tell artifacts from real limits" story. Replaces the hollow "100% on 2–5 components".
- **Ablation study.** Show a number *with and without* a design choice. ✅ **First one done:**
  the extractor ablation (`metrics/extractor_ablation.py`) — blob-proximity vs skeleton-graph,
  40% → 100% overall. More to come (e.g. drafter split vs naive random split once training runs,
  with/without a preprocessing step). Ablations are the single most "I did science" signal there is.
- **Formal error taxonomy.** Turn the honest-limitations section into a labelled table: each
  failure mode (crossing wires, misread junction, terminal-match miss), how often it happens,
  and a worked example image. This is the credibility centerpiece of the writeup.
- **Baseline comparison.** ✅ **Done for wire extraction** (`metrics/extractor_ablation.py`): the
  skeleton-graph redesign vs the frozen blob-proximity baseline on identical circuits — overall
  **40% → 100%**, rescuing three layouts the baseline scored 0% on with no regression on the two
  it handled. (Still to do once the detector exists: compare the from-scratch detector against
  off-the-shelf / published CGHD mAP.)
- **Hosted / shareable demo.** A tiny web page or recorded pipeline that a stranger can try or
  watch end-to-end — shows you can *ship*, not just prototype. Note: most target schools don't
  interview, which makes the *artifacts* (writeup, repo, site, honest demo) carry everything —
  see [`docs/presentation_plan.md`](docs/presentation_plan.md) for the ranked plan and the rule
  that a camera demo only comes *after* the detector (never staged).
- **Reproducibility.** ✅ **Done.** Seeds fixed (every metric takes `--seed`), `requirements.txt`
  pinned to exact versions, and a one-command `./reproduce.sh` (with `PYTHONHASHSEED=0`)
  regenerates every number and figure. Quietly signals engineering discipline; also makes the
  writeup bulletproof.

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

  **RESOLVED — skeleton-graph redesign (branch `wire-extraction-redesign`).** The original
  blob-proximity extractor was layout-overfit (200/200 on the two original templates, 0/30 on
  a new horizontal layout; coupled heuristics made patching whack-a-mole). The redesign
  (see `docs/wire_extraction_redesign_plan.md`) builds a real graph of the wires
  (`vision/skeleton_graph.py`) and matches component terminals to the *cut endpoints* that
  erasure creates, via four evidence-ordered rules with zero layout assumptions. Results:
  **200/200 on all three templates** (incl. the previously-impossible series_loop, 30/30) and
  102 tests green. Investigating the failure also exposed two generator bugs now fixed:
  the series_divider drawing never visually closed its loop, and value labels had no `text`
  ground-truth boxes (label ink masqueraded as wire). The old extractor is frozen at
  `vision/wire_extraction_baseline.py` for before/after comparison — writeup material.
  `vision/debug_viz.py` renders the extractor's full mental model onto the image for
  debugging real photos later.

  **Post-redesign adversarial poke (same branch):** probing with two more unseen layouts
  (corner-turning chain: 7/30; mirrored loop: 16/30) exposed two real weaknesses, both fixed:
  (1) matching caps scaled with *image* size but the relevant geometry scales with *component*
  size — caps now derive from the median component box length; (2) the touching-terminals rule
  required both terminals unresolved — now a two-tier rule lets a resolved terminal connect an
  unresolved neighbour at tight range (ground symbol pressed against a source's foot). After
  the fixes: 30/30 on ALL unseen layouts, kept as permanent guards in
  `tests/test_wire_extraction_generalization.py` (104 tests green, official metric 200/200).
  Also caught: my own probe script initially drew a short-circuit (wire through a component
  body) — the extractor read that drawing *correctly* and the "failure" was the test's. Same
  lesson as the generator bugs: verify what the picture shows before blaming the algorithm.
- **Preprocessing** (`vision/`) — grayscale → adaptive threshold → morphology → clean binary.
- **End-to-end extraction metric** — graph-isomorphism (networkx) check that an extracted
  netlist is electrically equivalent to ground truth. (Elevation layer #2.)
- **ngspice validation harness** — compare our MNA solver to ngspice on a suite of circuits.
  (Elevation layer #1.) ✅ **Built** (`solver/ngspice_validation.py`): builds a SPICE deck, runs
  `ngspice -b`, parses the output, diffs against our `solve()` within tolerance. Deck-building
  and parsing are unit-tested against canned ngspice output (12 tests); the one live comparison
  test auto-skips until ngspice is installed (one-time admin step), then runs the hand-checked
  suite. Run it any time with `python -m solver.ngspice_validation`.

## Hardware notes

- **Mac (M1, 16 GB RAM):** daily dev machine — code, tests, debug. Gets hot under sustained
  load; do NOT use it for training or heavy inference.
- **GPU PC (16 GB VRAM):** the right machine for GPU-heavy work. Claude Code isn't on it, but
  it can run training jobs or serve Ollama.
- **LLM inference plan:** the planned LLM critique feature (Phase 4) will run via **Ollama on
  the GPU PC**, not the Anthropic API — zero operating cost. Training (YOLO fine-tuning) can
  still use Google Colab; that's a separate decision.

---

## What genuinely needs you (a human)

- Downloading CGHD + running training on Colab (Phase 1 finish).
- Drawing/photographing/annotating your own ~35 circuits (Phase 5 — the biggest moat).
- Mounting the camera and recording the demo (Phase 3 / Phase 6).
