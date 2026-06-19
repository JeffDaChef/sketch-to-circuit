# Sketch-to-Circuit: Turning Hand-Drawn Schematics into Solved Simulations

**A computer-vision and numerical-analysis pipeline that reads a hand-drawn circuit, rebuilds it as a netlist, and solves it from first principles.**

*Author: Neel — independent project, 2026*

---

## Abstract

I built a system that takes a picture of a hand-drawn circuit schematic and turns it into a
working, solved simulation: it locates the components, traces the wires into a connection
graph, assembles a *netlist* (the machine-readable description of the circuit), and then
solves that circuit with a numerical engine I wrote from scratch. The solver is not a wrapper
around an existing tool — it is a Modified Nodal Analysis (MNA) implementation that grew into
a **four-mode engine**: steady-state (DC), time-domain (transient), non-linear (real diode
physics), and frequency-domain (AC). To prove the math is correct rather than merely
plausible, every result is checked against either a closed-form textbook formula or the
industry-standard SPICE simulator. The vision half currently runs on synthetic schematics that
I generate and control, where it recovers the correct circuit **200 / 200** times; the honest
limitation is that reading *real* photographs of *my own* handwriting requires a trained
component detector and a hand-labelled dataset, which is the project's next phase. This paper
reports the method, the measured results — including where the system breaks — and the steps
taken to make every number reproducible.

---

## 1. The problem

A circuit schematic on paper is trivial for a person to read and tedious for a person to solve.
The reverse is true for a computer: solving a circuit is fast and exact, but *reading* a
hand-drawn one — with wobbly lines, inconsistent symbols, crossing wires, and handwritten
values — is genuinely hard. The goal of this project is to bridge that gap end to end:

> point a camera at a sketch on paper → get back the node voltages and branch currents,
> overlaid on the drawing.

That single arrow hides five sub-problems: **detect** the components, **trace** the wires,
**assemble** the netlist, **solve** the circuit, and **present** the answer. This report
focuses on the parts that are built and measured today — wire tracing, netlist assembly, and
the solver — and is explicit about the part that is not yet built: the trained detector that
makes real photographs work.

---

## 2. Why the input is constrained (on purpose)

A recurring decision in the project was to *deliberately restrict* the input rather than
pretend to handle everything. The system assumes a small, stated drawing convention
(`docs/drawing_convention.md`): standard two-terminal symbols, components drawn roughly
axis-aligned, ground drawn as the usual symbol. This is not hiding the hard case — it is the
standard engineering move of defining a tractable sub-problem first, measuring it honestly, and
then *lifting* constraints one at a time. The clearest example of lifting a constraint is the
crossing-wires work in §4.

The convention also makes the data problem solvable: because I can *generate* clean schematics
that obey the convention, I get unlimited labelled examples (image + ground-truth boxes + the
known-correct netlist) to test against, long before any real photographs exist.

---

## 3. Method

The pipeline is five stages. Stages 2–4 are built and measured; stage 1 (the trained detector)
is scaffolded and awaiting data; stage 5 (live overlay) follows the detector.

### 3.1 Synthetic schematic generator

Using the `schemdraw` library, I generate circuit images in three families (series loops,
voltage dividers, parallel banks) at controllable sizes, each paired with a JSON file
recording the pixel bounding box of every component *and* the known-correct solved voltages.
This synthetic generator is the project's test backbone: it lets every downstream stage be
checked against a *known answer*, which is what makes the accuracy numbers in §4 meaningful.

### 3.2 Component detection (scaffolded, not yet trained)

The plan is a small YOLO-family object detector fine-tuned on the public CGHD hand-drawn
schematic dataset, using a *drafter split* (training and test drawings come from different
people, so the score measures generalisation to a new hand, not memorisation). The
data-preparation and training/evaluation scripts are written and unit-tested; the actual
training run and the dataset download are the human-gated next step. **This is the single
biggest open piece, and the paper does not claim real-photo performance until it is done.**

### 3.3 Wire extraction (the hard vision module)

This is the part I am most proud of. Given an image and the component boxes, the extractor:

1. **erases** the component boxes, leaving only wire ink;
2. **skeletonises** the remaining ink to one-pixel-wide lines;
3. **walks** the skeleton into a graph of nodes (junctions) and edges (wire segments);
4. **matches** each component terminal to the cut wire-ends that erasure created, using four
   evidence-ordered rules with *no* assumptions about layout orientation.

The first version of this module was layout-overfit — it scored 200/200 on the two templates
it was designed against but **0/30** on a new horizontal layout, because its heuristics were
coupled to the original geometry. I rebuilt it around a real wire graph
(`vision/skeleton_graph.py`), and the redesign is what produces the headline accuracy and the
ablation in §4.

A **crossing-wires** capability lifts the most common simplifying constraint: a `+` crossing
fuses two wires into a four-way junction in the skeleton, which would wrongly short two
separate nets. Given a `crossover` marker (a class the detector can emit), the extractor splits
that junction back into two independent straight wires by pairing the two most-opposite
branches. On a divider where one rail crosses another wire, the netlist comes out correct *with*
the marker and shorted *without* it — a clean before/after demonstration that the feature does
real work.

### 3.4 The solver: one core, four modes

The electrical engine is a from-scratch **Modified Nodal Analysis** solver. MNA writes
Kirchhoff's current law at every node as a system of linear equations `A x = z` and solves for
the unknown node voltages; it is "modified" because voltage sources, which have no admittance,
are handled by adding their branch currents as extra unknowns. Everything else in the solver is
built *on top of this one linear solve* using companion models, so there is only ever one piece
of linear algebra to trust:

- **DC** — the base case: resistors, voltage sources, current sources.
- **Transient (time-domain)** — capacitors and inductors are replaced, *at each time step*, by
  a resistor in parallel with a current source (the *companion model*), so every instant is an
  ordinary DC circuit handed to the same solver. Supports backward-Euler and trapezoidal
  integration, and time-varying sources (e.g. a sine input).
- **Non-linear** — diodes obey the real exponential Shockley equation, solved by
  Newton-Raphson iteration; each iteration linearises the diode into the same
  resistor-plus-source companion form. No fixed "0.7 V" shortcut.
- **AC (frequency-domain)** — the *same* MNA core run with complex numbers, where a capacitor's
  impedance is `1/(jωC)` and an inductor's is `jωL`. Sweeping frequency yields a transfer
  function and a Bode plot.

These compose: a **half-wave rectifier with a smoothing capacitor** exercises transient,
non-linear, and time-varying sources *simultaneously* in one simulation. And **small-signal
AC** composes the non-linear and AC modes — it finds a diode's DC operating point, replaces the
diode with its small-signal resistance (`r_d = n·V_T / I_D`, the "26 mV / I" rule), and sweeps
frequency, so a diode behaves as a *bias-tunable resistor* whose filter cutoff moves with the
bias current.

### 3.5 Trust before wow: validation

Two independent checks back every solver claim:

- **Closed-form anchors.** Where textbook formulas exist, results are pinned to them: the RC
  charging curve matches `V(1 − e^(−t/τ))`; an RC low-pass hits exactly −3 dB and −45° at its
  cutoff with a −20 dB/decade rolloff; a series-RLC band-pass peaks at `f₀ = 1/(2π√(LC))`.
- **ngspice cross-check.** A validation harness builds a SPICE deck from any circuit, runs the
  industry-standard `ngspice` simulator, parses its output, and diffs it against my solver
  within tolerance. The deck-building and parsing are unit-tested against canned ngspice output;
  the one *live* comparison test auto-skips until ngspice is installed (a one-time
  administrator step on the family Mac), then runs the hand-checked suite.

The system also **exports a complete, runnable SPICE/KiCad `.cir` file** — proper diode
`.model` lines, engineering-notation values, optional analysis directive — so the toy connects
to the real engineering ecosystem.

---

## 4. Results

All numbers below are produced by the test suite and the `metrics/` scripts, and regenerate
deterministically via `./reproduce.sh` (§6). The suite is **211 tests passing, 1 skipped** (the
skip is the live ngspice run, pending the one-time install).

### 4.1 Extraction accuracy (synthetic)

| Measurement | Result |
|---|---|
| Correct netlist recovered (3 templates + unseen layouts) | **200 / 200** |
| Equivalence check used | graph-isomorphism on the electrical netlist |
| Accuracy vs. circuit size | **100 %** up to ~10–12 components, then a resolution-driven fall-off |

The accuracy is measured by *graph isomorphism*: an extraction counts as correct only if its
netlist is electrically equivalent to the ground-truth netlist, not merely visually similar.

### 4.2 Ablation: the wire-extractor redesign

| Extractor | Score (5 layouts × 30 seeds) |
|---|---|
| Blob-proximity baseline (frozen) | **60 / 150 (40 %)** |
| Skeleton-graph redesign | **150 / 150 (100 %)** |

The redesign rescues three layouts the baseline scored **0 %** on, with **no regression** on
the two it already handled. The old extractor is kept frozen in
`vision/wire_extraction_baseline.py` precisely so this before/after can be reproduced.

### 4.3 Accuracy-vs-difficulty curve

Forcing circuit size shows **100 % accuracy up to ~10–12 components** across all three
families, then an honest fall-off. Building this curve caught a *benchmark artifact*: the
divider's accuracy appeared to collapse to 0 % at eight components, but an ablation
(text-erasure off → 12/12) proved the cause was value *labels* crowding the loop interior and
being erased over the wires, not a logic failure. Fixing the label placement restored it.
Distinguishing a benchmark artifact from a real algorithmic limit is itself a result.

### 4.4 Noise-robustness study

Corrupting clean images at rising severity (80 circuits per level) maps where the current,
synthetic-tuned pipeline breaks:

| Corruption | Finding |
|---|---|
| Gaussian blur | Tolerated — accuracy floors around **72 %** |
| Gaussian noise | A cliff past σ ≈ 45 (100 % → 20 %) |
| Salt-pepper speckle | **The weak point — even 0.4 % roughly halves accuracy** |

This directly predicts the Phase-2 preprocessing to-do list: despeckle/denoise and *adaptive*
(local) thresholding instead of the current global Otsu threshold. A weakness found and named
is worth more than one hidden.

### 4.5 Solver vs. ground truth (selected)

| Circuit | My solver | Reference |
|---|---|---|
| RC charging at one time constant τ | 3.151 V | 3.16 V (analytic) |
| Silicon diode operating point | 0.693 V, 4.31 mA | self-consistent (KCL = Shockley) |
| LED operating point | 1.805 V, 14.5 mA | matches preset model |
| RC low-pass at cutoff | −3.00 dB, −45° | −3 dB, −45° (closed form) |
| Series-RLC band-pass peak | \|H\| = 1 at f₀ | 1/(2π√(LC)) (closed form) |
| Half-wave rectifier + smoothing cap | 4.28 V out, sub-volt ripple | physically expected |

---

## 5. Limitations and honest failure modes

Reporting these *is* the point — they are the credibility centre of the project.

1. **No real-photo results yet.** Everything above is on synthetic images I generate. The
   trained detector (§3.2) does not exist yet, so the headline "reads hand-drawn circuits"
   claim is validated only on clean, convention-following input. This is the honest ceiling on
   today's results.
2. **Speckle and lighting break the current preprocessing** (§4.4). The global Otsu threshold
   will struggle with photographed paper under uneven light; this is measured, not guessed.
3. **Some early heuristics assumed the synthetic layout.** The original ground-detection and
   junction rules baked in "components are vertical, ground is directly below." The
   skeleton-graph redesign removed those assumptions, but real photos will surface more.
4. **The ngspice cross-check does not yet cover inductor circuits**, and the live comparison is
   gated on a one-time install. Closed-form anchors cover the inductor cases in the meantime.
5. **Crossing wires need the detector.** The crossover-splitting logic works, but it depends on
   the detector emitting a `crossover` marker on real photographs — which loops back to (1).

---

## 6. Reproducibility

Every headline number and figure regenerates from a clean checkout with one command:

```
uv pip install -r requirements.txt    # exact pinned versions (==)
./reproduce.sh                         # tests + all metrics + demos + figures
```

Dependencies are pinned to exact versions, every metric takes a fixed `--seed`, and the script
runs with `PYTHONHASHSEED=0`, so results are deterministic across machines. The script reports
the extraction accuracy, the 40 % → 100 % ablation, the difficulty curve, and the
noise-robustness curves, and writes every figure used in this paper.

---

## 7. What comes next

Ranked by impact:

1. **Train the detector** (CGHD + a Colab run). This is the keystone — it unlocks real
   photographs and validates the core claim on input I did not generate.
2. **Draw, photograph, and annotate ~35 of my own circuits**, then report the
   *domain-adaptation delta*: accuracy on strangers' drawings versus accuracy after
   fine-tuning on my own hand. That before/after number is the most distinctive result the
   project can produce, because no one else can generate it.
3. **Phase-2 preprocessing** the noise study already specifies: despeckle + adaptive
   thresholding, then re-run the robustness curves to show the improvement.
4. **Live camera overlay**, built only *after* the detector works on real photos — never staged.

---

## Appendix: figures

All figures regenerate via `./reproduce.sh`.

- `extractor_ablation.png` — blob-proximity vs. skeleton-graph, 40 % → 100 %.
- `difficulty_curve.png` — accuracy vs. component count.
- `noise_robustness.png` — accuracy vs. corruption severity (blur / noise / speckle).
- `rc_charging.png` — transient RC charging vs. the analytic curve.
- `rlc_ringing.png` — underdamped series-RLC step response.
- `rectifier.png` — half-wave rectifier with smoothing capacitor (all three solver modes at once).
- `bode_lowpass.png`, `bode_bandpass.png` — AC transfer functions vs. closed form.
- `bode_diode_tunable.png` — small-signal AC: a diode filter whose cutoff tracks bias current.
