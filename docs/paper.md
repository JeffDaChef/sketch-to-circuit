# Sketch-to-Circuit: Turning Hand-Drawn Schematics into Solved Simulations

A computer-vision and numerical-analysis pipeline that reads a hand-drawn circuit, rebuilds it as a netlist, and solves it from first principles.

Author: Neel, independent project, 2026

---

## Abstract

I built a system that takes a picture of a hand-drawn circuit schematic and turns it into a working, solved simulation. It locates the components, traces the wires into a connection graph, assembles a netlist (the machine-readable description of the circuit), and then solves that circuit with a numerical engine I wrote from scratch. The solver does not wrap an existing tool. It is a Modified Nodal Analysis (MNA) implementation that grew into a four-mode engine, covering steady-state (DC), time-domain (transient), non-linear (real diode physics), and frequency-domain (AC). To show the math is correct rather than just plausible, every result is checked against either a closed-form textbook formula or the industry-standard SPICE simulator. The vision half currently runs on synthetic schematics that I generate and control, where it recovers the correct circuit 200 / 200 times. The honest limitation is that reading real photographs of my own handwriting needs a trained component detector and a hand-labelled dataset, which is the project's next phase. This paper reports the method, the measured results (including where the system breaks), and the steps taken to make every number reproducible.

---

## 1. The problem

A circuit schematic on paper is easy for a person to read and tedious for a person to solve. For a computer it is the other way around. Solving a circuit is fast and exact, but reading a hand-drawn one, with wobbly lines, inconsistent symbols, crossing wires, and handwritten values, is hard. The goal of this project is to bridge that gap end to end:

> point a camera at a sketch on paper and get back the node voltages and branch currents, drawn on top of the sketch.

That one step hides five sub-problems. You have to detect the components, trace the wires, assemble the netlist, solve the circuit, and present the answer. This report focuses on the parts that are built and measured today, which are wire tracing, netlist assembly, and the solver. It is also clear about the part that is not built yet, the trained detector that makes real photographs work.

---

## 2. Why the input is constrained (on purpose)

A recurring decision in the project was to restrict the input on purpose rather than pretend to handle everything. The system assumes a small, stated drawing convention (`docs/drawing_convention.md`): standard two-terminal symbols, components drawn roughly axis-aligned, ground drawn as the usual symbol. I am not hiding the hard case. This is the standard engineering move of defining a tractable sub-problem first, measuring it honestly, and then lifting constraints one at a time. The clearest example of lifting a constraint is the crossing-wires work in section 4.

The convention also makes the data problem solvable. Because I can generate clean schematics that obey the convention, I get unlimited labelled examples (image, ground-truth boxes, and the known-correct netlist) to test against, long before any real photographs exist.

---

## 3. Method

The pipeline has five stages. Stages 2 to 4 are built and measured. Stage 1 (the trained detector) is scaffolded and waiting on data. Stage 5 (live overlay) follows the detector.

### 3.1 Synthetic schematic generator

Using the `schemdraw` library, I generate circuit images in three families (series loops, voltage dividers, parallel banks) at controllable sizes, each paired with a JSON file recording the pixel bounding box of every component and the known-correct solved voltages. This synthetic generator is the project's test backbone. It lets every downstream stage be checked against a known answer, which is what makes the accuracy numbers in section 4 mean something.

### 3.2 Component detection (scaffolded, not yet trained)

The plan is a small YOLO-family object detector fine-tuned on the public CGHD hand-drawn schematic dataset, using a drafter split, where the training and test drawings come from different people, so the score measures how well it generalises to a new hand instead of memorising the training set. The data-preparation and training and evaluation scripts are written and unit-tested. The actual training run and the dataset download are the human-gated next step. This is the single biggest open piece, and the paper does not claim real-photo performance until it is done.

### 3.3 Wire extraction (the hard vision module)

This is the part I am most proud of. Given an image and the component boxes, the extractor does four things.

1. It erases the component boxes, leaving only wire ink.
2. It skeletonises the remaining ink to one-pixel-wide lines.
3. It walks the skeleton into a graph of nodes (junctions) and edges (wire segments).
4. It matches each component terminal to the cut wire-ends that erasure created, using four evidence-ordered rules with no assumptions about layout orientation.

The first version of this module was overfit to the layout. It scored 200/200 on the two templates it was designed against but 0/30 on a new horizontal layout, because its heuristics were tied to the original geometry. I rebuilt it around a real wire graph (`vision/skeleton_graph.py`), and the redesign is what produces the headline accuracy and the ablation in section 4.

A crossing-wires capability lifts the most common simplifying constraint. A plus-shaped crossing fuses two wires into a four-way junction in the skeleton, which would wrongly short two separate nets. Given a `crossover` marker (a class the detector can emit), the extractor splits that junction back into two independent straight wires by pairing the two most-opposite branches. On a divider where one rail crosses another wire, the netlist comes out correct with the marker and shorted without it. That is a clean before and after showing the feature does real work.

### 3.4 The solver: one core, four modes

The electrical engine is a from-scratch Modified Nodal Analysis solver. MNA writes Kirchhoff's current law at every node as a system of linear equations `A x = z` and solves for the unknown node voltages. It is "modified" because voltage sources, which have no admittance, are handled by adding their branch currents as extra unknowns. Everything else in the solver is built on top of this one linear solve using companion models, so there is only ever one piece of linear algebra to trust.

- DC. The base case, with resistors, voltage sources, and current sources.
- Transient (time-domain). Capacitors and inductors are replaced, at each time step, by a resistor in parallel with a current source (the companion model), so every instant is an ordinary DC circuit handed to the same solver. It supports backward-Euler and trapezoidal integration, and time-varying sources such as a sine input.
- Non-linear. Diodes obey the real exponential Shockley equation, solved by Newton-Raphson iteration. Each iteration linearises the diode into the same resistor-plus-source companion form. There is no fixed "0.7 V" shortcut.
- AC (frequency-domain). The same MNA core run with complex numbers, where a capacitor's impedance is 1/(j*omega*C) and an inductor's is j*omega*L. Sweeping frequency gives a transfer function and a Bode plot.

These compose. A half-wave rectifier with a smoothing capacitor exercises transient, non-linear, and time-varying sources at the same time in one simulation. Small-signal AC composes the non-linear and AC modes. It finds a diode's DC operating point, replaces the diode with its small-signal resistance (r_d = n*V_T / I_D, the "26 mV / I" rule), and sweeps frequency, so a diode behaves as a bias-tunable resistor whose filter cutoff moves with the bias current.

### 3.5 Validation

Two independent checks back every solver claim.

- Closed-form anchors. Where textbook formulas exist, results are pinned to them. The RC charging curve matches V(1 - e^(-t/tau)). An RC low-pass hits exactly -3 dB and -45 degrees at its cutoff with a -20 dB/decade rolloff. A series-RLC band-pass peaks at f0 = 1/(2*pi*sqrt(LC)).
- ngspice cross-check. A validation harness builds a SPICE deck from any circuit, runs the industry-standard ngspice simulator, parses its output, and compares it against my solver within tolerance. The deck-building and parsing are unit-tested against canned ngspice output. The one live comparison test auto-skips until ngspice is installed (a one-time administrator step on the family Mac), then runs the hand-checked suite.

The system also exports a complete, runnable SPICE/KiCad `.cir` file, with proper diode `.model` lines, engineering-notation values, and an optional analysis directive, so the toy connects to the real engineering ecosystem.

---

## 4. Results

All numbers below are produced by the test suite and the `metrics/` scripts, and regenerate the same way every time via `./reproduce.sh` (section 6). The suite is 211 tests passing and 1 skipped (the skip is the live ngspice run, waiting on the one-time install).

### 4.1 Extraction accuracy (synthetic)

| Measurement | Result |
|---|---|
| Correct netlist recovered (3 templates and unseen layouts) | 200 / 200 |
| Equivalence check used | graph-isomorphism on the electrical netlist |
| Accuracy vs. circuit size | 100 % up to around 10 to 12 components, then a resolution-driven fall-off |

The accuracy is measured by graph isomorphism. An extraction counts as correct only if its netlist is electrically equivalent to the ground-truth netlist. Visual similarity is not enough.

### 4.2 Ablation: the wire-extractor redesign

| Extractor | Score (5 layouts x 30 seeds) |
|---|---|
| Blob-proximity baseline (frozen) | 60 / 150 (40 %) |
| Skeleton-graph redesign | 150 / 150 (100 %) |

The redesign rescues three layouts the baseline scored 0 % on, with no regression on the two it already handled. The old extractor is kept frozen in `vision/wire_extraction_baseline.py` so this before and after can be reproduced.

### 4.3 Accuracy-vs-difficulty curve

Forcing circuit size shows 100 % accuracy up to around 10 to 12 components across all three families, then an honest fall-off. Building this curve caught a benchmark artifact. The divider's accuracy appeared to collapse to 0 % at eight components, but an ablation (turning text-erasure off gave 12/12) proved the cause was value labels crowding the loop interior and getting erased over the wires. It was not a logic failure. Fixing the label placement restored it. Telling a benchmark artifact apart from a real algorithmic limit is itself a result.

### 4.4 Noise-robustness study

Corrupting clean images at rising severity (80 circuits per level) maps where the current, synthetic-tuned pipeline breaks.

| Corruption | Finding |
|---|---|
| Gaussian blur | Tolerated, accuracy floors around 72 % |
| Gaussian noise | A cliff past sigma about 45 (100 % down to 20 %) |
| Salt-pepper speckle | The weak point. Even 0.4 % roughly halves accuracy |

This directly predicts the Phase-2 preprocessing to-do list: despeckle or denoise, and adaptive (local) thresholding instead of the current global Otsu threshold. I would rather find and name a weakness than hide one.

### 4.5 Solver vs. ground truth (selected)

| Circuit | My solver | Reference |
|---|---|---|
| RC charging at one time constant tau | 3.151 V | 3.16 V (analytic) |
| Silicon diode operating point | 0.693 V, 4.31 mA | self-consistent (KCL = Shockley) |
| LED operating point | 1.805 V, 14.5 mA | matches preset model |
| RC low-pass at cutoff | -3.00 dB, -45 degrees | -3 dB, -45 degrees (closed form) |
| Series-RLC band-pass peak | \|H\| = 1 at f0 | 1/(2*pi*sqrt(LC)) (closed form) |
| Half-wave rectifier + smoothing cap | 4.28 V out, sub-volt ripple | physically expected |

---

## 5. Limitations and honest failure modes

Reporting these is the point. They are where the project earns its credibility. The table
below is the project's error taxonomy. Each row is a way the system fails or could fail, how
often it happens, why it happens, and whether it is still open or already fixed.

| Failure mode | When it shows up | How often | Why it happens | Status |
|---|---|---|---|---|
| Real photographs | Any real photo of a drawing | Untested | The component detector is not trained yet, and the preprocessing is tuned for clean synthetic images | Open, and it is the next step |
| Salt-pepper speckle | Printed or photographed paper with small specks | About 0.4 percent speckle already halves accuracy | The global Otsu threshold reads the specks as ink and corrupts the traced wires | Open, fix is adaptive thresholding |
| Heavy image noise | Low quality or low light photos | Accuracy drops from 100 percent to about 20 percent past sigma around 45 | The single global threshold cannot separate ink from a noisy background | Open, fix is denoise preprocessing |
| Large or dense circuits | More than about 10 to 12 components | Accuracy falls off above roughly 12 components | At the fixed image size the parts shrink below the limit the wire thinner can resolve | Open, fix is higher resolution or tiling |
| Crossing wires with no marker | Two wires cross in a plus shape and the detector did not flag it | Deterministic, those two nets always merge | A four-way junction in the thinned image fuses the two separate nets | Open, needs the detector to emit a crossover box (the splitting logic is built) |
| Inductor circuits in the ngspice check | Validating an RLC circuit against ngspice | Cannot run that check yet | The ngspice harness does not accept the L component yet | Open, the closed-form formulas cover these for now |
| Layout-specific heuristics | Unusual layouts, before the redesign | Was 0 percent on some unseen layouts | The old ground and junction rules assumed vertical parts with ground below | Fixed by the skeleton-graph redesign, now 200/200 plus the unseen layouts |
| Value labels over wires | Dense circuits where a label sits on top of the wiring | Collapsed one divider from 100 percent to 0 percent at eight components | The label ink was erased along with the component and cut the wire underneath | Fixed by moving the labels to the side |

The first four rows are measured directly by the noise and difficulty studies, so see
`noise_robustness.png` and `difficulty_curve.png` for the evidence. The last two rows are bugs I
already found and fixed. They stay in the table because finding and fixing them is part of the
story, and because the same kind of artifact can come back on real photos.

---

## 6. Reproducibility

Every headline number and figure regenerates from a clean checkout with one command:

```
uv pip install -r requirements.txt    # exact pinned versions (==)
./reproduce.sh                         # tests + all metrics + demos + figures
```

Dependencies are pinned to exact versions, every metric takes a fixed `--seed`, and the script runs with `PYTHONHASHSEED=0`, so the results come out the same across machines. The script reports the extraction accuracy, the 40 % to 100 % ablation, the difficulty curve, and the noise-robustness curves, and writes every figure used in this paper.

---

## 7. What comes next

Ranked by impact:

1. Train the detector (CGHD plus a Colab run). This is the keystone. It unlocks real photographs and validates the core claim on input I did not generate.
2. Draw, photograph, and annotate about 35 of my own circuits, then report the domain-adaptation delta: accuracy on strangers' drawings versus accuracy after fine-tuning on my own hand. That before-and-after number is the most distinctive result the project can produce, because no one else can generate it.
3. The Phase-2 preprocessing the noise study already specifies: despeckle plus adaptive thresholding, then re-run the robustness curves to show the improvement.
4. Live camera overlay, built only after the detector works on real photos, and never staged.

---

## Appendix: figures

All figures regenerate via `./reproduce.sh`.

- `extractor_ablation.png`, blob-proximity versus skeleton-graph, 40 % to 100 %.
- `difficulty_curve.png`, accuracy versus component count.
- `noise_robustness.png`, accuracy versus corruption severity (blur, noise, speckle).
- `rc_charging.png`, transient RC charging against the analytic curve.
- `rlc_ringing.png`, underdamped series-RLC step response.
- `rectifier.png`, half-wave rectifier with smoothing capacitor (all three solver modes at once).
- `bode_lowpass.png` and `bode_bandpass.png`, AC transfer functions against closed form.
- `bode_diode_tunable.png`, small-signal AC, a diode filter whose cutoff tracks the bias current.
