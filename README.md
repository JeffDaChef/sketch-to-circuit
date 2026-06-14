# Sketch-to-Circuit

A computer-vision system that turns a **hand-drawn circuit schematic on paper** into a
**live, solved circuit simulation**. A downward-facing camera watches the drawing; the
software detects components, traces wires, builds a netlist, solves the circuit with a
from-scratch Modified Nodal Analysis (MNA) engine, and overlays the computed node
voltages and branch currents back onto the live camera image.

Full project spec: [`sketch_to_circuit_brief.md`](sketch_to_circuit_brief.md)
Plain-English explanation of all the code: [`EXPLAINED.md`](EXPLAINED.md)
Input drawing rules: [`docs/drawing_convention.md`](docs/drawing_convention.md)

## Project structure

| Folder | What lives here |
|---|---|
| `data_collection/` | Synthetic schematic generator + alternative layouts; later, dataset prep |
| `training/` | YOLO detector training scripts (Phase 1) |
| `vision/` | Camera capture, preprocessing, wire extraction (Phases 2–3) |
| `solver/` | Netlist + MNA solver (DC, transient, nonlinear), ngspice validation, SPICE export |
| `metrics/` | Accuracy, noise-robustness study, extractor ablation |
| `ui/` | Live overlay UI (Phase 3) |
| `tests/` | Unit tests for every module |
| `docs/` | Specs and documentation |

## Reproducing the results

Every headline number and figure regenerates from a clean checkout with one command:

```bash
uv pip install -r requirements.txt   # exact pinned versions
./reproduce.sh                        # tests + metrics + demos + figures
```

Dependencies are pinned (`==`) and every metric takes a fixed `--seed`, so results
are deterministic. The script reports extraction accuracy (200/200), the
blob-proximity → skeleton-graph ablation (40% → 100%), the noise-robustness curves,
and the solver demos; it writes `rc_charging.png`, `rectifier.png`,
`noise_robustness.png`, and `extractor_ablation.png`.

## Status

_(166 tests passing, 1 skipped — the live ngspice run, pending a one-time install.)_

- [x] Project brief & plan
- [x] **Phase 0:** synthetic pipeline — netlist + MNA solver + generator + end-to-end demo
- [x] **Solver, expanded:** the MNA core is now a real numerical engine — **DC**, **transient**
  (RC/RL/RLC via companion models, backward-Euler *or* trapezoidal, time-varying sources), and
  **nonlinear** (real diodes/LEDs via Newton-Raphson). Plus an **ngspice validation** harness and
  **SPICE/KiCad export**.
- [~] **Phase 1:** YOLO detector on CGHD — data-prep + training/eval scripts written & tested;
  awaiting dataset download + Colab training run
- [x] **Phase 2:** wire extraction — recovers the netlist from an image + boxes; **200/200**
  synthetic circuits correct across 3 templates + unseen layouts (graph-isomorphism checked). A
  redesign lifted it from the blob-proximity baseline (ablation: 40% → 100%).
- [x] **Rigor:** end-to-end accuracy metric, noise-robustness study, extractor ablation, pinned
  deps + one-command `reproduce.sh`
- [ ] Phase 3: live camera + overlay
- [ ] Phase 4: LLM explain mode, what-if (ngspice validation ✅ already built)
- [ ] Phase 5: domain adaptation on my own drawings
- [ ] Phase 6: polish + demo video
