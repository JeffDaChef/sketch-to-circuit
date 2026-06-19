# Sketch-to-Circuit

A computer-vision system that turns a hand-drawn circuit schematic on paper into a live, solved circuit simulation. A downward-facing camera watches the drawing. The software finds the components, traces the wires, builds a netlist, solves the circuit with a Modified Nodal Analysis (MNA) engine I wrote from scratch, and draws the computed node voltages and branch currents back onto the live camera image.

Technical paper and write-up: [docs/paper.md](docs/paper.md). Project website: https://jeffdachef.github.io/sketch-to-circuit/

Full project spec: [`sketch_to_circuit_brief.md`](sketch_to_circuit_brief.md).
Plain-English explanation of all the code: [`EXPLAINED.md`](EXPLAINED.md).
Input drawing rules: [`docs/drawing_convention.md`](docs/drawing_convention.md).

## Project structure

| Folder | What lives here |
|---|---|
| `data_collection/` | Synthetic schematic generator and alternative layouts, and later the dataset prep |
| `training/` | YOLO detector training scripts (Phase 1) |
| `vision/` | Camera capture, preprocessing, and wire extraction (Phases 2 and 3) |
| `solver/` | Netlist and MNA solver (DC, transient, nonlinear), ngspice validation, SPICE export |
| `metrics/` | Accuracy, noise-robustness study, extractor ablation |
| `ui/` | Live overlay UI (Phase 3) |
| `tests/` | Unit tests for every module |
| `docs/` | Specs and documentation |

## Reproducing the results

Every headline number and figure regenerates from a clean checkout with one command.

```bash
uv pip install -r requirements.txt   # exact pinned versions
./reproduce.sh                        # tests + metrics + demos + figures
```

Dependencies are pinned (`==`) and every metric takes a fixed `--seed`, so the results come out the same every time. The script reports extraction accuracy (200/200), the blob-proximity to skeleton-graph ablation (40% to 100%), the noise-robustness curves, and the solver demos. It writes `rc_charging.png`, `rectifier.png`, `noise_robustness.png`, and `extractor_ablation.png`.

## Status

(211 tests passing, 1 skipped. The skip is the live ngspice run, which is waiting on a one-time install.)

- [x] Project brief and plan
- [x] Phase 0: synthetic pipeline. Netlist, MNA solver, generator, and an end-to-end demo.
- [x] Solver, expanded. The MNA core is now a four-mode numerical engine. It does DC, transient (RC/RL/RLC via companion models, backward-Euler or trapezoidal, time-varying sources), nonlinear (real diodes and LEDs via Newton-Raphson), and AC or frequency-domain (complex-MNA phasor analysis with Bode plots). It also has an ngspice validation harness and SPICE/KiCad export.
- [~] Phase 1: YOLO detector on CGHD. The data-prep and training/eval scripts are written and tested. It is waiting on the dataset download and a Colab training run.
- [x] Phase 2: wire extraction. It recovers the netlist from an image plus the component boxes, and gets 200/200 synthetic circuits correct across 3 templates and some unseen layouts (checked by graph isomorphism). A redesign lifted it from the blob-proximity baseline (ablation: 40% to 100%).
- [x] Rigor: end-to-end accuracy metric, noise-robustness study, extractor ablation, pinned deps, and a one-command `reproduce.sh`.
- [ ] Phase 3: live camera and overlay
- [ ] Phase 4: LLM explain mode and what-if (the ngspice validation is already built)
- [ ] Phase 5: domain adaptation on my own drawings
- [ ] Phase 6: polish and demo video

## License

MIT. See [LICENSE](LICENSE). You are free to use, copy, and build on this, just keep the
copyright notice.
