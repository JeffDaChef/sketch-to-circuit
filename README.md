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
| `data_collection/` | Synthetic schematic generator; later, dataset prep scripts |
| `training/` | YOLO detector training scripts (Phase 1) |
| `vision/` | Camera capture, preprocessing, wire extraction (Phases 2–3) |
| `solver/` | Netlist data structures + the MNA circuit solver |
| `ui/` | Live overlay UI (Phase 3) |
| `tests/` | Unit tests for every module |
| `docs/` | Specs and documentation |

## Status

- [x] Project brief & plan
- [x] **Phase 0:** synthetic pipeline — netlist + MNA solver + generator + end-to-end demo (22 tests passing)
- [~] **Phase 1:** YOLO detector on CGHD — data-prep + training/eval scripts written & tested; awaiting dataset download + Colab training run
- [~] **Phase 2:** wire extraction — recovers the netlist from an image + boxes; 60/60 synthetic circuits correct (graph-isomorphism checked). Two template-specific heuristics flagged for real-photo hardening (see ROADMAP.md). 84 tests passing.
- [ ] Phase 3: live camera + overlay
- [ ] Phase 4: ngspice validation, extraction metric, LLM explain mode, what-if
- [ ] Phase 5: domain adaptation on my own drawings
- [ ] Phase 6: polish + demo video
