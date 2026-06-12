# Plan: Redesign the wire-extraction module (skeleton-graph approach)

## Context

Wire extraction (`vision/wire_extraction.py`) is the project's hardest module. The current
version scores 200/200 on the two existing synthetic templates, but when we tried to add a
third template (horizontal "series-loop"), it scored **0/30**, and every patch attempt broke
something else (whack-a-mole, measured live: 48% → 84% → 48%). We reverted to the known-good
state and recorded the finding in ROADMAP.md. This plan is the principled redesign so the
extractor generalizes to *any* reasonable layout — which is required before real hand-drawn
photos (Phase 2) ever work.

---

## 1. What is wrong with the current extractor

The root cause is one design decision; the rest are symptoms:

**Root cause — connectivity by *blob proximity*, not by *tracing*.** The extractor erases
component bounding boxes, splits the leftover wires into connected "blobs", and asks "which
blob is *near* this terminal?" Distance-to-nearest-pixel carries no information about *where
along a wire* the connection happens, so every layout quirk needs its own patch:

- **The ground x-overlap hack.** Erasing the voltage source's (large) bbox severs the bottom
  rail, so V1's negative terminal finds no wire nearby. Patch: "if a dangling terminal is in
  the same x-column as a ground symbol, call it ground." Works only when the source is stacked
  vertically above ground — wrongly grounds *both* source terminals in a horizontal layout
  (measured: this is exactly why series-loop scored 0/30).
- **Leads attach far outside the bbox.** The schemdraw source symbol's bbox includes its label,
  so its actual wire attachment points are ~50–90 px outside the box (measured: 49 px top,
  91 px bottom on seed 1). Close-range probing at the bbox edge can't see them.
- **Coupled patches.** Adding "follow the lead outward along its axis" fixed the source but
  broke junction-free resistor chains; the close-probe radius, the ground hack, and the
  junction rule each compensate for a different symptom of the same root cause, so they
  can't be fixed independently.
- **Topology is an accident of erasure.** What counts as "one net" is whatever the bbox
  erasure happens to leave connected. The erasure rectangles (which include label text in
  synthetic data) decide the circuit, instead of the wires deciding the circuit.

## 2. Files to touch / files to avoid

**Touch:**
| File | Change |
|---|---|
| `vision/skeleton_graph.py` | **NEW** — build a proper graph from a skeleton mask (the core of the redesign) |
| `vision/wire_extraction.py` | Rewrite the *internals* of `extract_netlist()`; keep its public signature exactly (`(image, components, *, match_radius, debug)` → `Netlist`) so all callers/tests run unchanged |
| `tests/test_skeleton_graph.py` | **NEW** — hand-checkable unit tests on tiny drawn-by-code masks |
| `tests/test_wire_extraction.py` | Keep as-is initially (it is the regression gate); extend with the new-template oracle test at the end |
| `data_collection/synthetic.py` | **Only at the very end**, add the `series_loop` template once extraction passes it — never while debugging the extractor |
| `EXPLAINED.md` / `ROADMAP.md` | Update after it works |

**Do NOT touch:**
- `solver/netlist.py`, `solver/mna.py`, `solver/equivalence.py` — the proven Phase-0 core and
  the measuring stick. If the redesign "needs" a solver change, the redesign is wrong.
- `metrics/extraction_accuracy.py` — the scoreboard must stay fixed so before/after numbers
  are comparable (this is also ablation-study material for the writeup).
- `data_collection/cghd_prep.py`, `training/*` — unrelated (Phase 1).

## 3. The proposed new algorithm, step by step

Replace "blobs + proximity + fallbacks" with a **skeleton graph + endpoint snapping**:

1. **Binarize** (unchanged — keep `_to_ink_mask`, it works).
2. **Erase component bodies** (unchanged mechanically — keep `_erase_components`). Erasure is
   now only for *removing symbol strokes*, NOT for defining topology.
3. **Skeletonize** the remaining wire ink to 1-px centrelines (unchanged).
4. **NEW — build the skeleton graph** (`vision/skeleton_graph.py`):
   - Classify every skeleton pixel by its number of skeleton neighbours (8-connectivity):
     degree 1 = **endpoint**, degree 2 = path interior, degree ≥3 = **branch point**.
   - Walk paths between endpoints/branch points to produce a `networkx` graph:
     nodes = endpoints + branch points (with pixel coords), edges = traced wire paths
     (with length + the pixel list).
   - **Prune spurs on the graph, not the pixels**: drop endpoint-edges shorter than a few px
     (skeletonization whiskers), then re-merge degree-2 nodes. Graph-level pruning can't
     accidentally delete a whole real wire the way pixel peeling can.
5. **Infer terminals from bbox geometry, orientation-aware** (small fix to the existing
   `_corner_probes` idea): a tall component's pins are at top/bottom mid-edges; a wide one's at
   left/right. (The current code has this bug: it always probes top/bottom.)
6. **NEW — snap each terminal to a skeleton-graph *endpoint*, not to any nearby pixel.**
   Key insight: erasing a component's body *cuts* the wires it touched, and every cut creates
   an endpoint **exactly where a wire entered that component**. So the correct match for a
   terminal is a nearby *endpoint*, and endpoints are sparse — unlike "any wire pixel", which
   is everywhere. Matching rule, in order:
   a. Endpoints within `match_radius` of the terminal point → take them all (handles the
      parallel-bank "two segments at one pin" case).
   b. Otherwise the nearest endpoint within a generous cap (~15% of image diagonal) whose
      direction from the terminal is roughly along the component's lead axis (±45° cone) —
      this finds the source's far-attached leads *without* being able to grab a passing rail,
      because a rail passing nearby has no endpoint there (its midpoints don't qualify).
   c. Otherwise floating → its own net (honest behaviour; shows up in the metric).
7. **Junctions**: a junction dot's erasure also cuts wires, leaving 2+ endpoints around the
   dot. Merge every endpoint within `match_radius` of the junction's bbox into one net. (This
   replaces the "top-half/bottom-half of the component" rule — no orientation assumption.)
   Ground symbols: same snapping as a 1-pin terminal; its net becomes `"0"`.
8. **Union-find** over {terminals ∪ graph connected-components}, exactly as today, then build
   the `Netlist` with placeholder values (unchanged).

Why this generalizes: every rule above is stated in terms of *what erasure does to wires*
(cuts → endpoints → attachment points). None of it references "vertical", "above the
ground", or any template's layout. There are zero special-case fallbacks left.

## 4. What tests to write first (TDD order)

1. **`tests/test_skeleton_graph.py` — pure unit tests, written before the graph code.**
   Draw tiny masks with numpy (no schemdraw): a straight line (→ 2 endpoints, 1 edge), an
   L-bend (same), a T (→ 3 endpoints + 1 branch point, 3 edges), a plus (+) (→ 4 + 1), a line
   with a 2-px whisker (→ whisker pruned). Every expected count is hand-checkable.
2. **Terminal-snapping unit tests**: a small mask with one wire cut + a fake bbox; assert the
   terminal snaps to the cut endpoint; assert a *passing* wire with no endpoint nearby is NOT
   grabbed by the cone rule (the exact failure mode of my axial-probe attempt).
3. **Regression gate (already exists)**: the 20-seed oracle test + `metrics` run — the rewrite
   must hold **200/200 on seeds 0 and 7** for the two existing templates. Non-negotiable.
4. **Generalization target (the point of all this)**: a `series_loop` fixture (source on the
   left, resistor chain running horizontally, wire looping back to ground) defined in the test
   file first — extraction must score ≥ 28/30 across seeds. This is the test that currently
   fails 0/30; it is the definition of done.

## 5. What could go wrong (and the mitigations)

- **Endpoint ambiguity in dense regions** — several components' cuts cluster near a rail
  (e.g. R4's bottom pin near the ground symbol). Mitigation: the cone rule (b) only fires
  when (a) found nothing, and prefers direction-aligned endpoints; the metric harness will
  tell us immediately if it mis-snaps (per-template breakdown).
- **Junction erasure merging too much** — a junction's radius catching an unrelated endpoint.
  Mitigation: junction radius stays small (`match_radius`, ~10 px); junction dots in the
  drawing convention are *always* drawn exactly at the meeting point.
- **Spur pruning deleting a real short wire** — mitigated by pruning at the graph level with
  a tiny threshold (≤ 4 px) instead of pixel peeling.
- **The synthetic bbox-includes-label quirk** — source bboxes are inflated by their text
  label, pushing cuts far from pins. The cone rule handles it; worth noting in EXPLAINED as a
  difference vs. real data (where text is a separate detection class).
- **Regression risk on the working templates** — mitigated by the hard gate (step 4.3), small
  commits, and `git` (we already proved revert works cleanly today).
- **Cost control** — per your instruction: NO Sonnet for this module (too hard / too coupled).
  Opus (me) implements it directly in small verified steps; no sub-agents unless something is
  genuinely mechanical. The TDD order means each step is cheap to check.

## 6. Exact first implementation steps after approval

1. Write `tests/test_skeleton_graph.py` (the tiny-mask unit tests) — they fail, since the
   module doesn't exist.
2. Write `vision/skeleton_graph.py` (`build_skeleton_graph(mask) -> nx.Graph` with pruning)
   until those tests pass.
3. Add the `series_loop` test fixture (schemdraw template defined inside the test file, NOT
   in `synthetic.py`) and the 30-seed generalization test — it fails (0/30 today).
4. Rewrite `extract_netlist()` internals to use the graph + endpoint snapping (steps 5–8 of
   the algorithm), keeping the public signature. Run after each sub-step:
   `pytest tests/test_wire_extraction.py` + `metrics --count 200 --seed 0` (must stay 200/200)
   and the new series_loop test (drive it from 0/30 → ≥28/30).
5. Only when both gates pass: move `series_loop` into `data_collection/synthetic.py` as an
   official third template, re-run the full metric (difficulty curve now meaningful), update
   EXPLAINED.md + ROADMAP.md, commit.

## Verification

- `.venv/bin/python -m pytest` — full suite green (91 existing + new tests).
- `.venv/bin/python metrics/extraction_accuracy.py --count 200 --seed 0` (and seed 7) —
  **200/200 on existing templates** before and after; per-template table shows the new
  `series_loop` ≥ ~93%.
- The before/after of this redesign (0/30 → ≥28/30 on the new layout, same harness) is
  itself ablation-style evidence for the project writeup.
