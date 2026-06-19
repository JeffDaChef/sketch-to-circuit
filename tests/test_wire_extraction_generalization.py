"""Generalization test: a circuit LAYOUT the extractor was never tuned on.

HISTORY: the first (baseline) extractor scored 200/200 on the two original
synthetic templates and 0/30 on this layout — proof it had memorised layouts
rather than learned wires. The skeleton-graph redesign passes both, and the
template has since been promoted into data_collection/synthetic.py as the
official third template (we import it from there). This test remains as the
regression guard for the generalization property itself.

The layout: a voltage source standing on the left, a chain of HORIZONTAL
resistors marching right along the top, and a wire looping back along the
bottom to a ground at the source's foot. Exercises everything the original
templates don't: horizontal components, terminals at left/right faces, and a
source-lead/resistor-lead meeting with no junction dot.
"""

import random

import pytest

from data_collection.extra_layouts import make_corner_chain, make_mirrored_loop, render
from data_collection.synthetic import _series_loop as make_series_loop
from solver.equivalence import circuit_equivalent
from vision.wire_extraction import extract_netlist

N_SEEDS = 30
MIN_PASSES = 28


LAYOUTS = {
    "series_loop": make_series_loop,
    "corner_chain": make_corner_chain,
    "mirrored_loop": make_mirrored_loop,
}


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_generalization(layout):
    """Unseen-layout guard: >=28/30 per layout or the extractor is overfit."""
    maker = LAYOUTS[layout]
    results = []
    for seed in range(N_SEEDS):
        rng = random.Random(seed)
        d, truth, boxed = maker(rng)
        image, comps = render(d, boxed)
        results.append(circuit_equivalent(extract_netlist(image, comps), truth))
    passes = sum(results)
    failing = [s for s, ok in enumerate(results) if not ok]
    print(f"\n{layout} generalization: {passes}/{N_SEEDS} (failing seeds: {failing})")
    assert passes >= MIN_PASSES, (
        f"Only {passes}/{N_SEEDS} {layout} circuits extracted correctly "
        f"(failing seeds: {failing}). The extractor is layout-overfit again."
    )
