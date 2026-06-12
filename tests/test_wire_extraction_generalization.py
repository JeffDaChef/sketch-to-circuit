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

import numpy as np
import pytest

from data_collection.synthetic import _bbox_to_pixels, _series_loop as make_series_loop
from solver.equivalence import circuit_equivalent
from solver.netlist import Netlist
from vision.wire_extraction import extract_netlist

N_SEEDS = 30
MIN_PASSES = 28          # allow a sliver of geometric bad luck, no more


def render(d, boxed):
    """Render a drawing to (image, components) — same recipe as generate_one."""
    from PIL import Image
    import matplotlib.pyplot as plt

    d.draw(show=False)
    fig, ax = d.fig.fig, d.fig.ax
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    pad = 0.12 * max(x1 - x0, y1 - y0)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    fig.canvas.draw()

    rgba = np.asarray(fig.canvas.buffer_rgba())
    height_px, width_px = rgba.shape[0], rgba.shape[1]
    image = np.asarray(Image.fromarray(rgba).convert("RGB"))

    comps = []
    for element, kind, name, value in boxed:
        px = _bbox_to_pixels(element.get_bbox(transform=True),
                             fig, ax, width_px, height_px)
        comps.append({"name": name, "kind": kind, "value": value, "bbox": px})

    # Text labels too — same reason as in generate_one: the pipeline erases
    # detected text before tracing wires.
    to_fraction = fig.transFigure.inverted()
    for i, artist in enumerate(ax.texts):
        ext = artist.get_window_extent(renderer=fig.canvas.get_renderer())
        fx0, fy0 = to_fraction.transform((ext.x0, ext.y0))
        fx1, fy1 = to_fraction.transform((ext.x1, ext.y1))
        xmin, xmax = sorted((fx0 * width_px, fx1 * width_px))
        ymin, ymax = sorted(((1 - fy0) * height_px, (1 - fy1) * height_px))
        comps.append({"name": f"TXT{i + 1}", "kind": "text",
                      "value": artist.get_text(),
                      "bbox": [xmin, ymin, xmax, ymax]})

    plt.close(fig)
    return image, comps


def run_one(seed: int) -> bool:
    rng = random.Random(seed)
    d, truth, boxed = make_series_loop(rng)
    image, comps = render(d, boxed)
    extracted = extract_netlist(image, comps)
    return circuit_equivalent(extracted, truth)


def test_series_loop_generalization():
    """The redesign's definition of done: >=28/30 on the unseen layout."""
    results = [run_one(seed) for seed in range(N_SEEDS)]
    passes = sum(results)
    failing = [s for s, ok in enumerate(results) if not ok]
    print(f"\nseries_loop generalization: {passes}/{N_SEEDS} (failing seeds: {failing})")
    assert passes >= MIN_PASSES, (
        f"Only {passes}/{N_SEEDS} series_loop circuits extracted correctly "
        f"(failing seeds: {failing}). The extractor is layout-overfit again."
    )
