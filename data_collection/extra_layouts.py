"""Alternative circuit layouts + a shared renderer, beyond the core templates.

The three "official" templates live in synthetic.py. These are *additional*
layouts that stress the wire extractor in ways the core templates don't —
corner-turning chains and a mirrored loop — first written to prove the
skeleton-graph extractor generalizes to shapes it was never tuned on.

They started life inside the generalization test, but they're useful beyond
testing (e.g. the baseline-vs-redesign ablation in metrics/), so they live here
where both the test and the metrics scripts can import them. Each maker has the
same signature as a synthetic template — ``maker(rng) -> (drawing, netlist,
boxed)`` — and ``render(drawing, boxed)`` turns one into ``(image, components)``
using the exact recipe as ``generate_one``.
"""

from __future__ import annotations

import random

import numpy as np

from data_collection.synthetic import RESISTOR_VALUES, SOURCE_VOLTAGES, _bbox_to_pixels
from solver.netlist import Netlist


def render(d, boxed):
    """Render a schemdraw drawing to (image, components) — same recipe as generate_one."""
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


def make_corner_chain(rng: random.Random):
    """A chain that TURNS A CORNER: resistors right along the top, then DOWN.

    Mixes horizontal and vertical components in one chain — found 23/30 broken
    during adversarial poking (component-scaled caps fixed it).
    """
    import schemdraw
    import schemdraw.elements as elm

    k_across, k_down = rng.randint(1, 2), rng.randint(1, 2)
    volts = rng.choice(SOURCE_VOLTAGES)
    netlist = Netlist()
    boxed = []
    d = schemdraw.Drawing(show=False)
    src = elm.SourceV().up().label(f"{volts}V")
    d += src
    netlist.add("V", "V1", volts, "n1", "0")
    boxed.append((src, "voltage source", "V1", f"{volts}V"))
    prev, idx = "n1", 1
    for _ in range(k_across):
        val = rng.choice(RESISTOR_VALUES)
        r = elm.Resistor().right().label(val)
        d += r
        nxt = f"n{idx + 1}"
        netlist.add("R", f"R{idx}", val, prev, nxt)
        boxed.append((r, "resistor", f"R{idx}", val))
        prev, idx = nxt, idx + 1
    for j in range(k_down):
        val = rng.choice(RESISTOR_VALUES)
        r = elm.Resistor().down().label(val)
        d += r
        nxt = "0" if j == k_down - 1 else f"n{idx + 1}"
        netlist.add("R", f"R{idx}", val, prev, nxt)
        boxed.append((r, "resistor", f"R{idx}", val))
        prev, idx = nxt, idx + 1
    d += elm.Line().tox(src.start)          # rail LEFT along the chain's bottom
    d += elm.Line().toy(src.start)          # then UP to the source's foot
    gnd = elm.Ground().at(src.start)
    d += gnd
    boxed.append((gnd, "ground", "GND", None))
    return d, netlist, boxed


def make_mirrored_loop(rng: random.Random):
    """series_loop MIRRORED: source on the RIGHT, resistors marching LEFT.

    Found 14/30 broken during poking: the ground symbol sits ~30 px from the
    source's foot with all nearby wire erased — the two-tier touching-terminals
    rule (claimed<->unclaimed at tight range) fixed it.
    """
    import schemdraw
    import schemdraw.elements as elm

    k = rng.randint(2, 4)
    volts = rng.choice(SOURCE_VOLTAGES)
    netlist = Netlist()
    boxed = []
    d = schemdraw.Drawing(show=False)
    src = elm.SourceV().up().label(f"{volts}V")
    d += src
    netlist.add("V", "V1", volts, "n1", "0")
    boxed.append((src, "voltage source", "V1", f"{volts}V"))
    prev = "n1"
    for i in range(k):
        val = rng.choice(RESISTOR_VALUES)
        r = elm.Resistor().left().label(val)
        d += r
        nxt = "0" if i == k - 1 else f"n{i + 2}"
        netlist.add("R", f"R{i + 1}", val, prev, nxt)
        boxed.append((r, "resistor", f"R{i + 1}", val))
        prev = nxt
    d += elm.Line().down().toy(src.start)
    d += elm.Line().to(src.start)
    gnd = elm.Ground().at(src.start)
    d += gnd
    boxed.append((gnd, "ground", "GND", None))
    return d, netlist, boxed
