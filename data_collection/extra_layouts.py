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


def crossover_circuit(rng: random.Random | None = None):
    """A circuit with ONE genuine wire crossover, drawn directly for full control.

    Topology (a series divider): V1: n1→0, R1: n1→n2, R2: n2→0. It is laid out so
    the **n1** horizontal rail and the **n2** vertical wire *cross without
    connecting*. If that crossing is treated as a normal junction the two nets fuse
    (n1≡n2) and R1 becomes a short — the whole reason the no-crossing-wires
    constraint existed. With crossover handling the three nets stay distinct.

    Unlike the schemdraw makers above this is rasterised directly (axis-aligned
    segments) because forcing schemdraw to draw a clean, single, known-location
    crossing is far more fragile than placing the pixels ourselves. Returns
    ``(image, components, truth)`` where ``components`` already includes the
    ``crossover`` box (the detector stand-in); drop it to see the un-threaded
    failure. `rng` (optional) jitters the whole drawing so a suite gets variety.
    """
    rng = rng or random.Random(0)
    ox, oy = rng.randint(-12, 12), rng.randint(-12, 12)        # global translation jitter
    va, vb = rng.choice(RESISTOR_VALUES), rng.choice(RESISTOR_VALUES)
    volts = rng.choice(SOURCE_VOLTAGES)

    H, W = 460, 420
    img = np.full((H, W), 255, dtype=np.uint8)

    def hseg(y, x0, x1, t=2):
        img[y - t:y + t + 1, min(x0, x1):max(x0, x1) + 1] = 0
    def vseg(x, y0, y1, t=2):
        img[min(y0, y1):max(y0, y1) + 1, x - t:x + t + 1] = 0

    # All coordinates are shifted by the (ox, oy) jitter via these helpers.
    def hx(y, x0, x1): hseg(y + oy, x0 + ox, x1 + ox)
    def vx(x, y0, y1): vseg(x + ox, y0 + oy, y1 + oy)
    def box(x0, y0, x1, y1): return [x0 + ox, y0 + oy, x1 + ox, y1 + oy]

    # net n1: rail + riser + source/R1 stubs
    hx(240, 90, 330); vx(330, 120, 240); vx(90, 240, 260); hx(120, 310, 330)
    # net n2: R1 stub + the vertical wire that crosses the n1 rail at (210, 240)
    hx(120, 210, 230); vx(210, 120, 300)
    # net 0: source foot + bottom rail + R2 foot
    vx(90, 360, 400); hx(400, 90, 210); vx(210, 380, 400)

    components = [
        {"name": "V1", "kind": "voltage source", "value": f"{volts}V", "bbox": box(60, 260, 120, 360)},
        {"name": "R1", "kind": "resistor", "value": va, "bbox": box(230, 100, 310, 140)},
        {"name": "R2", "kind": "resistor", "value": vb, "bbox": box(185, 300, 235, 380)},
        {"name": "GND", "kind": "ground", "value": None, "bbox": box(125, 385, 175, 415)},
        {"name": "X1", "kind": "crossover", "value": None, "bbox": box(190, 220, 230, 260)},
    ]

    truth = Netlist()
    truth.add("V", "V1", volts, "n1", "0")
    truth.add("R", "R1", va, "n1", "n2")
    truth.add("R", "R2", vb, "n2", "0")
    return img, components, truth


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
