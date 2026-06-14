"""Synthetic schematic generator.

WHY THIS EXISTS
---------------
The camera/vision modules (built in later phases) need lots of circuit images to
test against — and for each image we need to already KNOW the right answer
(which components exist, where they are, and what circuit they form). Hand-drawing
and hand-labelling hundreds of circuits is slow, so instead we GENERATE them:
this script draws clean circuits with `schemdraw` and, because our own code placed
every part, it can write a perfect "answer key" (ground-truth JSON) for free.

Each generated sample is two files that share a name:
  <name>.png   — the picture
  <name>.json  — the answer key: the netlist, every component's pixel bounding
                 box, and the solved node voltages.

This is Phase-0 infrastructure: no camera, no machine learning. The images are
clean (not hand-drawn-looking) — adding hand-drawn jitter/rotation augmentation is
a documented TODO for when we actually train on synthetic data.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

# When you run a script directly, Python only searches that script's own folder.
# Our `solver` package lives one level up (the repo root), so we add the repo
# root to the import path here. (__file__ is this file; .parent.parent is the
# project root.) This lets `python data_collection/synthetic.py` find `solver`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
import schemdraw
import schemdraw.elements as elm

from solver.mna import solve
from solver.netlist import Netlist

# Realistic component values to pick from, so the dataset looks like real circuits.
RESISTOR_VALUES = ["220", "470", "1k", "2.2k", "4.7k", "10k", "22k", "47k", "100k"]
SOURCE_VOLTAGES = ["5", "9", "12", "3.3"]


@dataclass
class CompBox:
    """One component's ground-truth label: what it is and where it is in the image."""

    name: str           # 'R1', 'V1', ...
    kind: str           # 'resistor', 'voltage source', 'ground', 'junction'
    value: str | None   # the written value ('10k') or None for ground/junction
    bbox: list[float]   # [xmin, ymin, xmax, ymax] in PIXELS, top-left origin


# --- circuit templates -------------------------------------------------------
# Each template function places schemdraw elements AND records, for every element,
# the electrical info (so the netlist always matches the picture). It returns:
#   drawing, netlist, list of (element, kind, name, value-or-None) to be boxed.


def _series_divider(rng: random.Random, k: int | None = None):
    """A voltage source driving a vertical chain of resistors (a divider).

    Layout: source on the left going up; a wire across the top; then the
    resistors stacked going down the right-hand side back to ground. The nodes
    between resistors are the divider 'taps'.

    `k` (optional) forces the number of resistors, for difficulty-scaling studies;
    by default it's random in [2, 4] as before.
    """
    k = k if k is not None else rng.randint(2, 4)   # how many resistors in the chain
    volts = rng.choice(SOURCE_VOLTAGES)
    netlist = Netlist()
    boxed = []                                  # (element, kind, name, value)

    d = schemdraw.Drawing(show=False)

    # Voltage source: bottom node is ground '0', top node is 'n1'.
    src = elm.SourceV().up().label(f"{volts}V")
    d += src
    netlist.add("V", "V1", volts, "n1", "0")    # + terminal (top) is 'n1'
    boxed.append((src, "voltage source", "V1", f"{volts}V"))

    # Wire across the top to the start of the resistor column.
    d += elm.Line().right().length(3)

    # Resistors going down: node names n1 -> n2 -> ... -> '0'.
    top_node = "n1"
    for i in range(k):
        bottom_node = "0" if i == k - 1 else f"n{i + 2}"
        val = rng.choice(RESISTOR_VALUES)
        # loc="right" places the value label on the OUTER side of the column. The
        # default (interior) side sits between the source and resistor columns, and
        # at high resistor counts those label boxes crowd the wires there — and
        # since the pipeline erases detected text, erasing a label over a wire would
        # cut the connection. Keeping labels clear of the wiring is also just good
        # schematic practice, and lets the difficulty study reach large circuits.
        res = elm.Resistor().down().label(val, loc="right")
        d += res
        rname = f"R{i + 1}"
        netlist.add("R", rname, val, top_node, bottom_node)
        boxed.append((res, "resistor", rname, val))
        # Junction dot at every internal tap (drawing convention §3).
        if i < k - 1:
            dot = elm.Dot()
            d += dot
            boxed.append((dot, "junction", f"J{i + 1}", None))
        top_node = bottom_node

    # Wire back along the bottom to the source, a ground symbol at the corner,
    # and a wire UP to the source's bottom terminal so the loop is visibly
    # closed on paper. (An earlier version left this gap — electrically implied
    # but never drawn — which made the image unextractable from pixels alone
    # and violated our own drawing convention.)
    bottom_rail = elm.Line().left().length(3)
    d += bottom_rail
    gnd = elm.Ground()
    d += gnd
    boxed.append((gnd, "ground", "GND", None))
    d += elm.Line().at(bottom_rail.end).to(src.start)

    return d, netlist, boxed


def _parallel_bank(rng: random.Random, k: int | None = None):
    """A voltage source driving several resistors all in parallel.

    Layout: source on the left; a top rail and a bottom rail; resistors hang as
    vertical rungs between the rails, so every resistor sees the full source
    voltage. All rungs share the same two nodes -> parallel.
    """
    k = k if k is not None else rng.randint(2, 3)
    volts = rng.choice(SOURCE_VOLTAGES)
    netlist = Netlist()
    boxed = []

    d = schemdraw.Drawing(show=False)

    # Source on the left. Its top is the 'top' rail node; its bottom is ground.
    src = elm.SourceV().up().length(3).label(f"{volts}V")
    d += src
    netlist.add("V", "V1", volts, "top", "0")
    boxed.append((src, "voltage source", "V1", f"{volts}V"))

    top_anchor = src.end                        # top rail position (moves right each rung)
    bot_anchor = src.start                      # bottom rail position (moves right each rung)

    for i in range(k):
        # Extend the top and bottom rails one step to the right, in lockstep so
        # they stay parallel and aligned. The resistor 'rung' then bridges them.
        top_seg = elm.Line().right().length(3).at(top_anchor)
        d += top_seg
        bot_seg = elm.Line().right().length(3).at(bot_anchor)
        d += bot_seg

        top_dot = elm.Dot().at(top_seg.end)
        d += top_dot
        bot_dot = elm.Dot().at(bot_seg.end)
        d += bot_dot

        # Resistor drawn explicitly from the top rail down to the bottom rail, so
        # it is genuinely connected at both ends (no dangling open circuit).
        val = rng.choice(RESISTOR_VALUES)
        res = elm.Resistor().at(top_seg.end).to(bot_seg.end).label(val)
        d += res
        rname = f"R{i + 1}"
        netlist.add("R", rname, val, "top", "0")
        boxed.append((res, "resistor", rname, val))
        boxed.append((top_dot, "junction", f"JT{i + 1}", None))
        boxed.append((bot_dot, "junction", f"JB{i + 1}", None))

        top_anchor = top_seg.end
        bot_anchor = bot_seg.end

    # Ground symbol hanging off the bottom rail at the source.
    gnd = elm.Ground().at(src.start)
    d += gnd
    boxed.append((gnd, "ground", "GND", None))

    return d, netlist, boxed


def _series_loop(rng: random.Random, k: int | None = None):
    """A source on the left driving a chain of HORIZONTAL resistors.

    Layout: source standing on the left; resistors marching right along the
    top; a wire dropping down the right side and returning along the bottom to
    a ground at the source's foot. This template exists to keep the vision
    code honest: it exercises horizontal components (left/right terminals) and
    a source-lead-to-resistor-lead connection with no junction dot — the exact
    things the first wire extractor couldn't handle (it scored 0/30 here; see
    docs/wire_extraction_redesign_plan.md). `k` (optional) forces the chain length
    for difficulty studies; default random [2, 4].
    """
    k = k if k is not None else rng.randint(2, 4)
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
        res = elm.Resistor().right().label(val)
        d += res
        nxt = "0" if i == k - 1 else f"n{i + 2}"
        netlist.add("R", f"R{i + 1}", val, prev, nxt)
        boxed.append((res, "resistor", f"R{i + 1}", val))
        prev = nxt

    # Close the loop: down the right edge, back along the bottom, ground at
    # the source's foot.
    d += elm.Line().down().toy(src.start)
    d += elm.Line().to(src.start)
    gnd = elm.Ground().at(src.start)
    d += gnd
    boxed.append((gnd, "ground", "GND", None))
    return d, netlist, boxed


TEMPLATES = [_series_divider, _parallel_bank, _series_loop]


# --- rendering + ground-truth extraction ------------------------------------


def _bbox_to_pixels(data_bbox, fig, ax, width_px: int, height_px: int) -> list[float]:
    """Convert a schemdraw data-coordinate bbox into image pixel coordinates.

    schemdraw places elements in the matplotlib axes' data space. We can't use
    raw display pixels because on a Retina/HiDPI screen matplotlib's pixel count
    is half the real image size (device_pixel_ratio = 2). So we map data ->
    FIGURE FRACTION first (a 0..1 coordinate that is resolution-independent),
    then multiply by the ACTUAL image width/height. Figure-fraction y runs
    bottom-to-top, but images count rows top-to-bottom, so we flip with (1 - y).
    """
    to_fraction = ax.transData + fig.transFigure.inverted()
    (fx0, fy0) = to_fraction.transform((data_bbox.xmin, data_bbox.ymin))
    (fx1, fy1) = to_fraction.transform((data_bbox.xmax, data_bbox.ymax))
    xmin, xmax = sorted((fx0 * width_px, fx1 * width_px))
    ymin, ymax = sorted(((1 - fy0) * height_px, (1 - fy1) * height_px))
    return [round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)]


def generate_one(rng: random.Random):
    """Build one random circuit and return (PIL image, ground-truth dict)."""
    template = rng.choice(TEMPLATES)
    d, netlist, boxed = template(rng)
    d.draw(show=False)                          # render onto matplotlib

    fig = d.fig.fig                             # the real matplotlib Figure
    ax = d.fig.ax

    # Add a uniform margin around the drawing so components or labels sitting at
    # the very edge (e.g. the source's voltage label) don't get clipped — clipped
    # parts would be wrong ground truth for the vision modules later.
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    pad = 0.12 * max(x1 - x0, y1 - y0)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)

    fig.canvas.draw()

    # Pull the rendered pixels straight from the canvas so the image matches the
    # transform we use for bounding boxes exactly. On HiDPI this buffer is bigger
    # than get_width_height() reports, so we read the TRUE size from its shape.
    rgba = np.asarray(fig.canvas.buffer_rgba())
    height_px, width_px = rgba.shape[0], rgba.shape[1]
    image = Image.fromarray(rgba).convert("RGB")

    boxes = []
    for element, kind, name, value in boxed:
        px = _bbox_to_pixels(element.get_bbox(transform=True), fig, ax, width_px, height_px)
        boxes.append(CompBox(name=name, kind=kind, value=value, bbox=px))

    # Ground-truth boxes for the VALUE LABELS too (kind="text"). The real
    # pipeline detects text as its own class and erases it before tracing
    # wires — if the synthetic answer key omitted these, label ink would
    # masquerade as wire. matplotlib knows each rendered label's exact extent
    # in display coords; map display -> figure fraction -> image pixels (the
    # same HiDPI-safe route _bbox_to_pixels takes).
    to_fraction = fig.transFigure.inverted()
    for i, artist in enumerate(ax.texts):
        ext = artist.get_window_extent(renderer=fig.canvas.get_renderer())
        fx0, fy0 = to_fraction.transform((ext.x0, ext.y0))
        fx1, fy1 = to_fraction.transform((ext.x1, ext.y1))
        xmin, xmax = sorted((fx0 * width_px, fx1 * width_px))
        ymin, ymax = sorted(((1 - fy0) * height_px, (1 - fy1) * height_px))
        boxes.append(CompBox(
            name=f"TXT{i + 1}", kind="text", value=artist.get_text(),
            bbox=[round(xmin, 1), round(ymin, 1), round(xmax, 1), round(ymax, 1)],
        ))

    result = solve(netlist)                     # solve it, so the answer key is complete

    ground_truth = {
        "template": template.__name__.lstrip("_"),
        "image_size": [width_px, height_px],
        "components": [asdict(b) for b in boxes],
        "netlist_spice": netlist.to_spice().strip(),
        "node_voltages": {k: round(v, 6) for k, v in result.node_voltages.items()},
    }
    import matplotlib.pyplot as plt
    plt.close(fig)                              # free the figure (avoid memory buildup)
    return image, ground_truth


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic circuit schematics + ground truth.")
    parser.add_argument("--count", type=int, default=10, help="how many samples to make")
    parser.add_argument("--out", type=str, default="synth_out", help="output folder")
    parser.add_argument("--seed", type=int, default=0, help="random seed (reproducible datasets)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for i in range(args.count):
        image, gt = generate_one(rng)
        stem = f"circuit_{i:03d}"
        image.save(out_dir / f"{stem}.png")
        with open(out_dir / f"{stem}.json", "w") as f:
            json.dump(gt, f, indent=2)
        print(f"  {stem}.png  ({gt['template']}, {len(gt['components'])} components)")

    print(f"Wrote {args.count} samples to {out_dir}/")


if __name__ == "__main__":
    main()
