"""Visual debugger for wire extraction — see what the extractor sees.

WHY THIS EXISTS
---------------
When extraction gets a circuit wrong, a netlist diff tells you THAT it failed
but not WHERE the pixels misled it. This module draws the extractor's entire
mental model on top of the original image:

  * the erased-region rectangles (light yellow) — what got blanked out,
  * the skeletonised wires, coloured by which net they ended up in,
  * every cut ENDPOINT as a red circle — the redesign's key evidence,
  * every terminal as a marker labelled with its assigned net name,
  * junction dots outlined in purple.

One glance answers "which rule misfired" — far faster than print statements.

Run on a synthetic circuit:
    python vision/debug_viz.py --seed 3 --out debug_out
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")            # no window needed; we save straight to a file
import matplotlib.pyplot as plt
import numpy as np


def save_debug_image(image: np.ndarray, components: list[dict],
                     info: dict, out_path: str | Path) -> Path:
    """Render the extractor's debug payload over the image and save a PNG.

    `info` is the dict returned by ``extract_netlist(..., debug=True)``.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labeled = info["labeled"]
    h, w = labeled.shape
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    ax.imshow(image, alpha=0.35)

    # Erased regions — the rectangles the extractor blanked out.
    for comp in components:
        xmin, ymin, xmax, ymax = comp["bbox"]
        face = {"junction": "violet", "ground": "green",
                "text": "khaki"}.get(comp["kind"], "gold")
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=True, facecolor=face, alpha=0.15,
                                   edgecolor=face, linewidth=0.8))

    # Skeleton wires, coloured by connected wire id.
    ys, xs = np.where(labeled > 0)
    if xs.size:
        ax.scatter(xs, ys, c=labeled[ys, xs], s=0.5, cmap="tab20")

    # Cut endpoints — the evidence the matcher works from.
    for ep in info["endpoints"]:
        x, y = ep["pos"]
        ax.add_patch(plt.Circle((x, y), info["match_radius"],
                                fill=False, color="red", linewidth=1.2))

    # Terminals, labelled with their final net.
    for rec in info["records"]:
        x, y = rec["point"]
        comp_name = rec["comp"]["name"]
        net = info["terminal_nets"].get(f"{comp_name}_{rec['side']}", "0?")
        if rec["claimed"]:
            colour = "tab:blue"               # matched to a cut wire endpoint
        elif rec["on_junction"]:
            colour = "purple"                 # connected through a junction dot
        elif rec.get("touch_matched"):
            colour = "darkorange"             # connected by terminal adjacency
        else:
            colour = "crimson"                # genuinely unresolved
        ax.plot(x, y, "x", color=colour, markersize=8, markeredgewidth=2)
        ax.annotate(f"{comp_name}.{rec['side']}={net}", (x, y),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7, color=colour)

    ax.set_title("wire-extraction debug — x: blue=face-matched, purple=junction, "
                 "orange=terminal-adjacency, crimson=unresolved; red ◯ = cut endpoint")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the wire extractor's debug view for a synthetic circuit.")
    parser.add_argument("--seed", type=int, default=0, help="generator seed")
    parser.add_argument("--out", type=str, default="debug_out",
                        help="output folder")
    args = parser.parse_args()

    from data_collection.synthetic import generate_one
    from vision.wire_extraction import extract_netlist

    rng = random.Random(args.seed)
    pil_image, gt = generate_one(rng)
    image = np.asarray(pil_image)
    netlist, info = extract_netlist(image, gt["components"], debug=True)

    out = Path(args.out) / f"debug_seed{args.seed}_{gt['template']}.png"
    save_debug_image(image, gt["components"], info, out)
    print(f"template: {gt['template']}")
    print(f"extracted:\n{netlist.to_spice()}")
    print(f"debug image: {out}")


if __name__ == "__main__":
    main()
