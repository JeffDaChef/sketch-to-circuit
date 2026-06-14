"""Ablation: the old (blob-proximity) wire extractor vs the new (skeleton-graph) one.

WHY THIS EXISTS
---------------
The wire extractor was redesigned once already. The first version matched
component terminals to nearby wire *blobs* by proximity — which quietly memorised
the two layouts it was tuned on and fell apart on anything else. The rewrite
builds a real graph of the wires and matches terminals to the cut endpoints that
component-erasure creates, with no layout assumptions.

"We rewrote it and it's better" is a claim. This script turns it into a *number*:
it runs BOTH extractors (the old one is frozen verbatim in
`vision/wire_extraction_baseline.py`) on the exact same circuits and reports the
before/after accuracy per layout. That before/after table is the single most
"I did science" artifact in the project — an honest ablation showing a design
choice paid off, not just that it exists.

The layouts span the easy cases both handle (vertical divider, parallel bank) and
the cases that expose the old one's overfitting (a looped layout, a corner-turning
chain, a mirrored loop).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_collection.extra_layouts import make_corner_chain, make_mirrored_loop, render
from data_collection.synthetic import _parallel_bank, _series_divider, _series_loop
from solver.equivalence import circuit_equivalent
from vision.wire_extraction import extract_netlist as extract_new
from vision.wire_extraction_baseline import extract_netlist as extract_old

# Every layout has the signature maker(rng) -> (drawing, truth_netlist, boxed).
LAYOUT_MAKERS = {
    "series_divider": _series_divider,
    "parallel_bank": _parallel_bank,
    "series_loop": _series_loop,
    "corner_chain": make_corner_chain,
    "mirrored_loop": make_mirrored_loop,
}

EXTRACTORS = {
    "baseline": extract_old,
    "skeleton_graph": extract_new,
}


def run_ablation(n_seeds: int = 30, makers: dict | None = None) -> dict:
    """Run both extractors on the same circuits; return per-layer [correct, total].

    Each (layout, seed) circuit is rendered ONCE and fed to both extractors, so the
    comparison is exactly fair and we don't pay to render twice.
    """
    makers = makers if makers is not None else LAYOUT_MAKERS
    results = {layout: {name: [0, 0] for name in EXTRACTORS} for layout in makers}
    for layout, maker in makers.items():
        for seed in range(n_seeds):
            d, truth, boxed = maker(random.Random(seed))
            image, comps = render(d, boxed)
            for name, fn in EXTRACTORS.items():
                results[layout][name][1] += 1
                try:
                    if circuit_equivalent(fn(image, comps), truth):
                        results[layout][name][0] += 1
                except Exception:
                    pass            # a crash is just a wrong answer for this comparison
    return results


def _totals(results: dict) -> dict[str, list[int]]:
    """Sum [correct, total] across all layouts, per extractor."""
    totals = {name: [0, 0] for name in EXTRACTORS}
    for per_extractor in results.values():
        for name, (c, t) in per_extractor.items():
            totals[name][0] += c
            totals[name][1] += t
    return totals


def _print_table(results: dict) -> None:
    names = list(EXTRACTORS)
    print(f"\n  {'layout':<16}  " + "  ".join(f"{n:>15}" for n in names))
    print(f"  {'-'*16}  " + "  ".join("-" * 15 for _ in names))
    for layout in results:
        cells = []
        for n in names:
            c, t = results[layout][n]
            cells.append(f"{c}/{t} ({100*c/t:>3.0f}%)" if t else "n/a")
        print(f"  {layout:<16}  " + "  ".join(f"{cell:>15}" for cell in cells))
    print(f"  {'-'*16}  " + "  ".join("-" * 15 for _ in names))
    tot = _totals(results)
    cells = [f"{c}/{t} ({100*c/t:>3.0f}%)" for c, t in (tot[n] for n in names)]
    print(f"  {'OVERALL':<16}  " + "  ".join(f"{cell:>15}" for cell in cells))


def save_plot(results: dict, path: str) -> None:
    """Grouped bar chart: baseline vs skeleton-graph accuracy per layout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    layouts = list(results)
    names = list(EXTRACTORS)
    x = np.arange(len(layouts))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, name in enumerate(names):
        accs = [100 * results[l][name][0] / results[l][name][1] if results[l][name][1] else 0
                for l in layouts]
        ax.bar(x + (i - 0.5) * width, accs, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(layouts, rotation=20, ha="right")
    ax.set_ylabel("extraction accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Wire extractor ablation: blob-proximity vs skeleton-graph")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Baseline vs skeleton-graph extractor ablation.")
    p.add_argument("--seeds", type=int, default=30, help="circuits per layout (default 30)")
    p.add_argument("--plot", default="extractor_ablation.png", help="output PNG path")
    args = p.parse_args()

    print(f"Running both extractors on {len(LAYOUT_MAKERS)} layouts × {args.seeds} seeds …")
    results = run_ablation(args.seeds)
    _print_table(results)
    save_plot(results, args.plot)
    print(f"\nSaved chart to {args.plot}")


if __name__ == "__main__":
    main()
