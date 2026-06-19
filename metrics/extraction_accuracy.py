"""End-to-end extraction accuracy metric for the sketch-to-circuit pipeline.

WHY THIS EXISTS
---------------
After building the wire-extraction algorithm we need a single headline number
that answers "how often does the full pipeline give the right answer?"  We call
this the **end-to-end extraction accuracy**: we generate a batch of random
circuits (with known-correct netlists), run extraction on each, and ask the
equivalence checker whether the recovered topology matches the truth.

The result is broken down two ways:

1. **By template** (circuit family: series_divider, parallel_bank, …) — so we
   can see if one family is harder to extract than another.
2. **By component count** (how many two-terminal components are in the truth
   netlist) — the "difficulty curve", because more components means more wires
   and more opportunities for the extractor to miss a connection.

This module is also importable as a library so other scripts can call
``evaluate_extraction(count, seed)`` programmatically.

Run it directly:
    python metrics/extraction_accuracy.py --count 200 --seed 0
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from data_collection.synthetic import generate_one
from solver.equivalence import circuit_equivalent
from solver.netlist import Netlist
from vision.wire_extraction import extract_netlist



def evaluate_extraction(count: int, seed: int = 0) -> dict:
    """Run the extraction pipeline on ``count`` synthetic circuits and score them.

    Uses a single ``random.Random(seed)`` instance across ALL circuits so each
    call to ``generate_one`` advances the same RNG — changing the seed gives a
    completely different batch, but calling this function twice with the same
    arguments always returns the same result (deterministic).

    Parameters
    ----------
    count:
        How many circuits to generate and evaluate.
    seed:
        Random seed. Default 0.

    Returns
    -------
    A dict with the following keys:

    * ``"total"``      — int, equals ``count``.
    * ``"correct"``    — int, circuits where extraction matched ground truth.
    * ``"accuracy"``   — float, correct/total (0.0 if total == 0).
    * ``"by_template"``        — dict[str, [correct, total]] per circuit family.
    * ``"by_component_count"`` — dict[int, [correct, total]] per component count.
    * ``"failures"``   — list of up to 20 dicts describing failed circuits.
    """
    rng = random.Random(seed)

    total_correct = 0
    by_template: dict[str, list[int]] = {}
    by_comp_count: dict[int, list[int]] = {}
    failures: list[dict] = []

    for i in range(count):
        img, gt = generate_one(rng)

        template_name: str = gt["template"]

        extracted: Netlist = extract_netlist(np.asarray(img), gt["components"])

        truth: Netlist = Netlist.from_spice(gt["netlist_spice"])

        n_components: int = len(truth.components)

        is_correct: bool = circuit_equivalent(extracted, truth)

        if is_correct:
            total_correct += 1

        if template_name not in by_template:
            by_template[template_name] = [0, 0]
        by_template[template_name][1] += 1
        if is_correct:
            by_template[template_name][0] += 1

        if n_components not in by_comp_count:
            by_comp_count[n_components] = [0, 0]
        by_comp_count[n_components][1] += 1
        if is_correct:
            by_comp_count[n_components][0] += 1

        if not is_correct and len(failures) < 20:
            failures.append({
                "index": i,
                "template": template_name,
                "n_components": n_components,
            })

    accuracy = total_correct / count if count > 0 else 0.0

    return {
        "total": count,
        "correct": total_correct,
        "accuracy": accuracy,
        "by_template": by_template,
        "by_component_count": by_comp_count,
        "failures": failures,
    }



def _print_report(result: dict) -> None:
    """Print a human-readable accuracy report to stdout."""
    total   = result["total"]
    correct = result["correct"]
    pct     = result["accuracy"] * 100.0

    print(f"\nExtraction accuracy: {correct}/{total} = {pct:.1f}%")

    print("\nBy template:")
    print(f"  {'Template':<22}  {'Correct':>7}  {'Total':>5}  {'Accuracy':>8}")
    print(f"  {'-'*22}  {'-'*7}  {'-'*5}  {'-'*8}")
    for tname, (c, t) in sorted(result["by_template"].items()):
        acc_str = f"{100.0*c/t:.1f}%" if t > 0 else "  n/a"
        print(f"  {tname:<22}  {c:>7}  {t:>5}  {acc_str:>8}")

    print("\nBy circuit complexity (# components):")
    print(f"  {'# components':>12}  {'Correct':>7}  {'Total':>5}  {'Accuracy':>8}")
    print(f"  {'-'*12}  {'-'*7}  {'-'*5}  {'-'*8}")
    for n in sorted(result["by_component_count"].keys()):
        c, t = result["by_component_count"][n]
        acc_str = f"{100.0*c/t:.1f}%" if t > 0 else "  n/a"
        print(f"  {n:>12}  {c:>7}  {t:>5}  {acc_str:>8}")

    if result["failures"]:
        print("\nFirst failures:")
        for f in result["failures"][:5]:
            print(f"  circuit #{f['index']:>4}  template={f['template']}  "
                  f"n_components={f['n_components']}")
    else:
        print("\nNo failures.")

    print()



def main() -> None:
    """Parse arguments, run evaluation, and print a report."""
    parser = argparse.ArgumentParser(
        description="Measure end-to-end extraction accuracy on synthetic circuits.",
    )
    parser.add_argument(
        "--count", type=int, default=200,
        help="how many circuits to evaluate (default: 200)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="random seed for reproducibility (default: 0)",
    )
    args = parser.parse_args()

    print(f"Evaluating {args.count} circuits (seed={args.seed}) …")
    result = evaluate_extraction(args.count, seed=args.seed)
    _print_report(result)


if __name__ == "__main__":
    main()
