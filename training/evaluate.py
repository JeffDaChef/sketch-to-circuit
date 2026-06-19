"""Evaluate a trained YOLO detector on the held-out test split.

WHY THIS EXISTS
---------------
After training you have a `best.pt` file, but how do you know if it actually works?
This script runs the trained model on the TEST set — images (and drafters) the
model NEVER saw during training — and reports accuracy numbers.

WHAT mAP MEANS (plain English)
-------------------------------
mAP stands for "mean Average Precision". You don't need to memorise the maths,
just remember two things:

  1. Higher is better. A perfect detector would score 1.0 (or 100%).
     A random guess scores near 0.

  2. It measures how well the predicted boxes match the real (ground-truth) boxes.
     A predicted box "matches" if it overlaps enough with the true box AND has the
     right class label. Two thresholds are reported:

     • mAP50     — a box is counted as correct if it overlaps the true box by ≥50%.
                   This is the "easy" version; component detectors usually score
                   higher here.

     • mAP50-95  — the same idea but averaged over overlap thresholds from 50% to
                   95% in steps of 5%. This is stricter and the standard benchmark.

WHY EVALUATE ON HELD-OUT DRAFTERS (not random images)?
-------------------------------------------------------
The CGHD dataset is split by drafter (the person who drew the circuits), NOT just
by random image. That means every image from one person ends up entirely in train
OR entirely in test — never split across both.

Why does that matter? If you split randomly, the same drafter's handwriting appears
in both train and test. The model learns "when the lines look like Person A's
handwriting, it is probably a resistor" — which gives an artificially high score
because it has already seen that handwriting style. Splitting by drafter forces the
model to generalise to entirely new handwriting it has never seen, which is the
honest, real-world test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def summarize_results(results) -> dict:
    """Pull the metrics we report out of an ultralytics val() results object.

    Returns ``{"map50": float, "map50_95": float, "per_class": {name: ap50},
    "missing_classes": [names with no test examples]}``.

    THE SUBTLETY (and why this is a function we can unit-test): ultralytics stores
    per-class arrays (``box.ap50``) aligned with ``box.ap_class_index`` — i.e. only
    the classes that actually APPEAR in the evaluated split, in that order — NOT one
    entry per global class id. On a drafter-split test set a rare class (e.g.
    "switch") is often absent, so indexing ``ap50`` with a global class id silently
    reads the wrong class's score (or runs off the end). We therefore walk
    ``ap_class_index`` and map each position back to its name, and report the
    truly-absent classes separately instead of printing a bogus number for them.
    """
    box = results.box
    names = results.names
    present = [int(i) for i in box.ap_class_index]
    ap50 = box.ap50
    per_class = {names[cls_id]: float(ap50[pos]) for pos, cls_id in enumerate(present)}
    missing = [names[i] for i in names if i not in present]
    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": per_class,
        "missing_classes": missing,
    }


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "\n[ERROR] The 'ultralytics' package is not installed.\n"
            "Fix it by running:  pip install ultralytics\n"
        )
        sys.exit(1)
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLO detector on the circuit test split."
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to best.pt (or any YOLO .pt weights file).",
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the data.yaml file produced by cghd_prep.py.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size in pixels (must match training). Default 640.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which data split to evaluate on. Default 'test' (the honest held-out set).",
    )

    args = parser.parse_args()

    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        print(f"\n[ERROR] Weights file not found: {weights_path}")
        print("Run training first, then pass the path to best.pt.")
        sys.exit(1)

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        print(f"\n[ERROR] data.yaml not found at: {data_path}")
        sys.exit(1)

    print(f"\n[INFO] Loading weights from: {weights_path}")
    model = YOLO(str(weights_path))

    print(f"[INFO] Evaluating on split: '{args.split}'")
    print(f"[INFO] Data config:          {data_path}\n")

    results = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        split=args.split,
    )

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS  (split = {args.split})")
    print("=" * 60)

    try:
        summary = summarize_results(results)
    except AttributeError as exc:
        print(f"[WARN] Could not read metrics ({exc}) — printing raw results:")
        print(results)
        return

    print(f"  mAP50      : {summary['map50']:.4f}   (boxes correct at ≥50% overlap)")
    print(f"  mAP50-95   : {summary['map50_95']:.4f}   (stricter benchmark — lower is normal)")

    print("\nPer-class mAP50 (classes present in this split):")
    print("-" * 40)
    for name, score in sorted(summary["per_class"].items()):
        print(f"  {name:<20s}: {score:.4f}")
    if summary["missing_classes"]:
        print(f"\n  (no test examples for: {', '.join(sorted(summary['missing_classes']))})")

    print("=" * 60)
    print("\nTip: mAP50 > 0.80 on this dataset is a solid result for circuit components.")
    print("     If a class scores low, check that its labels look correct in the dataset.\n")


if __name__ == "__main__":
    main()
