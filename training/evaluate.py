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

# Guard the ultralytics import so a missing install gives a friendly message
# instead of a confusing ModuleNotFoundError stack trace.
try:
    from ultralytics import YOLO
except ImportError:
    print(
        "\n[ERROR] The 'ultralytics' package is not installed.\n"
        "Fix it by running:  pip install ultralytics\n"
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLO detector on the circuit test split."
    )

    # Path to the weights file produced by training. This is the "trained brain"
    # of the detector — without it, we have nothing to evaluate.
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to best.pt (or any YOLO .pt weights file).",
    )

    # data.yaml tells the validator where the test images and labels live, and
    # what the class names are. Must be the same file used during training.
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the data.yaml file produced by cghd_prep.py.",
    )

    # Must match the size used during training. Changing it can hurt accuracy
    # because the model was trained to expect 640×640 inputs.
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size in pixels (must match training). Default 640.",
    )

    # 'test' uses the held-out test split (never seen during training).
    # You can pass 'val' to quickly check the validation split instead.
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which data split to evaluate on. Default 'test' (the honest held-out set).",
    )

    args = parser.parse_args()

    # Sanity-check paths before loading the (possibly large) model.
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

    # .val() runs the model over every image in the chosen split, computes
    # predictions, compares them to the ground-truth labels, and returns a
    # results object with all the metrics packed in.
    results = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        split=args.split,
    )

    # --- Overall metrics -------------------------------------------------------
    # results.box holds the detection metrics. The .map and .map50 attributes are
    # the standard mAP50-95 and mAP50 respectively.
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS  (split = {args.split})")
    print("=" * 60)

    try:
        map50_95 = results.box.map       # mAP averaged over IoU 0.50:0.95
        map50    = results.box.map50     # mAP at IoU threshold 0.50
        print(f"  mAP50      : {map50:.4f}   (boxes correct at ≥50% overlap)")
        print(f"  mAP50-95   : {map50_95:.4f}   (stricter benchmark — lower is normal)")
    except AttributeError:
        # Fallback: ultralytics occasionally restructures its results object
        # between versions. In that case, print whatever it gives us.
        print("[WARN] Could not read .box.map — printing full results object instead:")
        print(results)

    # --- Per-class breakdown ---------------------------------------------------
    # The per-class mAP lets us see which component types the model struggles with.
    # For example, 'junction' (a simple dot) and 'text' often score lower because
    # they are small or visually similar to noise.
    print()
    print("Per-class mAP50:")
    print("-" * 40)

    try:
        # results.box.maps is a numpy array of per-class mAP50-95 values, ordered
        # the same as the class names in data.yaml. results.names is a dict mapping
        # integer index -> class name string.
        class_names = results.names           # {0: 'capacitor', 1: 'diode', ...}
        per_class_maps = results.box.maps     # array of mAP50-95 per class

        # Also try to get per-class mAP50 specifically. ultralytics does not always
        # expose this directly, so we fall back to mAP50-95 if needed.
        try:
            # ap_class_index exists in some ultralytics versions; try it first.
            per_class_map50 = results.box.ap50  # per-class AP at IoU=0.50
        except AttributeError:
            per_class_map50 = per_class_maps    # fall back to mAP50-95

        for idx, name in sorted(class_names.items()):
            try:
                score = float(per_class_map50[idx])
                print(f"  {name:<20s}: {score:.4f}")
            except (IndexError, TypeError):
                # If the array length does not align with the class count,
                # skip gracefully rather than crashing.
                print(f"  {name:<20s}: (n/a)")

    except (AttributeError, TypeError) as exc:
        # If the ultralytics API changed and none of the above attributes exist,
        # we print a warning and fall back to the full results summary. This keeps
        # the script useful even if an ultralytics update renames things.
        print(f"[WARN] Per-class breakdown not available ({exc}).")
        print("       Full results summary printed above instead.")

    print("=" * 60)
    print("\nTip: mAP50 > 0.80 on this dataset is a solid result for circuit components.")
    print("     If a class scores low, check that its labels look correct in the dataset.\n")


if __name__ == "__main__":
    main()
