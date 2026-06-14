"""CGHD dataset converter — turns hand-drawn circuit annotations into YOLO format.

WHY THIS EXISTS
---------------
The CGHD (Circuit Graph Hand-Drawn) dataset is a public collection of scanned
hand-drawn schematics.  Each image was annotated by a human "drafter" who drew
the circuit, so every image from drafter_3, say, has the same handwriting style.
If we let one drafter's images appear in BOTH train and val, the model can just
memorise that person's style and will look great on paper but fail on new people's
drawings.  That is called data leakage.

This script avoids leakage by splitting at the DRAFTER level: every image from a
given drafter goes to exactly one of {train, val, test}.  The model therefore
always sees new handwriting styles when it is evaluated.

What it actually does, step by step:
  1. Walk the CGHD source tree, find every drafter_N folder.
  2. For each drafter, read all PASCAL-VOC XML annotations and translate them
     into YOLO label files (one text file per image).
  3. Remap CGHD's many fine-grained class names down to 8 broad classes we care
     about for v1 (see REMAP and CLASSES below).
  4. Skip any image that has zero of our 8 classes after remapping.
  5. Copy the images (optionally) into the standard Ultralytics folder layout:
       images/{train,val,test}/   and   labels/{train,val,test}/
  6. Write a data.yaml that Ultralytics YOLO can load directly for training.
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import warnings
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------------

# REMAP translates every CGHD label that we KEEP down to one of our 8 canonical
# class names.  Any CGHD label that does NOT appear in this dict is silently
# dropped.
#
# Judgment calls (edit these if you want to expand v2):
#   - "vss" is mapped to "ground" because VSS is just a named ground reference
#     rail in schematics; electrically it is the same reference node as GND.
#   - "capacitor.polarized" and "capacitor.unpolarized" both map to "capacitor"
#     because YOLO only needs to know "there is a capacitor here" for circuit
#     recognition; polarity matters at the netlist level (handled later).
#   - "voltage.dc" and "voltage.battery" both map to "voltage_source" for the
#     same reason: the detection job is just to spot the component.
#   - "diode.light_emitting" maps to "diode" — an LED IS a diode.
#
# INTENTIONALLY DROPPED for v1 (too rare or too complex to train well):
#   - "voltage.ac"                 — AC sources; rare in student schematics
#   - "resistor.adjustable"        — potentiometers; distinct shape but rare
#   - "resistor.photo"             — photoresistors; very rare
#   - "diode.zener"                — zener diodes; looks similar to regular diode
#   - "diode.thyrector"            — thyrectors; extremely rare
#   - "inductor" / "inductor.*"    — inductors; rare in beginner circuits
#   - "transistor.bjt"             — BJTs; needs a separate detection effort
#   - "transistor.fet"             — FETs; same reason
#   - "ic.*"                       — ICs; too many variants to unify now
#   - "logic.*"                    — logic gates; separate project
#   - "opamp"                      — op-amps; separate project
REMAP: dict[str, str] = {
    "resistor":               "resistor",
    "capacitor.unpolarized":  "capacitor",
    "capacitor.polarized":    "capacitor",
    "voltage.dc":             "voltage_source",
    "voltage.battery":        "voltage_source",
    "gnd":                    "ground",
    "vss":                    "ground",       # VSS = named ground rail
    "diode":                  "diode",
    "diode.light_emitting":   "diode",        # LED is electrically a diode
    "switch":                 "switch",
    "junction":               "junction",
    "text":                   "text",
}

# CLASSES is our fixed ordered list of the 8 final label names.
# The INDEX in this list is the integer class id that YOLO uses.
# It is sorted alphabetically so the mapping is deterministic and easy to check.
#
#   0 → capacitor
#   1 → diode
#   2 → ground
#   3 → junction
#   4 → resistor
#   5 → switch
#   6 → text
#   7 → voltage_source
CLASSES: list[str] = [
    "capacitor",
    "diode",
    "ground",
    "junction",
    "resistor",
    "switch",
    "text",
    "voltage_source",
]

# Pre-build a reverse lookup so we don't call .index() in a tight loop.
_CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}


# ---------------------------------------------------------------------------
# Core parsing + conversion functions
# ---------------------------------------------------------------------------


def parse_voc(xml_path: Path) -> tuple[int, int, list[tuple[str, int, int, int, int]]]:
    """Parse a PASCAL VOC XML annotation file.

    Returns:
        (width, height, objects)

    where ``objects`` is a list of (class_name, xmin, ymin, xmax, ymax) tuples
    in pixel coordinates (integers, top-left origin) exactly as stored in the
    XML.  The class_name is the raw CGHD label — not yet remapped.

    WHY VOC FORMAT: CGHD ships annotations as VOC XML because that was the
    dominant format when the dataset was created.  We parse it ourselves using
    stdlib xml.etree.ElementTree so we don't need any extra dependencies.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # <size> holds the image dimensions so we don't have to open the image.
    size_el = root.find("size")
    if size_el is None:
        raise ValueError(f"No <size> element in {xml_path}")
    width  = int(size_el.findtext("width",  default="0"))
    height = int(size_el.findtext("height", default="0"))
    # A zero/negative size would silently divide-by-zero during YOLO normalisation
    # (caught as a generic "parse failure" upstream, dropping the image with a
    # misleading message). Fail clearly instead.
    if width <= 0 or height <= 0:
        raise ValueError(f"non-positive image size {width}x{height} in {xml_path}")

    objects: list[tuple[str, int, int, int, int]] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", default="").strip().lower()
        bbox = obj.find("bndbox")
        if bbox is None:
            continue   # malformed annotation — skip this object
        xmin = int(float(bbox.findtext("xmin", default="0")))
        ymin = int(float(bbox.findtext("ymin", default="0")))
        xmax = int(float(bbox.findtext("xmax", default="0")))
        ymax = int(float(bbox.findtext("ymax", default="0")))
        objects.append((name, xmin, ymin, xmax, ymax))

    return width, height, objects


def remap_objects(
    objs: list[tuple[str, int, int, int, int]],
) -> list[tuple[int, int, int, int, int]]:
    """Apply REMAP to a list of raw VOC objects.

    Drops any object whose class name is not in REMAP (e.g. inductors,
    transistors, AC sources — see the REMAP comment block for the full list).

    Returns a list of (class_id, xmin, ymin, xmax, ymax) tuples ready for
    YOLO formatting.
    """
    mapped: list[tuple[int, int, int, int, int]] = []
    for name, xmin, ymin, xmax, ymax in objs:
        canonical = REMAP.get(name)
        if canonical is None:
            continue   # intentionally dropped — see REMAP comment block
        class_id = _CLASS_TO_ID[canonical]
        mapped.append((class_id, xmin, ymin, xmax, ymax))
    return mapped


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float into [lo, hi].  Defensive helper for YOLO normalisation."""
    return max(lo, min(hi, value))


def to_yolo_lines(
    width: int,
    height: int,
    mapped_objs: list[tuple[int, int, int, int, int]],
) -> list[str]:
    """Convert pixel bounding boxes to YOLO label-file lines.

    YOLO stores boxes as:
        <class_id>  <cx>  <cy>  <w>  <h>

    where cx, cy, w, h are normalised to [0, 1] by dividing by the image
    dimensions.  The centre (cx, cy) is the middle of the box, and w/h are
    the full box width/height — not half-widths.

    We clamp to [0, 1] as a defensive measure against annotations that have
    a pixel coordinate very slightly outside the image boundary (rounding
    artefacts in some annotation tools).
    """
    lines: list[str] = []
    for class_id, xmin, ymin, xmax, ymax in mapped_objs:
        # Clamp the CORNERS to [0,1] first, then derive centre/size from the clamped
        # corners. Clamping cx/cy/w/h independently (the old way) breaks for boxes
        # that overflow the image: e.g. a box from -5..105 px on a 100-px image
        # would keep cx=0.5 but clamp w 1.1->1.0, so centre+width no longer
        # reconstruct the (clamped) box. Corner-first keeps the box consistent.
        x0, x1 = _clamp(xmin / width), _clamp(xmax / width)
        y0, y1 = _clamp(ymin / height), _clamp(ymax / height)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        w, h = x1 - x0, y1 - y0
        # Six decimal places is standard for YOLO labels; more is noise.
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def split_drafters(
    drafter_ids: list[str],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[str]]:
    """Assign each drafter id to exactly one of {train, val, test}.

    The split is at the DRAFTER level (not the image level) to prevent data
    leakage: every image from one drafter goes to the same fold, so the model
    never trains and evaluates on images from the same person's handwriting.

    Allocation:
        n_test  = ceil(test_frac  * n_drafters)
        n_val   = ceil(val_frac   * n_drafters)
        n_train = remainder

    When there are fewer than 3 drafters the guarantees break down (you cannot
    give one each to train/val/test), but we still try our best by giving the
    remainder to train and clamping the others to at least 0.

    The drafter list is shuffled with a seeded RNG before allocation so the
    result is deterministic but not sorted.
    """
    ids = sorted(drafter_ids)   # sort first for a reproducible baseline
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)

    # Compute how many drafters go to each non-train split.
    n_test = min(math.ceil(test_frac * n), n)
    n_val  = min(math.ceil(val_frac  * n), n - n_test)
    n_train = n - n_test - n_val   # everything that's left goes to train

    # If the dataset is tiny and there aren't enough drafters for all three
    # splits to get at least one, put the remainder in train and leave the
    # others empty — callers should warn the user about this.
    test_ids  = ids[:n_test]
    val_ids   = ids[n_test : n_test + n_val]
    train_ids = ids[n_test + n_val :]

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def write_data_yaml(out_dir: Path) -> None:
    """Write the Ultralytics-compatible data.yaml into out_dir.

    Ultralytics YOLO expects:
        path:  absolute path to the dataset root
        train: relative path to train images
        val:   relative path to val images
        test:  relative path to test images
        nc:    number of classes (integer)
        names: list of class names in order (index = class id)

    We store this as a plain text YAML file written with the stdlib — no
    PyYAML dependency needed because the structure is simple.
    """
    names_inline = "[" + ", ".join(CLASSES) + "]"
    yaml_text = (
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {names_inline}\n"
    )
    (out_dir / "data.yaml").write_text(yaml_text)


# ---------------------------------------------------------------------------
# Image-extension helper
# ---------------------------------------------------------------------------

# CGHD images can have any of these extensions.  We try them in order.
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def _find_image(annotation_xml: Path) -> Path | None:
    """Given an annotation XML path, return the matching image path or None.

    The image lives in the sibling ``images/`` directory (at the same level
    as ``annotations/``) and shares the XML stem but has an image extension.
    """
    # annotations/ and images/ are siblings under the same drafter folder.
    images_dir = annotation_xml.parent.parent / "images"
    stem = annotation_xml.stem
    for ext in _IMAGE_EXTS:
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: parse arguments, run the full conversion pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert the CGHD hand-drawn circuit dataset into an "
            "Ultralytics YOLO training set, splitting by drafter to avoid "
            "data leakage."
        )
    )
    parser.add_argument(
        "--src",
        required=True,
        type=Path,
        help="CGHD source root folder containing drafter_* subfolders.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cghd_yolo"),
        help="Output folder for the YOLO dataset (default: cghd_yolo).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of DRAFTERS to put in val split (default: 0.15).",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.15,
        help="Fraction of DRAFTERS to put in test split (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible drafter shuffling (default: 0).",
    )
    # --copy-images / --no-copy-images: when --no-copy-images is used, we still
    # write all labels and data.yaml.  This is handy for fast tests that do not
    # need the actual image bytes.
    parser.add_argument(
        "--copy-images",
        action="store_true",
        default=True,
        dest="copy_images",
        help="Copy source images into the output tree (default: on).",
    )
    parser.add_argument(
        "--no-copy-images",
        action="store_false",
        dest="copy_images",
        help="Skip copying images; only write labels and data.yaml.",
    )
    args = parser.parse_args(argv)

    src: Path = args.src.resolve()
    out: Path = args.out.resolve()

    # -----------------------------------------------------------------------
    # 1. Discover drafter folders
    # -----------------------------------------------------------------------
    drafter_dirs: dict[str, Path] = {}
    for d in sorted(src.iterdir()):
        # We only care about folders whose names start with "drafter_".
        if d.is_dir() and d.name.startswith("drafter_"):
            drafter_id = d.name   # e.g. "drafter_3"
            drafter_dirs[drafter_id] = d

    if not drafter_dirs:
        raise SystemExit(
            f"No drafter_* folders found under {src}.  "
            "Check that --src points to the CGHD root."
        )

    # -----------------------------------------------------------------------
    # 2. Assign drafters to splits
    # -----------------------------------------------------------------------
    splits = split_drafters(
        list(drafter_dirs.keys()),
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )
    # Build the reverse map: drafter_id → split name
    drafter_to_split: dict[str, str] = {}
    for split_name, ids in splits.items():
        for did in ids:
            drafter_to_split[did] = split_name

    # -----------------------------------------------------------------------
    # 3. Create output directory tree
    # -----------------------------------------------------------------------
    for split_name in ("train", "val", "test"):
        (out / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 4. Process every annotation, write label files, (optionally) copy images
    # -----------------------------------------------------------------------
    # Accumulators for the final report.
    images_per_split: Counter[str] = Counter()
    kept_boxes_total = 0
    dropped_boxes_total = 0
    dropped_no_class = 0     # images dropped because they had 0 kept boxes
    boxes_per_class: Counter[str] = Counter()

    for drafter_id, drafter_dir in sorted(drafter_dirs.items()):
        split_name = drafter_to_split[drafter_id]
        annotations_dir = drafter_dir / "annotations"

        if not annotations_dir.exists():
            warnings.warn(
                f"No annotations/ subfolder in {drafter_dir} — skipping this drafter."
            )
            continue

        for xml_path in sorted(annotations_dir.glob("*.xml")):
            # --- parse the VOC annotation ---
            try:
                width, height, raw_objs = parse_voc(xml_path)
            except Exception as exc:
                warnings.warn(f"Could not parse {xml_path}: {exc} — skipping.")
                continue

            # --- track dropped boxes (classes not in REMAP) ---
            mapped_objs = remap_objects(raw_objs)
            dropped_boxes_total += len(raw_objs) - len(mapped_objs)

            # Skip this image entirely if it has no boxes from our 8 classes.
            if not mapped_objs:
                dropped_no_class += 1
                continue

            # --- find the matching image on disk ---
            img_path = _find_image(xml_path)
            if img_path is None and args.copy_images:
                warnings.warn(
                    f"Image not found for {xml_path.stem} — skipping sample."
                )
                continue

            # --- write the YOLO label file ---
            stem = xml_path.stem
            label_lines = to_yolo_lines(width, height, mapped_objs)
            label_out = out / "labels" / split_name / f"{stem}.txt"
            label_out.write_text("\n".join(label_lines) + "\n")

            # --- optionally copy the image ---
            if args.copy_images and img_path is not None:
                # Always store as .jpg in the output tree for consistency.
                img_out = out / "images" / split_name / f"{stem}.jpg"
                shutil.copy2(img_path, img_out)

            # --- accumulate stats ---
            images_per_split[split_name] += 1
            kept_boxes_total += len(mapped_objs)
            for class_id, *_ in mapped_objs:
                boxes_per_class[CLASSES[class_id]] += 1

    # -----------------------------------------------------------------------
    # 5. Write data.yaml
    # -----------------------------------------------------------------------
    write_data_yaml(out)

    # -----------------------------------------------------------------------
    # 6. Print a human-readable summary report
    # -----------------------------------------------------------------------
    print("\n=== CGHD → YOLO conversion complete ===\n")

    print("Drafter split:")
    for split_name in ("train", "val", "test"):
        ids = splits[split_name]
        print(f"  {split_name:5s}: {len(ids):3d} drafter(s) → {ids}")

    print(f"\nImages per split:")
    for split_name in ("train", "val", "test"):
        print(f"  {split_name:5s}: {images_per_split[split_name]}")

    print(f"\nBoxes kept  : {kept_boxes_total}")
    print(f"Boxes dropped (unmapped class): {dropped_boxes_total}")
    print(f"Images dropped (0 kept boxes) : {dropped_no_class}")

    print(f"\nPer-class box counts:")
    for cls in CLASSES:
        print(f"  {cls:20s}: {boxes_per_class[cls]}")

    print(f"\ndata.yaml written to {out / 'data.yaml'}")
    print(f"Dataset root: {out}\n")


if __name__ == "__main__":
    main()
