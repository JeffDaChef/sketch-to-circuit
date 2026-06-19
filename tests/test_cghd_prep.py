"""Tests for data_collection/cghd_prep.py — every expected value here is hand-checkable.

WHY THIS EXISTS
---------------
cghd_prep.py does several fiddly transformations: it reads XML, remaps class
names, converts pixel boxes to normalised floats, and splits a dataset by
drafter.  Each step is easy to get slightly wrong (off-by-one on the centre
coordinate, wrong index for a class, splitting an individual drafter across
train and val).  These tests pin down the exact correct output for small,
simple inputs so we catch any such mistake immediately.

Test strategy:
  - Unit tests for each pure function (parse_voc, remap_objects, to_yolo_lines,
    split_drafters) using hand-calculated expected values.
  - An end-to-end integration test that builds a tiny fake CGHD tree on disk,
    runs the full pipeline, and checks the output files.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from data_collection.cghd_prep import (
    CLASSES,
    REMAP,
    parse_voc,
    remap_objects,
    split_drafters,
    to_yolo_lines,
    write_data_yaml,
    main,
)




def _make_voc_xml(
    width: int,
    height: int,
    objects: list[tuple[str, int, int, int, int]],
) -> str:
    """Return a PASCAL-VOC XML string with the given image size and objects.

    Each item in ``objects`` is (class_name, xmin, ymin, xmax, ymax).
    This is the format that CGHD uses — we generate it here so the tests
    don't depend on any actual CGHD files being present.
    """
    obj_blocks = ""
    for name, xmin, ymin, xmax, ymax in objects:
        obj_blocks += (
            f"  <object>\n"
            f"    <name>{name}</name>\n"
            f"    <bndbox>\n"
            f"      <xmin>{xmin}</xmin>\n"
            f"      <ymin>{ymin}</ymin>\n"
            f"      <xmax>{xmax}</xmax>\n"
            f"      <ymax>{ymax}</ymax>\n"
            f"    </bndbox>\n"
            f"  </object>\n"
        )
    return (
        f'<annotation>\n'
        f'  <size>\n'
        f'    <width>{width}</width>\n'
        f'    <height>{height}</height>\n'
        f'    <depth>3</depth>\n'
        f'  </size>\n'
        f'{obj_blocks}'
        f'</annotation>\n'
    )




class TestParseVoc:
    """parse_voc reads a VOC XML file and returns (width, height, objects)."""

    def test_parses_image_size(self, tmp_path: Path):
        xml = _make_voc_xml(640, 480, [])
        xml_path = tmp_path / "sample.xml"
        xml_path.write_text(xml)

        width, height, objs = parse_voc(xml_path)

        assert width == 640
        assert height == 480

    def test_parses_single_object(self, tmp_path: Path):
        xml = _make_voc_xml(100, 200, [("resistor", 10, 20, 30, 60)])
        xml_path = tmp_path / "sample.xml"
        xml_path.write_text(xml)

        width, height, objs = parse_voc(xml_path)

        assert len(objs) == 1
        name, xmin, ymin, xmax, ymax = objs[0]
        assert name == "resistor"
        assert xmin == 10
        assert ymin == 20
        assert xmax == 30
        assert ymax == 60

    def test_parses_multiple_objects(self, tmp_path: Path):
        objects = [
            ("resistor",           10, 20, 30, 60),
            ("capacitor.polarized", 50, 50, 80, 90),
            ("gnd",                100, 150, 120, 180),
        ]
        xml = _make_voc_xml(300, 300, objects)
        xml_path = tmp_path / "multi.xml"
        xml_path.write_text(xml)

        width, height, objs = parse_voc(xml_path)

        assert width == 300
        assert height == 300
        assert len(objs) == 3
        assert objs[1][0] == "capacitor.polarized"
        assert objs[1][1] == 50

    def test_empty_annotation_has_no_objects(self, tmp_path: Path):
        xml = _make_voc_xml(200, 200, [])
        xml_path = tmp_path / "empty.xml"
        xml_path.write_text(xml)

        _, _, objs = parse_voc(xml_path)

        assert objs == []

    def test_zero_image_size_raises_clearly(self, tmp_path: Path):
        xml_path = tmp_path / "zerosize.xml"
        xml_path.write_text(_make_voc_xml(0, 0, [("resistor", 1, 1, 2, 2)]))
        with pytest.raises(ValueError, match="non-positive image size"):
            parse_voc(xml_path)




class TestRemapObjects:
    """remap_objects translates raw VOC class names to (class_id, box) tuples."""

    def test_resistor_maps_to_correct_id(self):
        objs = [("resistor", 0, 0, 10, 10)]
        result = remap_objects(objs)
        assert len(result) == 1
        class_id, *_ = result[0]
        assert class_id == CLASSES.index("resistor")

    def test_capacitor_polarized_and_unpolarized_both_map_to_capacitor(self):
        objs = [
            ("capacitor.polarized",   0, 0, 5, 5),
            ("capacitor.unpolarized", 5, 5, 10, 10),
        ]
        result = remap_objects(objs)
        assert len(result) == 2
        assert result[0][0] == result[1][0] == CLASSES.index("capacitor")

    def test_vss_maps_to_ground(self):
        objs = [("vss", 0, 0, 10, 10)]
        result = remap_objects(objs)
        assert len(result) == 1
        assert result[0][0] == CLASSES.index("ground")

    def test_voltage_battery_maps_to_voltage_source(self):
        objs = [("voltage.battery", 0, 0, 20, 20)]
        result = remap_objects(objs)
        assert len(result) == 1
        assert result[0][0] == CLASSES.index("voltage_source")

    def test_led_maps_to_diode(self):
        objs = [("diode.light_emitting", 0, 0, 15, 15)]
        result = remap_objects(objs)
        assert len(result) == 1
        assert result[0][0] == CLASSES.index("diode")

    def test_unmapped_classes_are_dropped(self):
        unmapped = [
            ("inductor",       0, 0, 10, 10),
            ("transistor.bjt", 10, 0, 20, 10),
            ("voltage.ac",     20, 0, 30, 10),
        ]
        result = remap_objects(unmapped)
        assert result == []

    def test_mix_of_kept_and_dropped(self):
        objs = [
            ("resistor",   0, 0, 10, 10),
            ("inductor",  10, 0, 20, 10),
            ("gnd",       20, 0, 30, 10),
        ]
        result = remap_objects(objs)
        assert len(result) == 2
        assert result[0][0] == CLASSES.index("resistor")
        assert result[1][0] == CLASSES.index("ground")

    def test_pixel_coordinates_are_preserved(self):
        objs = [("switch", 11, 22, 33, 44)]
        result = remap_objects(objs)
        _, xmin, ymin, xmax, ymax = result[0]
        assert (xmin, ymin, xmax, ymax) == (11, 22, 33, 44)




class TestToYoloLines:
    """to_yolo_lines converts pixel boxes to YOLO normalised-coordinate lines."""

    def test_single_box_hand_calculated(self):
        mapped = [(CLASSES.index("resistor"), 10, 20, 30, 60)]
        lines = to_yolo_lines(100, 200, mapped)

        assert len(lines) == 1
        parts = lines[0].split()
        assert parts[0] == "4"
        assert float(parts[1]) == pytest.approx(0.2)
        assert float(parts[2]) == pytest.approx(0.2)
        assert float(parts[3]) == pytest.approx(0.2)
        assert float(parts[4]) == pytest.approx(0.2)

    def test_full_image_box_gives_centre_05(self):
        mapped = [(0, 0, 0, 200, 100)]
        lines = to_yolo_lines(200, 100, mapped)
        parts = lines[0].split()
        assert float(parts[1]) == pytest.approx(0.5)
        assert float(parts[2]) == pytest.approx(0.5)
        assert float(parts[3]) == pytest.approx(1.0)
        assert float(parts[4]) == pytest.approx(1.0)

    def test_multiple_boxes_produce_multiple_lines(self):
        mapped = [
            (CLASSES.index("ground"),    0, 0,  50,  50),
            (CLASSES.index("junction"), 50, 50, 100, 100),
        ]
        lines = to_yolo_lines(200, 200, mapped)
        assert len(lines) == 2
        assert lines[0].startswith(str(CLASSES.index("ground")))
        assert lines[1].startswith(str(CLASSES.index("junction")))

    def test_empty_objects_gives_empty_list(self):
        assert to_yolo_lines(100, 100, []) == []

    def test_values_are_rounded_to_6_decimals(self):
        mapped = [(0, 0, 0, 1, 3)]
        lines = to_yolo_lines(3, 9, mapped)
        assert "." in lines[0]
        parts = lines[0].split()
        for part in parts[1:]:
            if "." in part:
                decimals = part.split(".")[1]
                assert len(decimals) <= 6, f"Too many decimals in '{part}'"

    def test_out_of_range_coords_are_clamped(self):
        mapped = [(0, -5, -5, 105, 105)]
        lines = to_yolo_lines(100, 100, mapped)
        parts = lines[0].split()
        for val in parts[1:]:
            assert 0.0 <= float(val) <= 1.0, f"Out-of-range value: {val}"

    def test_overflowing_box_stays_geometrically_consistent(self):
        mapped = [(0, -20, -20, 80, 80)]
        _, cx, cy, w, h = to_yolo_lines(100, 100, mapped)[0].split()
        assert float(cx) == pytest.approx(0.4) and float(w) == pytest.approx(0.8)
        assert float(cx) - float(w) / 2 == pytest.approx(0.0)
        assert float(cx) + float(w) / 2 == pytest.approx(0.8)




class TestSplitDrafters:
    """split_drafters assigns whole drafter ids to train/val/test, no leakage."""

    def _make_ids(self, n: int) -> list[str]:
        return [f"drafter_{i}" for i in range(n)]

    def test_all_drafters_assigned(self):
        ids = self._make_ids(10)
        splits = split_drafters(ids, val_frac=0.2, test_frac=0.2, seed=0)
        assigned = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        assert assigned == set(ids)

    def test_splits_are_disjoint(self):
        ids = self._make_ids(10)
        splits = split_drafters(ids, val_frac=0.2, test_frac=0.2, seed=0)
        train_s = set(splits["train"])
        val_s   = set(splits["val"])
        test_s  = set(splits["test"])
        assert train_s.isdisjoint(val_s),   "train and val overlap!"
        assert train_s.isdisjoint(test_s),  "train and test overlap!"
        assert val_s.isdisjoint(test_s),    "val and test overlap!"

    def test_correct_counts_for_10_drafters(self):
        ids = self._make_ids(10)
        splits = split_drafters(ids, val_frac=0.2, test_frac=0.2, seed=0)
        assert len(splits["test"])  == 2
        assert len(splits["val"])   == 2
        assert len(splits["train"]) == 6

    def test_deterministic_with_same_seed(self):
        ids = self._make_ids(12)
        splits_a = split_drafters(ids, val_frac=0.15, test_frac=0.15, seed=42)
        splits_b = split_drafters(ids, val_frac=0.15, test_frac=0.15, seed=42)
        assert splits_a == splits_b

    def test_different_seeds_give_different_splits(self):
        ids = self._make_ids(20)
        splits_0 = split_drafters(ids, val_frac=0.15, test_frac=0.15, seed=0)
        splits_1 = split_drafters(ids, val_frac=0.15, test_frac=0.15, seed=99)
        assert splits_0["test"] != splits_1["test"]

    def test_three_drafters_each_gets_at_least_one(self):
        ids = self._make_ids(3)
        splits = split_drafters(ids, val_frac=0.15, test_frac=0.15, seed=0)
        assert len(splits["test"])  >= 1
        assert len(splits["val"])   >= 1
        assert len(splits["train"]) >= 1




class TestWriteDataYaml:
    """write_data_yaml produces the Ultralytics-compatible YAML file."""

    def test_yaml_contains_nc_8(self, tmp_path: Path):
        write_data_yaml(tmp_path)
        text = (tmp_path / "data.yaml").read_text()
        assert "nc: 8" in text

    def test_yaml_contains_all_8_class_names(self, tmp_path: Path):
        write_data_yaml(tmp_path)
        text = (tmp_path / "data.yaml").read_text()
        for cls in CLASSES:
            assert cls in text, f"'{cls}' missing from data.yaml"

    def test_yaml_has_correct_train_val_test_keys(self, tmp_path: Path):
        write_data_yaml(tmp_path)
        text = (tmp_path / "data.yaml").read_text()
        assert "train: images/train" in text
        assert "val: images/val"     in text
        assert "test: images/test"   in text

    def test_yaml_path_is_absolute(self, tmp_path: Path):
        write_data_yaml(tmp_path)
        text = (tmp_path / "data.yaml").read_text()
        path_line = next(l for l in text.splitlines() if l.startswith("path:"))
        assert path_line.split(":", 1)[1].strip().startswith("/")




def _build_fake_cghd(root: Path) -> None:
    """Create a tiny fake CGHD tree under root with 4 drafters.

    drafter_0: 2 images, all boxes mappable (resistor + gnd)
    drafter_1: 1 image,  all boxes mappable (capacitor.polarized)
    drafter_2: 2 images: one with a mix (resistor + inductor → only resistor kept),
                         one with ONLY unmapped boxes (voltage.ac → 0 kept → dropped)
    drafter_3: 1 image,  junction + text (both kept)

    This exercises: normal path, partial-drop path, full-drop path.
    """
    for d_id, samples in [
        ("drafter_0", [
            (200, 200, [("resistor", 10, 10, 50, 50), ("gnd", 60, 60, 90, 90)]),
            (300, 150, [("resistor", 5, 5, 40, 40)]),
        ]),
        ("drafter_1", [
            (100, 100, [("capacitor.polarized", 20, 20, 80, 80)]),
        ]),
        ("drafter_2", [
            (400, 300, [("resistor", 0, 0, 50, 50), ("inductor", 60, 60, 100, 100)]),
            (200, 200, [("voltage.ac", 0, 0, 100, 100)]),
        ]),
        ("drafter_3", [
            (500, 400, [("junction", 10, 10, 30, 30), ("text", 40, 40, 100, 80)]),
        ]),
    ]:
        ann_dir = root / d_id / "annotations"
        img_dir = root / d_id / "images"
        ann_dir.mkdir(parents=True)
        img_dir.mkdir(parents=True)

        for i, (w, h, objs) in enumerate(samples):
            stem = f"{d_id}_img{i}"
            xml  = _make_voc_xml(w, h, objs)
            (ann_dir / f"{stem}.xml").write_text(xml)
            (img_dir / f"{stem}.jpg").write_bytes(b"JFIF_FAKE")


class TestEndToEnd:
    """Run the full pipeline on a tiny fake CGHD tree and check the output."""

    def test_end_to_end(self, tmp_path: Path):
        src = tmp_path / "cghd_src"
        out = tmp_path / "cghd_yolo"
        _build_fake_cghd(src)

        main([
            "--src", str(src),
            "--out", str(out),
            "--val-frac",  "0.25",
            "--test-frac", "0.25",
            "--seed", "7",
            "--no-copy-images",
        ])

        yaml_path = out / "data.yaml"
        assert yaml_path.exists(), "data.yaml was not created"
        yaml_text = yaml_path.read_text()
        assert "nc: 8" in yaml_text
        for cls in CLASSES:
            assert cls in yaml_text, f"'{cls}' missing from data.yaml"

        all_label_stems = set()
        for split_name in ("train", "val", "test"):
            label_dir = out / "labels" / split_name
            if label_dir.exists():
                for lf in label_dir.glob("*.txt"):
                    all_label_stems.add(lf.stem)

        assert "drafter_2_img1" not in all_label_stems, (
            "Label file was created for an image that had only unmapped classes!"
        )

        expected_kept_stems = {
            "drafter_0_img0",
            "drafter_0_img1",
            "drafter_1_img0",
            "drafter_2_img0",
            "drafter_3_img0",
        }
        assert expected_kept_stems == all_label_stems, (
            f"Unexpected label stems.\n"
            f"  Expected: {sorted(expected_kept_stems)}\n"
            f"  Got:      {sorted(all_label_stems)}"
        )

        drafter_to_splits: dict[str, set[str]] = {}
        for split_name in ("train", "val", "test"):
            label_dir = out / "labels" / split_name
            if not label_dir.exists():
                continue
            for lf in label_dir.glob("*.txt"):
                drafter_id = "_".join(lf.stem.split("_")[:2])
                drafter_to_splits.setdefault(drafter_id, set()).add(split_name)

        for did, split_set in drafter_to_splits.items():
            assert len(split_set) == 1, (
                f"Drafter {did} appears in multiple splits: {split_set}"
            )

        drafter_2_label = None
        for split_name in ("train", "val", "test"):
            candidate = out / "labels" / split_name / "drafter_2_img0.txt"
            if candidate.exists():
                drafter_2_label = candidate
                break
        assert drafter_2_label is not None, "drafter_2_img0.txt not found in any split"
        lines = [l for l in drafter_2_label.read_text().strip().splitlines() if l]
        assert len(lines) == 1, f"Expected 1 kept box for drafter_2_img0, got {len(lines)}"
        assert lines[0].startswith("4 "), "Box should have class_id 4 (resistor)"
