"""Tests for training/evaluate.py's pure metric summariser.

The training scripts can't run here (no GPU/dataset/ultralytics), but the metric
extraction is pure logic and was subtly buggy: per-class AP arrays in ultralytics
are aligned with `ap_class_index` (classes PRESENT in the split), not with the
global class id. Indexing by global id silently mislabels scores — or runs off the
end — when a rare class is absent from a drafter-split test set. These tests pin
the correct behaviour with a fake results object, no ultralytics needed.
"""

from types import SimpleNamespace

import pytest

from training.evaluate import summarize_results


def _fake_results(map50, map50_95, names, ap_class_index, ap50):
    box = SimpleNamespace(map50=map50, map=map50_95,
                          ap_class_index=ap_class_index, ap50=ap50)
    return SimpleNamespace(box=box, names=names)


def test_per_class_scores_map_to_the_right_names():
    names = {0: "capacitor", 1: "diode", 2: "resistor"}
    res = _fake_results(0.8, 0.6, names, [0, 1, 2], [0.9, 0.7, 0.95])
    s = summarize_results(res)
    assert s["map50"] == pytest.approx(0.8)
    assert s["map50_95"] == pytest.approx(0.6)
    assert s["per_class"] == {"capacitor": pytest.approx(0.9),
                              "diode": pytest.approx(0.7),
                              "resistor": pytest.approx(0.95)}
    assert s["missing_classes"] == []


def test_absent_class_does_not_misalign_scores():
    names = {0: "capacitor", 1: "diode", 2: "resistor"}
    res = _fake_results(0.5, 0.4, names, [0, 2], [0.90, 0.80])
    s = summarize_results(res)
    assert s["per_class"] == {"capacitor": pytest.approx(0.90),
                              "resistor": pytest.approx(0.80)}
    assert "diode" not in s["per_class"]
    assert s["missing_classes"] == ["diode"]
