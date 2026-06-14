"""Tests for metrics/difficulty.py (accuracy vs circuit size).

Kept fast: small seed counts and a single template. The point is to pin the two
claims the study rests on — small circuits extract perfectly, and the sweep
machinery is sound — not to re-run the full multi-minute study.
"""

from metrics.difficulty import accuracy_at_size, run_study
from data_collection.synthetic import _parallel_bank, _series_divider


def test_small_circuits_extract_perfectly():
    # The honest baseline: at small sizes every family is 100%.
    c, n = accuracy_at_size(_series_divider, k=3, n_seeds=4)
    assert c == n == 4


def test_divider_holds_at_ten_components():
    # Regression guard for the label-placement fix: the vertical divider used to
    # collapse to 0% by ~8 components because value labels crowded the wiring and
    # got erased. It must now stay perfect at 10.
    c, n = accuracy_at_size(_series_divider, k=10, n_seeds=6)
    assert c == n == 6


def test_run_study_structure():
    study = run_study(n_seeds=2, sizes=(2, 4), templates={"parallel_bank": _parallel_bank})
    assert set(study) == {"parallel_bank"}
    assert set(study["parallel_bank"]) == {2, 4}
    assert study["parallel_bank"][2] == 100.0          # small -> perfect
