"""Tests for metrics/extractor_ablation.py.

The substantive check is the ablation's whole point: on the looped layout that
exposed the old extractor's overfitting, the skeleton-graph redesign must do
strictly better. Kept small (few seeds, two layouts) because rendering is slow;
the full sweep is what `python -m metrics.extractor_ablation` is for.
"""

import pytest

from metrics.extractor_ablation import EXTRACTORS, _totals, run_ablation
from data_collection.synthetic import _parallel_bank, _series_loop


def test_redesign_beats_baseline_on_the_loop_layout():
    # series_loop is the layout the blob-proximity baseline could not handle.
    results = run_ablation(n_seeds=4, makers={"series_loop": _series_loop})
    base = results["series_loop"]["baseline"]
    new = results["series_loop"]["skeleton_graph"]
    assert new[0] > base[0]                      # strictly more correct
    assert new[0] == new[1]                       # and the redesign gets them all


def test_both_handle_an_easy_layout():
    # On the parallel bank both extractors should succeed — the redesign didn't
    # regress the cases the baseline already handled.
    results = run_ablation(n_seeds=3, makers={"parallel_bank": _parallel_bank})
    assert results["parallel_bank"]["baseline"][0] == 3
    assert results["parallel_bank"]["skeleton_graph"][0] == 3


def test_structure_and_totals():
    results = run_ablation(n_seeds=2, makers={"parallel_bank": _parallel_bank})
    assert set(results["parallel_bank"]) == set(EXTRACTORS)
    totals = _totals(results)
    assert totals["skeleton_graph"][1] == 2      # one layout × 2 seeds
