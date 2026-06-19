"""Tests for metrics/extraction_accuracy.py — the end-to-end extraction metric.

We call ``evaluate_extraction(count=40, seed=0)`` once (a module-scoped fixture so
a rendering failure is a clean test failure, not a collection error) and check:

1. **The headline number** is exactly right — all 40 circuits recover correctly
   (the pipeline is deterministic; the official figure is 200/200, so anything
   below 100% at count=40 is a real regression, not noise).
2. **Bookkeeping consistency** — per-template and per-count tallies reconcile.
3. **The difficulty breakdown actually spans >1 component count** (else the
   "by component count" curve has silently collapsed to a single bucket).
4. **Determinism** — same (count, seed) -> identical dicts.
"""

from __future__ import annotations

import pytest

from metrics.extraction_accuracy import evaluate_extraction


@pytest.fixture(scope="module")
def result() -> dict:
    return evaluate_extraction(count=40, seed=0)



def test_total_is_correct(result) -> None:
    assert result["total"] == 40


def test_accuracy_is_perfect(result) -> None:
    assert result["correct"] == 40, f"regressed; failures: {result['failures']}"
    assert result["accuracy"] == 1.0



def test_correct_matches_template_sum(result) -> None:
    assert result["correct"] == sum(c for c, _ in result["by_template"].values())


def test_by_template_total_sums_to_count(result) -> None:
    assert sum(t for _, t in result["by_template"].values()) == 40


def test_by_component_count_total_sums_to_count(result) -> None:
    assert sum(t for _, t in result["by_component_count"].values()) == 40


def test_difficulty_breakdown_spans_multiple_sizes(result) -> None:
    assert len(result["by_component_count"]) > 1



def test_determinism(result) -> None:
    again = evaluate_extraction(count=40, seed=0)
    assert again["total"] == result["total"]
    assert again["correct"] == result["correct"]
    assert again["accuracy"] == result["accuracy"]
    assert again["by_template"] == result["by_template"]
    assert again["by_component_count"] == result["by_component_count"]
    assert again["failures"] == result["failures"]
