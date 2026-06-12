"""Tests for metrics/extraction_accuracy.py — the end-to-end extraction metric.

HOW THIS WORKS
--------------
We call ``evaluate_extraction(count=40, seed=0)`` and check three things:

1. **Correctness of the headline number**: total == 40 and accuracy >= 0.95.
   (We expect ~1.0 currently; the margin of 0.05 guards against rare
   platform-specific rendering differences.)

2. **Internal consistency of the bookkeeping**: the sum of per-template and
   per-component-count tallies must add up to 40 for both "correct" and "total".

3. **Determinism**: calling evaluate_extraction twice with the same (count, seed)
   must return identical dicts, proving the single-RNG design works.
"""

from __future__ import annotations

from metrics.extraction_accuracy import evaluate_extraction


# ---------------------------------------------------------------------------
# Shared fixture — run once, share across tests
# ---------------------------------------------------------------------------

# We deliberately call this at module level so the 40-circuit run only happens
# once (pytest re-uses the module), keeping test-suite time low.
_RESULT = evaluate_extraction(count=40, seed=0)


# ---------------------------------------------------------------------------
# Headline number
# ---------------------------------------------------------------------------

def test_total_is_correct() -> None:
    """evaluate_extraction(40, 0) must process exactly 40 circuits."""
    assert _RESULT["total"] == 40


def test_accuracy_threshold() -> None:
    """Accuracy must be >= 95% — we expect ~100% with the current algorithm."""
    assert _RESULT["accuracy"] >= 0.95, (
        f"Accuracy {_RESULT['accuracy']:.1%} is below the 95% threshold.  "
        f"Failing circuits: {_RESULT['failures']}"
    )


def test_correct_matches_template_sum() -> None:
    """The top-level 'correct' count must equal the sum of per-template corrects."""
    template_correct_sum = sum(c for c, _ in _RESULT["by_template"].values())
    assert _RESULT["correct"] == template_correct_sum, (
        f"'correct'={_RESULT['correct']} does not match "
        f"sum-of-template-corrects={template_correct_sum}"
    )


# ---------------------------------------------------------------------------
# Bookkeeping consistency
# ---------------------------------------------------------------------------

def test_by_template_total_sums_to_count() -> None:
    """Sum of all per-template 'total' values must equal 40."""
    total_sum = sum(t for _, t in _RESULT["by_template"].values())
    assert total_sum == 40, (
        f"by_template totals sum to {total_sum}, expected 40"
    )


def test_by_component_count_total_sums_to_count() -> None:
    """Sum of all per-component-count 'total' values must equal 40."""
    total_sum = sum(t for _, t in _RESULT["by_component_count"].values())
    assert total_sum == 40, (
        f"by_component_count totals sum to {total_sum}, expected 40"
    )


def test_by_component_count_correct_sums_match() -> None:
    """Sum of per-component-count 'correct' values must equal the headline correct."""
    comp_correct_sum = sum(c for c, _ in _RESULT["by_component_count"].values())
    assert _RESULT["correct"] == comp_correct_sum, (
        f"'correct'={_RESULT['correct']} does not match "
        f"sum-of-component-count-corrects={comp_correct_sum}"
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism() -> None:
    """Calling evaluate_extraction twice with the same args must give equal dicts."""
    result_again = evaluate_extraction(count=40, seed=0)
    # Compare the numeric fields directly.
    assert result_again["total"]    == _RESULT["total"]
    assert result_again["correct"]  == _RESULT["correct"]
    assert result_again["accuracy"] == _RESULT["accuracy"]
    # The breakdown dicts must also match exactly.
    assert result_again["by_template"]        == _RESULT["by_template"]
    assert result_again["by_component_count"] == _RESULT["by_component_count"]
    assert result_again["failures"]           == _RESULT["failures"]
