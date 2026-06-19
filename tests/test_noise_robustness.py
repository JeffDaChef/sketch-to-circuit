"""Tests for metrics/noise_robustness.py.

Kept fast: the corruption functions are pure array ops (tested directly), and the
end-to-end study is exercised on a tiny base set so it doesn't render dozens of
images. The expensive full sweep is what `python -m metrics.noise_robustness` is
for, not the test suite.
"""

import numpy as np
import pytest

from metrics.noise_robustness import (
    CORRUPTIONS,
    Corruption,
    blur,
    gaussian_noise,
    make_base_set,
    run_study,
    score,
    speckle,
)


@pytest.fixture
def img():
    return np.full((40, 50, 3), 128, dtype=np.uint8)


@pytest.fixture(scope="module")
def small_bases():
    return make_base_set(count=3, seed=0)


def test_severity_zero_is_a_no_op(img):
    for fn in (blur, gaussian_noise, speckle):
        assert np.array_equal(fn(img, 0.0, seed=0), img), fn.__name__


def test_corruptions_preserve_shape_and_dtype(img):
    for fn, sev in ((blur, 3.0), (gaussian_noise, 40.0), (speckle, 0.02)):
        out = fn(img, sev, seed=1)
        assert out.shape == img.shape and out.dtype == np.uint8


def test_corruptions_are_deterministic_per_seed(img):
    assert np.array_equal(gaussian_noise(img, 30.0, seed=7), gaussian_noise(img, 30.0, seed=7))
    assert np.array_equal(speckle(img, 0.05, seed=7), speckle(img, 0.05, seed=7))
    assert not np.array_equal(gaussian_noise(img, 30.0, seed=7), gaussian_noise(img, 30.0, seed=8))


def test_speckle_is_monotonic_superset(img):
    low = speckle(img, 0.02, seed=3)
    high = speckle(img, 0.06, seed=3)
    changed_low = np.any(low != img, axis=-1)
    changed_high = np.any(high != img, axis=-1)
    assert changed_high.sum() > changed_low.sum()
    assert np.all(changed_high[changed_low])


def test_more_blur_changes_more(img):
    edged = img.copy()
    edged[:, :25] = 30
    diff_small = np.abs(blur(edged, 1.0, 0).astype(int) - edged).sum()
    diff_big = np.abs(blur(edged, 4.0, 0).astype(int) - edged).sum()
    assert diff_big > diff_small > 0


def test_study_structure_and_clean_baseline_is_perfect(small_bases):
    one = [Corruption("blur", blur, (0.0, 3.0), "blur test")]
    study = run_study(bases=small_bases, corruptions=one)
    assert set(study) == {"blur"}
    data = study["blur"]
    assert len(data["levels"]) == len(data["accuracy"]) == 2
    assert data["accuracy"][0] == pytest.approx(100.0)


def test_score_returns_correct_total(small_bases):
    blur_corr = next(c for c in CORRUPTIONS if c.name == "blur")
    correct, total = score(small_bases, blur_corr, 0.0)
    assert total == 3 and correct == 3
