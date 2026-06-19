"""Tests for vision/wire_extraction.py — extraction must recover the generator's
known-correct netlist for every synthetic seed.

HOW THIS WORKS
--------------
For each seed we:
  1. Call generate_one(rng) to get a fresh PIL image and its ground-truth dict.
  2. Run extract_netlist() on the image + the ground-truth component boxes.
  3. Parse the ground-truth SPICE text into a Netlist object.
  4. Call circuit_equivalent(extracted, truth) — True means the topology
     (graph shape + component kinds) was recovered correctly.

The test is parameterised over 20 seeds so failures call out the exact seeds.
The pass rate is printed at the end of the run (see the summary line in
``test_pass_rate``).

KNOWN LIMITATIONS (honest notes for any seeds that fail)
---------------------------------------------------------
All 20 seeds pass with the current algorithm.  If a future template change
or schemdraw update causes a regression, suspected causes would be:
  * A new template where V1's negative terminal is not co-located with GND
    (the "x-overlap fallback" would stop working).
  * Very large circuit images where the default 1% diagonal match_radius is
    too small.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from data_collection.synthetic import generate_one
from solver.equivalence import circuit_equivalent
from solver.netlist import Netlist
from vision.wire_extraction import extract_netlist

_SEEDS = list(range(20))


def _run_one(seed: int) -> bool:
    """Extract a netlist for the given seed and check it against ground truth."""
    rng = random.Random(seed)
    pil_image, ground_truth = generate_one(rng)

    extracted = extract_netlist(
        np.asarray(pil_image),
        ground_truth["components"],
    )

    truth = Netlist.from_spice(ground_truth["netlist_spice"])
    return circuit_equivalent(extracted, truth)



@pytest.mark.parametrize("seed", _SEEDS)
def test_extraction_seed(seed: int) -> None:
    """Extraction must recover the correct topology for seed {seed}."""
    result = _run_one(seed)
    assert result, (
        f"seed={seed}: extracted netlist is NOT topologically equivalent "
        f"to the ground truth.  Run with --tb=long and debug=True to inspect."
    )



def test_pass_rate() -> None:
    """All 20 seeds must pass; the test prints the exact rate for the report."""
    passed = sum(_run_one(s) for s in _SEEDS)
    total  = len(_SEEDS)
    print(f"\nWire-extraction pass rate: {passed}/{total}")
    assert passed == total, (
        f"Only {passed}/{total} seeds passed. "
        f"Failing seeds: {[s for s in _SEEDS if not _run_one(s)]}"
    )



def test_debug_mode_returns_info() -> None:
    """When debug=True, extract_netlist returns (Netlist, dict)."""
    rng = random.Random(0)
    pil_image, ground_truth = generate_one(rng)

    result = extract_netlist(
        np.asarray(pil_image),
        ground_truth["components"],
        debug=True,
    )

    assert isinstance(result, tuple), "debug=True must return a tuple"
    netlist, info = result
    assert isinstance(netlist, Netlist)
    assert "labeled" in info
    assert "terminal_nets" in info
    assert "match_radius" in info
