"""Tests for solver/ac.py — frequency-domain (phasor) analysis.

Anchored to the closed-form transfer functions you derive on paper: an RC
low-pass is H = 1/(1+jωRC), a series-RLC band-pass peaks at 1/(2π√(LC)). Matching
those at specific frequencies pins the complex-MNA stamping (a wrong reactance
sign or a real-vs-complex slip would fail these immediately).
"""

import cmath
import math

import pytest

from solver.ac import ACError, logspace, solve_ac, transfer_function
from solver.netlist import Netlist


def rc_lowpass(R=1000.0, C=1e-6):
    n = Netlist()
    n.add("V", "Vin", 1, "in", "0")
    n.add("R", "R1", R, "in", "out")
    n.add("C", "C1", C, "out", "0")
    return n, R, C


def test_rc_lowpass_matches_closed_form():
    n, R, C = rc_lowpass()
    for f in (10.0, 159.155, 1000.0, 5000.0):
        h = solve_ac(n, f).gain("out")              # Vin=1 -> H = V(out)
        analytic = 1.0 / (1.0 + 1j * 2 * math.pi * f * R * C)
        assert h == pytest.approx(analytic, rel=1e-6)


def test_rc_cutoff_is_minus_3db_and_minus_45_deg():
    n, R, C = rc_lowpass()
    fc = 1.0 / (2 * math.pi * R * C)
    h = solve_ac(n, fc).gain("out")
    assert abs(h) == pytest.approx(1 / math.sqrt(2), rel=1e-6)        # -3 dB
    assert math.degrees(cmath.phase(h)) == pytest.approx(-45.0, abs=1e-6)


def test_lowpass_passes_dc_blocks_high_freq():
    n, R, C = rc_lowpass()
    fc = 1.0 / (2 * math.pi * R * C)
    assert abs(solve_ac(n, fc / 100).gain("out")) > 0.99             # well below cutoff: ~unity
    assert abs(solve_ac(n, fc * 100).gain("out")) < 0.02            # well above: strongly attenuated


def test_first_order_rolloff_is_20db_per_decade():
    n, R, C = rc_lowpass()
    fc = 1.0 / (2 * math.pi * R * C)
    # A decade above another, both well past cutoff: magnitude ratio ≈ 10x (20 dB).
    hi = abs(solve_ac(n, fc * 100).gain("out"))
    lo = abs(solve_ac(n, fc * 1000).gain("out"))
    assert hi / lo == pytest.approx(10.0, rel=0.05)


def test_series_rlc_bandpass_peaks_at_resonance():
    # Output across R in a series V-L-C-R loop peaks (|H|=1) at f0 = 1/(2π√(LC)).
    L, C = 10e-3, 1e-6
    f0 = 1.0 / (2 * math.pi * math.sqrt(L * C))
    n = Netlist()
    n.add("V", "Vin", 1, "in", "0")
    n.add("L", "L1", L, "in", "a")
    n.add("C", "C1", C, "a", "b")
    n.add("R", "R1", 100.0, "b", "0")
    at_res = abs(solve_ac(n, f0).gain("b"))
    below = abs(solve_ac(n, f0 / 10).gain("b"))
    above = abs(solve_ac(n, f0 * 10).gain("b"))
    assert at_res == pytest.approx(1.0, abs=1e-6)   # L and C cancel -> all of Vin across R
    assert below < 0.5 and above < 0.5              # rolled off either side of resonance


def test_inductor_is_short_at_low_freq_open_at_high():
    # A single inductor from in->out->gnd with a series R: at very low f the
    # inductor passes (V(out)~0 since it shorts to gnd), at high f it blocks (V(out)~Vin).
    n = Netlist()
    n.add("V", "Vin", 1, "in", "0")
    n.add("R", "R1", 1000.0, "in", "out")
    n.add("L", "L1", 1e-3, "out", "0")
    assert abs(solve_ac(n, 1.0).gain("out")) < 0.05        # low f: inductor ~short to gnd
    assert abs(solve_ac(n, 1e7).gain("out")) > 0.95        # high f: inductor ~open


def test_transfer_function_sweep_structure():
    n, _, _ = rc_lowpass()
    sweep = transfer_function(n, logspace(10, 1e5, points_per_decade=10), "out", "Vin")
    assert len(sweep["freq"]) == len(sweep["mag"]) == len(sweep["phase_deg"])
    # mag_db must be 20*log10(mag).
    for m, db in zip(sweep["mag"], sweep["mag_db"]):
        assert db == pytest.approx(20 * math.log10(m))


def test_errors():
    n, _, _ = rc_lowpass()
    with pytest.raises(ACError, match="frequency must be"):
        solve_ac(n, 0.0)
    nd = Netlist(); nd.add("V", "V1", 1, "a", "0"); nd.add("D", "D1", 0.0, "a", "0")
    with pytest.raises(ACError, match="non-linear"):
        solve_ac(nd, 100.0)
    with pytest.raises(ACError, match="not a voltage source"):
        transfer_function(n, [10, 100], "out", "R1")
