"""Tests for solver/ac.py — frequency-domain (phasor) analysis.

Anchored to the closed-form transfer functions you derive on paper: an RC
low-pass is H = 1/(1+jωRC), a series-RLC band-pass peaks at 1/(2π√(LC)). Matching
those at specific frequencies pins the complex-MNA stamping (a wrong reactance
sign or a real-vs-complex slip would fail these immediately).
"""

import cmath
import math

import pytest

from solver.ac import (
    ACError,
    logspace,
    operating_point,
    small_signal_ac,
    small_signal_transfer_function,
    solve_ac,
    transfer_function,
)
from solver.netlist import Netlist
from solver.nonlinear import SILICON


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


# --- small-signal AC (diodes linearised at the DC operating point) -----------

def _diode_bias_circuit(ibias):
    """vin (AC) -> D1 -> mid -> C -> 0 ; I_bias (mid->0) sets the diode current."""
    n = Netlist()
    n.add("V", "vin", 0.0, "in", "0")
    n.add("D", "D1", 0.0, "in", "mid")
    n.add("C", "C1", 1e-6, "mid", "0")
    n.add("I", "Ibias", ibias, "mid", "0")
    return n


def test_small_signal_resistance_matches_26mv_over_i():
    # The defining small-signal result: r_d = n·V_t / I_D.
    for ibias in (1e-4, 1e-3, 5e-3):
        net = _diode_bias_circuit(ibias)
        vd, rd = operating_point(net)["D1"]
        from solver.nonlinear import solve_nonlinear
        i_d = solve_nonlinear(net).branch_currents["D1"]
        assert rd == pytest.approx(SILICON.n * SILICON.V_t / i_d, rel=1e-3)


def test_reverse_biased_diode_is_high_resistance():
    # Diode reverse-biased -> tiny slope -> r_d is enormous (≈ open circuit).
    n = Netlist()
    n.add("V", "vin", 0.0, "in", "0")
    n.add("D", "D1", 0.0, "in", "mid")          # anode 'in'(0V), cathode 'mid'
    n.add("R", "Rpull", 1000.0, "mid", "0")     # holds mid near 0 -> diode ~0V/reverse
    n.add("I", "Irev", 1e-6, "0", "mid")        # pulls mid slightly positive -> reverse
    _, rd = operating_point(n)["D1"]
    assert rd > 1e6                              # effectively open


def test_bias_tunable_lowpass_cutoff_tracks_bias():
    # The headline: a diode + cap is a low-pass whose cutoff f_c = 1/(2π r_d C)
    # moves with the bias current. Check |H| ≈ -3 dB at the predicted cutoff, and
    # that 10x bias gives ~10x cutoff.
    fcs = []
    for ibias in (1e-4, 1e-3):
        net = _diode_bias_circuit(ibias)
        _, rd = operating_point(net)["D1"]
        fc = 1.0 / (2 * math.pi * rd * 1e-6)
        fcs.append(fc)
        sweep = small_signal_transfer_function(net, [fc], "mid", "vin")
        assert sweep["mag"][0] == pytest.approx(1 / math.sqrt(2), rel=0.02)   # -3 dB at f_c
    assert fcs[1] / fcs[0] == pytest.approx(10.0, rel=0.05)    # 10x bias -> 10x cutoff


def test_small_signal_reduces_to_linear_without_diodes():
    # With no diodes, small-signal AC == ordinary AC (same transfer function).
    n, _, _ = rc_lowpass()
    a = small_signal_transfer_function(n, [100.0, 1000.0], "out", "Vin")
    b = transfer_function(n, [100.0, 1000.0], "out", "Vin")
    for ha, hb in zip(a["H"], b["H"]):
        assert ha == pytest.approx(hb, rel=1e-9)


def test_small_signal_ac_returns_complex_phasors():
    res = small_signal_ac(_diode_bias_circuit(1e-3), 1000.0)
    assert isinstance(res.gain("mid"), complex)
