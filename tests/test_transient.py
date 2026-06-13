"""Tests for solver/transient.py.

The anchor test compares the simulator against the *analytic* RC charging curve
v(t) = V·(1 − e^(−t/RC)) — the formula you derive on paper. If the companion
model's sign or the integration is wrong, this curve comes out wrong, so it's a
strong correctness check, not just a smoke test.
"""

import math

import pytest

from solver.netlist import Netlist
from solver.transient import TransientError, sine, solve_transient


def rc_circuit(R="1k", C=1e-3, V="5"):
    """5 V source -> R -> node 'mid' -> C -> ground. τ = R·C."""
    n = Netlist()
    n.add("V", "V1", V, "in", "0")
    n.add("R", "R1", R, "in", "mid")
    n.add("C", "C1", C, "mid", "0")
    return n


def test_capacitor_charges_toward_source_voltage():
    # Uncharged cap, 5 V source: V(mid) should climb from ~0 to ~5 V, monotonically.
    res = solve_transient(rc_circuit(), t_stop=5.0, dt=0.005)
    mid = res.series("mid")
    assert mid[0] == pytest.approx(0.0, abs=1e-9)          # starts uncharged (a short)
    assert mid[-1] == pytest.approx(5.0, rel=0.02)         # settles at the source voltage
    assert all(b >= a - 1e-9 for a, b in zip(mid, mid[1:]))  # never decreases


def test_matches_analytic_rc_curve():
    # τ = 1k · 1mF = 1 s. Compare against V·(1 − e^(−t/τ)) at every sample.
    R, C, V = 1000.0, 1e-3, 5.0
    tau = R * C
    res = solve_transient(rc_circuit(R, C, str(int(V))), t_stop=5.0, dt=0.001)
    for t, v in zip(res.times, res.series("mid")):
        analytic = V * (1 - math.exp(-t / tau))
        assert v == pytest.approx(analytic, abs=0.02)      # backward-Euler lags slightly


def test_smaller_dt_is_more_accurate():
    # Backward-Euler error shrinks with dt -> the fine run must beat the coarse run.
    R, C, V, tau = 1000.0, 1e-3, 5.0, 1.0
    target = V * (1 - math.exp(-1.0 / tau))                # analytic value at t = 1 s
    err = {}
    for dt in (0.1, 0.005):
        res = solve_transient(rc_circuit(R, C, "5"), t_stop=1.0, dt=dt)
        err[dt] = abs(res.series("mid")[-1] - target)
    assert err[0.005] < err[0.1]


def test_nonzero_initial_condition_discharges():
    # Pre-charge the cap ABOVE the source: it should fall toward the source voltage.
    res = solve_transient(rc_circuit(V="2"), t_stop=5.0, dt=0.005,
                          initial_conditions={"C1": 5.0})
    mid = res.series("mid")
    assert mid[0] == pytest.approx(5.0, abs=1e-9)          # starts at the IC
    assert mid[-1] == pytest.approx(2.0, rel=0.02)         # settles at the source
    assert all(b <= a + 1e-9 for a, b in zip(mid, mid[1:]))  # never increases


def test_final_matches_dc_solution():
    # After long enough, transient steady state == the plain DC solve (cap = open).
    from solver.mna import solve
    n = Netlist()
    n.add("V", "V1", "10", "in", "0")
    n.add("R", "R1", "1k", "in", "mid")
    n.add("R", "R2", "1k", "mid", "0")
    n.add("C", "C1", 1e-4, "mid", "0")                     # cap just smooths; DC ignores it
    res = solve_transient(n, t_stop=2.0, dt=0.001)
    dc = solve(n)                                          # solve() treats C as open
    assert res.final()["mid"] == pytest.approx(dc.voltage("mid"), rel=1e-3)


def test_rejects_bad_step():
    with pytest.raises(TransientError):
        solve_transient(rc_circuit(), t_stop=1.0, dt=0.0)
    with pytest.raises(TransientError):
        solve_transient(rc_circuit(), t_stop=1.0, dt=2.0)  # dt > t_stop


def test_rejects_unknown_initial_condition():
    with pytest.raises(TransientError, match="unknown capacitor"):
        solve_transient(rc_circuit(), t_stop=1.0, dt=0.01, initial_conditions={"C9": 1.0})


# --- the deep-math lever: time-varying sources + diodes in transient ---------

def test_sine_helper():
    w = sine(amplitude=2.0, freq_hz=1.0, offset=1.0)
    assert w(0.0) == pytest.approx(1.0)                  # offset + 2·sin(0)
    assert w(0.25) == pytest.approx(3.0)                 # quarter period -> +amplitude
    assert w(0.75) == pytest.approx(-1.0)                # three-quarter -> −amplitude


def test_time_varying_source_drives_the_node():
    # V source = sine across node 'a' to ground, with a resistor to ground. The
    # ideal source pins node 'a', so V(a, t) must equal the sine at every sample.
    n = Netlist()
    n.add("V", "V1", 0.0, "a", "0")
    n.add("R", "R1", "1k", "a", "0")
    w = sine(amplitude=3.0, freq_hz=50.0)
    res = solve_transient(n, t_stop=0.02, dt=0.0005, sources={"V1": w})
    for t, v in zip(res.times, res.series("a")):
        assert v == pytest.approx(w(t), abs=1e-9)


def test_rejects_unknown_source():
    with pytest.raises(TransientError, match="unknown source"):
        solve_transient(rc_circuit(), t_stop=1.0, dt=0.01, sources={"Vnope": sine(1, 1)})


def test_peak_detector_holds_the_peak():
    # Sine -> diode -> cap, NO load: the cap charges toward the input peak (minus
    # one diode drop) and then HOLDS, because the diode blocks any discharge path.
    n = Netlist()
    n.add("V", "V1", 0.0, "ac", "0")
    n.add("D", "D1", 0.0, "ac", "out")                   # anode 'ac', cathode 'out'
    n.add("C", "C1", 10e-6, "out", "0")
    period = 1.0 / 60.0
    res = solve_transient(n, t_stop=4 * period, dt=period / 200,
                          sources={"V1": sine(amplitude=5.0, freq_hz=60.0)})
    out = res.series("out")
    assert 4.0 < out[-1] < 4.6                            # ~5 V − one silicon drop
    assert all(b >= a - 1e-3 for a, b in zip(out, out[1:]))  # non-decreasing (no discharge path)


def test_half_wave_rectifier_smooths_to_dc():
    # Sine -> diode -> cap ∥ load: the classic rectifier. Output stays POSITIVE the
    # whole time (the diode + cap never let it follow the negative half-cycle) and
    # sits near the peak with a bounded ripple.
    n = Netlist()
    n.add("V", "V1", 0.0, "ac", "0")
    n.add("D", "D1", 0.0, "ac", "out")
    n.add("C", "C1", 100e-6, "out", "0")
    n.add("R", "R1", "1k", "out", "0")
    period = 1.0 / 60.0
    res = solve_transient(n, t_stop=6 * period, dt=period / 200,
                          sources={"V1": sine(amplitude=5.0, freq_hz=60.0)})
    out = res.series("out")
    settled = out[len(out) // 2:]                         # skip the initial charge-up
    assert min(settled) > 0.0                             # never follows input negative
    assert 4.0 < max(settled) < 4.6                       # peak ≈ amplitude − diode drop
    assert (max(settled) - min(settled)) < 1.0            # cap smooths: ripple under a volt
