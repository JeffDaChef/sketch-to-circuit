"""Tests for solver/transient.py.

The anchor test compares the simulator against the *analytic* RC charging curve
v(t) = V·(1 − e^(−t/RC)) — the formula you derive on paper. If the companion
model's sign or the integration is wrong, this curve comes out wrong, so it's a
strong correctness check, not just a smoke test.
"""

import math

import pytest

from solver.netlist import Netlist
from solver.transient import TransientError, solve_transient


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
