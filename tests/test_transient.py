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
    res = solve_transient(rc_circuit(), t_stop=5.0, dt=0.005)
    mid = res.series("mid")
    assert mid[0] == pytest.approx(0.0, abs=1e-9)
    assert mid[-1] == pytest.approx(5.0, rel=0.02)
    assert all(b >= a - 1e-9 for a, b in zip(mid, mid[1:]))


def test_matches_analytic_rc_curve():
    R, C, V = 1000.0, 1e-3, 5.0
    tau = R * C
    res = solve_transient(rc_circuit(R, C, str(int(V))), t_stop=5.0, dt=0.001)
    for t, v in zip(res.times, res.series("mid")):
        analytic = V * (1 - math.exp(-t / tau))
        assert v == pytest.approx(analytic, abs=0.02)


def test_smaller_dt_is_more_accurate():
    R, C, V, tau = 1000.0, 1e-3, 5.0, 1.0
    target = V * (1 - math.exp(-1.0 / tau))
    err = {}
    for dt in (0.1, 0.005):
        res = solve_transient(rc_circuit(R, C, "5"), t_stop=1.0, dt=dt)
        err[dt] = abs(res.series("mid")[-1] - target)
    assert err[0.005] < err[0.1]


def test_nonzero_initial_condition_discharges():
    res = solve_transient(rc_circuit(V="2"), t_stop=5.0, dt=0.005,
                          initial_conditions={"C1": 5.0})
    mid = res.series("mid")
    assert mid[0] == pytest.approx(5.0, abs=1e-9)
    assert mid[-1] == pytest.approx(2.0, rel=0.02)
    assert all(b <= a + 1e-9 for a, b in zip(mid, mid[1:]))


def test_final_matches_dc_solution():
    from solver.mna import solve
    n = Netlist()
    n.add("V", "V1", "10", "in", "0")
    n.add("R", "R1", "1k", "in", "mid")
    n.add("R", "R2", "1k", "mid", "0")
    n.add("C", "C1", 1e-4, "mid", "0")
    res = solve_transient(n, t_stop=2.0, dt=0.001)
    dc = solve(n)
    assert res.final()["mid"] == pytest.approx(dc.voltage("mid"), rel=1e-3)


def test_rejects_bad_step():
    with pytest.raises(TransientError):
        solve_transient(rc_circuit(), t_stop=1.0, dt=0.0)
    with pytest.raises(TransientError):
        solve_transient(rc_circuit(), t_stop=1.0, dt=2.0)


def test_rejects_unknown_initial_condition():
    with pytest.raises(TransientError, match="unknown capacitor"):
        solve_transient(rc_circuit(), t_stop=1.0, dt=0.01, initial_conditions={"C9": 1.0})



def test_sine_helper():
    w = sine(amplitude=2.0, freq_hz=1.0, offset=1.0)
    assert w(0.0) == pytest.approx(1.0)
    assert w(0.25) == pytest.approx(3.0)
    assert w(0.75) == pytest.approx(-1.0)


def test_time_varying_source_drives_the_node():
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



def rl_circuit(R="1k", L=1.0, V="10"):
    """V source -> R -> node 'mid' -> L -> ground. τ = L/R; final current V/R."""
    n = Netlist()
    n.add("V", "V1", V, "in", "0")
    n.add("R", "R1", R, "in", "mid")
    n.add("L", "L1", L, "mid", "0")
    return n


def test_inductor_current_follows_analytic_rl_curve():
    R, L, V = 1000.0, 1.0, 10.0
    tau = L / R
    res = solve_transient(rl_circuit(), t_stop=5 * tau, dt=tau / 200)
    cur = res.inductor_currents["L1"]
    assert cur[0] == pytest.approx(0.0, abs=1e-12)
    assert cur[-1] == pytest.approx(V / R, rel=0.02)
    for t, i in zip(res.times, cur):
        assert i == pytest.approx((V / R) * (1 - math.exp(-t / tau)), abs=3e-4)


def test_inductor_dc_steady_state_is_a_short():
    from solver.mna import solve
    res = solve_transient(rl_circuit(), t_stop=0.02, dt=1e-5)
    dc = solve(rl_circuit())
    assert res.final()["mid"] == pytest.approx(dc.voltage("mid"), abs=1e-3)
    assert res.inductor_currents["L1"][-1] == pytest.approx(dc.branch_currents["L1"], rel=1e-2)


def test_series_rlc_is_underdamped_and_rings():
    n = Netlist()
    n.add("V", "V1", "5", "in", "0")
    n.add("R", "R1", "200", "in", "a")
    n.add("L", "L1", 1.0, "a", "b")
    n.add("C", "C1", 1e-6, "b", "0")
    res = solve_transient(n, t_stop=0.05, dt=1e-5)
    vc = res.series("b")
    assert max(vc) > 6.0
    assert min(vc[len(vc)//4:]) < 4.5
    assert vc[-1] == pytest.approx(5.0, abs=0.1)


def test_trapezoidal_beats_backward_euler_accuracy():
    target = 5.0 * (1 - math.exp(-1.0))
    be = solve_transient(rc_circuit(), t_stop=1.0, dt=0.05, method="backward-euler").series("mid")[-1]
    tr = solve_transient(rc_circuit(), t_stop=1.0, dt=0.05, method="trapezoidal").series("mid")[-1]
    assert abs(tr - target) < abs(be - target)


def test_nonzero_inductor_initial_current():
    n = Netlist()
    n.add("V", "V1", "0", "in", "0")
    n.add("R", "R1", "1k", "in", "mid")
    n.add("L", "L1", 1.0, "mid", "0")
    res = solve_transient(n, t_stop=0.01, dt=1e-5, initial_currents={"L1": 5e-3})
    cur = res.inductor_currents["L1"]
    assert cur[0] == pytest.approx(5e-3)
    assert abs(cur[-1]) < abs(cur[0])


def test_lands_exactly_on_t_stop_for_nondividing_dt():
    res = solve_transient(rc_circuit(), t_stop=1.0, dt=0.3)
    assert res.times[-1] == pytest.approx(1.0)
    assert res.times[0] == 0.0
    assert all(b >= a - 1e-12 for a, b in zip(res.times, res.times[1:]))
    mid = res.series("mid")
    assert all(b >= a - 1e-9 for a, b in zip(mid, mid[1:]))
    assert 0.0 < mid[-1] < 5.0


def test_finer_dt_lands_on_t_stop_with_correct_value():
    res = solve_transient(rc_circuit(), t_stop=1.0, dt=0.0007)
    assert res.times[-1] == pytest.approx(1.0)
    assert res.series("mid")[-1] == pytest.approx(5.0 * (1 - math.exp(-1.0)), abs=0.02)


def test_rejects_unknown_method_and_inductor_ic():
    with pytest.raises(TransientError, match="method must be"):
        solve_transient(rc_circuit(), t_stop=1.0, dt=0.01, method="rk4")
    with pytest.raises(TransientError, match="unknown inductor"):
        solve_transient(rc_circuit(), t_stop=1.0, dt=0.01, initial_currents={"L9": 1.0})


def test_peak_detector_holds_the_peak():
    n = Netlist()
    n.add("V", "V1", 0.0, "ac", "0")
    n.add("D", "D1", 0.0, "ac", "out")
    n.add("C", "C1", 10e-6, "out", "0")
    period = 1.0 / 60.0
    res = solve_transient(n, t_stop=4 * period, dt=period / 200,
                          sources={"V1": sine(amplitude=5.0, freq_hz=60.0)})
    out = res.series("out")
    assert 4.0 < out[-1] < 4.6
    assert all(b >= a - 1e-3 for a, b in zip(out, out[1:]))


def test_half_wave_rectifier_smooths_to_dc():
    n = Netlist()
    n.add("V", "V1", 0.0, "ac", "0")
    n.add("D", "D1", 0.0, "ac", "out")
    n.add("C", "C1", 100e-6, "out", "0")
    n.add("R", "R1", "1k", "out", "0")
    period = 1.0 / 60.0
    res = solve_transient(n, t_stop=6 * period, dt=period / 200,
                          sources={"V1": sine(amplitude=5.0, freq_hz=60.0)})
    out = res.series("out")
    settled = out[len(out) // 2:]
    assert min(settled) > 0.0
    assert 4.0 < max(settled) < 4.6
    assert (max(settled) - min(settled)) < 1.0
