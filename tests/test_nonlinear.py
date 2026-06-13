"""Tests for solver/nonlinear.py (diodes via Newton-Raphson).

The anchor test is a *self-consistency* check: at the converged solution the
current the diode equation predicts (from the diode voltage) must equal the
current Ohm's law predicts through the series resistor. If both laws hold at
once, the operating point is genuinely correct — no hardcoded expected number.
"""

import pytest

from solver.netlist import Netlist
from solver.nonlinear import LED, SILICON, NonlinearError, solve_nonlinear


def diode_resistor(V="5", R="1k"):
    """V source -> R -> node 'mid' -> diode (anode 'mid', cathode ground)."""
    n = Netlist()
    n.add("V", "V1", V, "in", "0")
    n.add("R", "R1", R, "in", "mid")
    n.add("D", "D1", 0.0, "mid", "0")
    return n


def test_forward_diode_drop_is_physical():
    # A silicon diode conducting a few mA should sit around 0.6-0.75 V.
    r = solve_nonlinear(diode_resistor())
    vd = r.voltage("mid")
    assert 0.6 < vd < 0.75


def test_kcl_and_diode_equation_agree_at_solution():
    # Self-consistency: resistor current (Ohm) == diode current (Shockley).
    r = solve_nonlinear(diode_resistor("5", "1k"))
    vd = r.voltage("mid")
    i_resistor = (5.0 - vd) / 1000.0
    i_diode = SILICON.current(vd)
    assert i_resistor == pytest.approx(i_diode, rel=1e-4)
    assert r.branch_currents["D1"] == pytest.approx(i_diode, rel=1e-6)


def test_reverse_biased_diode_blocks_current():
    # Diode flipped (cathode toward the source) -> it blocks; almost no current,
    # so almost the whole source voltage drops across the (reverse) diode.
    n = Netlist()
    n.add("V", "V1", "5", "in", "0")
    n.add("R", "R1", "1k", "in", "mid")
    n.add("D", "D1", 0.0, "0", "mid")          # anode at ground, cathode at 'mid' -> reverse
    r = solve_nonlinear(n)
    assert abs(r.branch_currents["D1"]) < 1e-9  # sub-nanoamp leakage
    assert r.voltage("mid") == pytest.approx(5.0, abs=1e-3)


def test_led_turns_on_higher_than_silicon():
    r = solve_nonlinear(diode_resistor("5", "220"), models={"D1": LED})
    assert 1.6 < r.voltage("mid") < 2.1       # LEDs glow around 1.8-2 V
    # And it should pass a sane LED current (a few to tens of mA), not zero/huge.
    assert 0.001 < r.branch_currents["D1"] < 0.05


def test_two_diodes_in_series_double_the_drop():
    n = Netlist()
    n.add("V", "V1", "5", "in", "0")
    n.add("R", "R1", "1k", "in", "a")
    n.add("D", "D1", 0.0, "a", "b")
    n.add("D", "D2", 0.0, "b", "0")
    r = solve_nonlinear(n)
    # Two silicon drops in series ~1.2-1.4 V across both diodes (node 'a').
    assert 1.2 < r.voltage("a") < 1.45


def test_no_diode_matches_linear_solver():
    # With no diodes, solve_nonlinear must agree exactly with the linear solve().
    from solver.mna import solve
    n = Netlist()
    n.add("V", "V1", "10", "in", "0")
    n.add("R", "R1", "1k", "in", "mid")
    n.add("R", "R2", "1k", "mid", "0")
    assert solve_nonlinear(n).voltage("mid") == pytest.approx(solve(n).voltage("mid"))


def test_diode_current_increases_with_source_voltage():
    # Sanity/monotonicity: more drive -> more diode current.
    i_low = solve_nonlinear(diode_resistor("3", "1k")).branch_currents["D1"]
    i_high = solve_nonlinear(diode_resistor("9", "1k")).branch_currents["D1"]
    assert i_high > i_low > 0


def test_nonconvergence_raises():
    # An impossibly tight tolerance with too few iterations should report failure,
    # not silently return a half-baked answer.
    with pytest.raises(NonlinearError):
        solve_nonlinear(diode_resistor(), max_iter=1, tol=1e-15)
