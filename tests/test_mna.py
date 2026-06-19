"""Tests for solver/mna.py.

Every circuit here is small enough to solve by hand, so each expected number is
written next to the hand-calculation that produced it. If the solver disagrees
with paper, the solver is wrong. These same circuits later seed the ngspice
validation suite (brief §6.1).
"""

import pytest

from solver.mna import SolverError, solve
from solver.netlist import Netlist


def test_voltage_divider():
    n = Netlist()
    n.add("V", "V1", "10", "in", "0")
    n.add("R", "R1", "1k", "in", "mid")
    n.add("R", "R2", "1k", "mid", "0")
    r = solve(n)
    assert r.voltage("in") == pytest.approx(10.0)
    assert r.voltage("mid") == pytest.approx(5.0)
    assert r.voltage("0") == 0.0
    assert r.source_currents["V1"] == pytest.approx(-0.005)


def test_unequal_divider():
    n = Netlist()
    n.add("V", "V1", "10", "a", "0")
    n.add("R", "R1", "1k", "a", "b")
    n.add("R", "R2", "3k", "b", "0")
    r = solve(n)
    assert r.voltage("b") == pytest.approx(7.5)
    assert r.branch_currents["R1"] == pytest.approx(0.0025)


def test_parallel_resistors():
    n = Netlist()
    n.add("V", "V1", "6", "a", "0")
    n.add("R", "R1", "1k", "a", "0")
    n.add("R", "R2", "1k", "a", "0")
    r = solve(n)
    assert r.voltage("a") == pytest.approx(6.0)
    assert r.branch_currents["R1"] == pytest.approx(0.006)
    assert r.source_currents["V1"] == pytest.approx(-0.012)


def test_current_source():
    n = Netlist()
    n.add("I", "I1", "1m", "0", "a")
    n.add("R", "R1", "1k", "a", "0")
    r = solve(n)
    assert r.voltage("a") == pytest.approx(1.0)
    assert r.branch_currents["I1"] == pytest.approx(0.001)


def test_two_sources():
    n = Netlist()
    n.add("V", "V1", "10", "a", "0")
    n.add("V", "V2", "4", "b", "0")
    n.add("R", "R1", "2k", "a", "b")
    r = solve(n)
    assert r.voltage("a") == pytest.approx(10.0)
    assert r.voltage("b") == pytest.approx(4.0)
    assert r.branch_currents["R1"] == pytest.approx(0.003)
    assert r.source_currents["V1"] == pytest.approx(-0.003)
    assert r.source_currents["V2"] == pytest.approx(+0.003)


def test_series_chain_node_voltages():
    n = Netlist()
    n.add("V", "V1", "9", "n3", "0")
    n.add("R", "R1", "1k", "n3", "n2")
    n.add("R", "R2", "1k", "n2", "n1")
    n.add("R", "R3", "1k", "n1", "0")
    r = solve(n)
    assert r.voltage("n2") == pytest.approx(6.0)
    assert r.voltage("n1") == pytest.approx(3.0)


def test_capacitor_is_open_at_dc():
    n = Netlist()
    n.add("V", "V1", "10", "in", "0")
    n.add("R", "R1", "1k", "in", "mid")
    n.add("R", "R2", "1k", "mid", "0")
    n.add("C", "C1", "1u", "mid", "0")
    r = solve(n)
    assert r.voltage("mid") == pytest.approx(5.0)


def test_no_ground_raises():
    n = Netlist()
    n.add("V", "V1", "5", "a", "b")
    n.add("R", "R1", "1k", "a", "b")
    with pytest.raises(SolverError):
        solve(n)


def test_floating_node_raises():
    n = Netlist()
    n.add("V", "Vref", "0", "gndtie", "0")
    n.add("I", "I1", "1m", "0", "a")
    n.add("R", "R1", "1k", "a", "b")
    n.add("I", "I2", "1m", "b", "0")
    with pytest.raises(SolverError):
        solve(n)


def test_diode_rejected():
    n = Netlist()
    n.add("V", "V1", "5", "a", "0")
    n.add("D", "D1", "2", "a", "0")
    with pytest.raises(SolverError, match="nonlinear|diode"):
        solve(n)


def test_solve_from_spice_text():
    text = """* divider
    V1 in 0 10
    R1 in mid 1000
    R2 mid 0 1000
    .end
    """
    r = solve(Netlist.from_spice(text))
    assert r.voltage("mid") == pytest.approx(5.0)
