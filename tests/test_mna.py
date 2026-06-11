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
    # 10V across two equal 1k resistors. By symmetry the midpoint sits at 5V.
    n = Netlist()
    n.add("V", "V1", "10", "in", "0")
    n.add("R", "R1", "1k", "in", "mid")
    n.add("R", "R2", "1k", "mid", "0")
    r = solve(n)
    assert r.voltage("in") == pytest.approx(10.0)
    assert r.voltage("mid") == pytest.approx(5.0)
    assert r.voltage("0") == 0.0
    # Total resistance 2k, so 10V / 2k = 5mA flows. SPICE sign: current into the
    # source's + terminal is negative when the source is delivering power.
    assert r.source_currents["V1"] == pytest.approx(-0.005)


def test_unequal_divider():
    # V(b) = 10V * R2/(R1+R2) = 10 * 3k/4k = 7.5V. Current = 10/4k = 2.5mA.
    n = Netlist()
    n.add("V", "V1", "10", "a", "0")
    n.add("R", "R1", "1k", "a", "b")
    n.add("R", "R2", "3k", "b", "0")
    r = solve(n)
    assert r.voltage("b") == pytest.approx(7.5)
    assert r.branch_currents["R1"] == pytest.approx(0.0025)


def test_parallel_resistors():
    # Both resistors see the full 6V. Each carries 6mA, so the source delivers 12mA.
    n = Netlist()
    n.add("V", "V1", "6", "a", "0")
    n.add("R", "R1", "1k", "a", "0")
    n.add("R", "R2", "1k", "a", "0")
    r = solve(n)
    assert r.voltage("a") == pytest.approx(6.0)
    assert r.branch_currents["R1"] == pytest.approx(0.006)
    assert r.source_currents["V1"] == pytest.approx(-0.012)


def test_current_source():
    # 1mA forced into node 'a' through a 1k resistor to ground -> V = I*R = 1.0V.
    n = Netlist()
    n.add("I", "I1", "1m", "0", "a")   # + node is ground, - node is 'a': feeds 'a'
    n.add("R", "R1", "1k", "a", "0")
    r = solve(n)
    assert r.voltage("a") == pytest.approx(1.0)
    assert r.branch_currents["I1"] == pytest.approx(0.001)


def test_two_sources():
    # 10V and 4V sources joined by a 2k resistor. Current = (10-4)/2k = 3mA from a to b.
    n = Netlist()
    n.add("V", "V1", "10", "a", "0")
    n.add("V", "V2", "4", "b", "0")
    n.add("R", "R1", "2k", "a", "b")
    r = solve(n)
    assert r.voltage("a") == pytest.approx(10.0)
    assert r.voltage("b") == pytest.approx(4.0)
    assert r.branch_currents["R1"] == pytest.approx(0.003)
    # V1 delivers 3mA (negative by SPICE sign); V2 absorbs 3mA (positive).
    assert r.source_currents["V1"] == pytest.approx(-0.003)
    assert r.source_currents["V2"] == pytest.approx(+0.003)


def test_series_chain_node_voltages():
    # Three 1k resistors in series across 9V -> equal 3V drops -> 6V and 3V taps.
    n = Netlist()
    n.add("V", "V1", "9", "n3", "0")
    n.add("R", "R1", "1k", "n3", "n2")
    n.add("R", "R2", "1k", "n2", "n1")
    n.add("R", "R3", "1k", "n1", "0")
    r = solve(n)
    assert r.voltage("n2") == pytest.approx(6.0)
    assert r.voltage("n1") == pytest.approx(3.0)


def test_capacitor_is_open_at_dc():
    # A capacitor blocks DC, so it carries no current and doesn't load the divider.
    n = Netlist()
    n.add("V", "V1", "10", "in", "0")
    n.add("R", "R1", "1k", "in", "mid")
    n.add("R", "R2", "1k", "mid", "0")
    n.add("C", "C1", "1u", "mid", "0")   # should change nothing at DC
    r = solve(n)
    assert r.voltage("mid") == pytest.approx(5.0)


def test_no_ground_raises():
    n = Netlist()
    n.add("V", "V1", "5", "a", "b")
    n.add("R", "R1", "1k", "a", "b")
    with pytest.raises(SolverError):
        solve(n)


def test_floating_node_raises():
    # A current source feeding a node with no resistive path to ground is unsolvable.
    n = Netlist()
    n.add("V", "Vref", "0", "gndtie", "0")  # gives the circuit a ground tie
    n.add("I", "I1", "1m", "0", "a")        # pushes current into 'a'...
    n.add("R", "R1", "1k", "a", "b")        # ...but 'b' floats: no path onward
    n.add("I", "I2", "1m", "b", "0")        # forces inconsistent current at 'b'
    with pytest.raises(SolverError):
        solve(n)


def test_diode_rejected():
    n = Netlist()
    n.add("V", "V1", "5", "a", "0")
    n.add("D", "D1", "2", "a", "0")
    with pytest.raises(SolverError):
        solve(n)


def test_solve_from_spice_text():
    # Confirms the netlist text format and the solver work together end to end.
    text = """* divider
    V1 in 0 10
    R1 in mid 1000
    R2 mid 0 1000
    .end
    """
    r = solve(Netlist.from_spice(text))
    assert r.voltage("mid") == pytest.approx(5.0)
