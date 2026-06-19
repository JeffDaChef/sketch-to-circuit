"""Modified Nodal Analysis (MNA) — a from-scratch DC circuit solver.

WHAT THIS DOES
--------------
Given a Netlist (see netlist.py), compute the voltage at every node and the
current through every voltage source. This is the same core algorithm that runs
inside SPICE, the industry-standard circuit simulator.

THE IDEA IN ONE PARAGRAPH
-------------------------
The unknowns in a circuit are the node voltages. Two laws give us enough
equations to find them: Ohm's law (current through a resistor = voltage across
it / resistance) and Kirchhoff's Current Law (at every node, currents in = currents
out). Writing KCL at every node, with each current rewritten via Ohm's law in
terms of the unknown voltages, gives one linear equation per node. Stack those
equations into a matrix A and a right-hand-side vector z, and the solution of
A x = z is the list of node voltages. "Modified" nodal analysis adds one extra
equation (and one extra unknown, a branch current) for each ideal voltage source,
because a voltage source fixes a voltage instead of obeying Ohm's law.

SCOPE (v1): linear DC only — resistors (R), ideal voltage sources (V), ideal
current sources (I). Capacitors (C) are open circuits at DC, so they are ignored.
Inductors (L) are short circuits at DC (no voltage across them in steady state),
so they are handled exactly like a 0 V voltage source — same "modified" trick,
adding one branch-current unknown each. Diodes/LEDs (D) are non-linear and need
Newton-Raphson iteration (solver/nonlinear.py), so here we refuse them with a
clear error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solver.netlist import GROUND, Netlist


class SolverError(Exception):
    """Raised when a circuit cannot be solved (no ground, floating node, etc.)."""


@dataclass
class SolveResult:
    """The answer: voltages everywhere, plus currents we can report."""

    node_voltages: dict[str, float]
    source_currents: dict[str, float]
    branch_currents: dict[str, float]

    def voltage(self, net: str) -> float:
        """Convenience: the voltage at one net."""
        return self.node_voltages[net]

    def __str__(self) -> str:
        lines = ["Node voltages:"]
        for net in sorted(self.node_voltages):
            lines.append(f"  V({net}) = {self.node_voltages[net]:+.4f} V")
        if self.source_currents:
            lines.append("Source currents (SPICE sign: + = into the + terminal):")
            for name in sorted(self.source_currents):
                lines.append(f"  I({name}) = {self.source_currents[name]*1e3:+.4f} mA")
        return "\n".join(lines)


def solve(netlist: Netlist) -> SolveResult:
    """Solve a netlist for its DC node voltages and branch currents."""

    if not netlist.has_ground():
        raise SolverError("circuit has no ground (net '0') — nothing to measure voltages against")
    for c in netlist.components:
        if c.kind == "D":
            raise SolverError(
                f"{c.name}: diodes/LEDs are non-linear — use "
                "solver.nonlinear.solve_nonlinear(), which wraps this solver in "
                "Newton-Raphson iteration"
            )

    nodes = netlist.node_names()
    node_index = {name: i for i, name in enumerate(nodes)}
    n = len(nodes)

    vsources = [c for c in netlist.components if c.kind in ("V", "L")]
    vsource_index = {c.name: n + k for k, c in enumerate(vsources)}
    m = len(vsources)

    size = n + m
    A = np.zeros((size, size))
    z = np.zeros(size)

    for c in netlist.components:
        if c.kind == "R":
            g = 1.0 / c.value
            a, b = c.nodes
            if a != GROUND:
                A[node_index[a], node_index[a]] += g
            if b != GROUND:
                A[node_index[b], node_index[b]] += g
            if a != GROUND and b != GROUND:
                A[node_index[a], node_index[b]] -= g
                A[node_index[b], node_index[a]] -= g
        elif c.kind == "I":
            a, b = c.nodes
            if a != GROUND:
                z[node_index[a]] -= c.value
            if b != GROUND:
                z[node_index[b]] += c.value

    for c in vsources:
        s = vsource_index[c.name]
        p, q = c.nodes
        if p != GROUND:
            A[node_index[p], s] += 1
            A[s, node_index[p]] += 1
        if q != GROUND:
            A[node_index[q], s] -= 1
            A[s, node_index[q]] -= 1
        z[s] = c.value if c.kind == "V" else 0.0

    try:
        x = np.linalg.solve(A, z)
    except np.linalg.LinAlgError as err:
        raise SolverError(
            "circuit is unsolvable (singular matrix) — likely a floating node "
            "(a node with no resistive path to ground), a shorted voltage source, "
            "or a missing connection"
        ) from err

    node_voltages = {GROUND: 0.0}
    for name, i in node_index.items():
        node_voltages[name] = float(x[i])

    source_currents = {c.name: float(x[vsource_index[c.name]])
                       for c in vsources if c.kind == "V"}

    branch_currents: dict[str, float] = {}
    for c in netlist.components:
        if c.kind == "R":
            va, vb = node_voltages[c.nodes[0]], node_voltages[c.nodes[1]]
            branch_currents[c.name] = (va - vb) / c.value
        elif c.kind == "I":
            branch_currents[c.name] = c.value
        elif c.kind == "L":
            branch_currents[c.name] = float(x[vsource_index[c.name]])

    return SolveResult(node_voltages, source_currents, branch_currents)
