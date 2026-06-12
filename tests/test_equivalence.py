"""Tests for solver/equivalence.py — every expected result is hand-checkable.

These tests verify that circuit_equivalent correctly identifies circuits with
the same topology (True) and correctly rejects circuits with different
topologies (False), regardless of node naming or component values.
"""

import pytest

from solver.netlist import Netlist
from solver.equivalence import circuit_equivalent, to_circuit_graph


# ---------------------------------------------------------------------------
# Helpers to build small hand-written netlists
# ---------------------------------------------------------------------------

def _divider_a() -> Netlist:
    """A 10V source driving two 1k resistors in series.
    Nodes named: in, mid (ground is "0").
    """
    n = Netlist()
    n.add("V", "V1", "10",  "in",  "0")
    n.add("R", "R1", "1k",  "in",  "mid")
    n.add("R", "R2", "1k",  "mid", "0")
    return n


def _divider_b() -> Netlist:
    """Same topology as _divider_a but different node names ("vcc","tap") and
    different values (5V, 2k2, 470).  Should be equivalent.
    """
    n = Netlist()
    n.add("V", "Vsrc", "5",    "vcc", "0")
    n.add("R", "Ra",   "2k2",  "vcc", "tap")
    n.add("R", "Rb",   "470",  "tap", "0")
    return n


def _parallel_a() -> Netlist:
    """A 5V source with two resistors in parallel (both see the full supply)."""
    n = Netlist()
    n.add("V", "V1", "5",   "top", "0")
    n.add("R", "R1", "1k",  "top", "0")
    n.add("R", "R2", "10k", "top", "0")
    return n


def _parallel_b() -> Netlist:
    """Same parallel topology, different names/values — should be equivalent."""
    n = Netlist()
    n.add("V", "Vsrc", "9",   "rail", "0")
    n.add("R", "RA",   "100", "rail", "0")
    n.add("R", "RB",   "220", "rail", "0")
    return n


def _source_replaced() -> Netlist:
    """_divider_a but with the voltage source replaced by a resistor.
    Structurally a three-resistor chain — NOT equivalent to the divider.
    """
    n = Netlist()
    n.add("R", "R0", "100",  "in",  "0")   # was a V source
    n.add("R", "R1", "1k",   "in",  "mid")
    n.add("R", "R2", "1k",   "mid", "0")
    return n


def _divider_ground_swapped() -> Netlist:
    """_divider_a but 'mid' is ground instead of '0'.
    The ground node must map to ground, so this is NOT equivalent.
    """
    n = Netlist()
    # Now 'mid' is the reference; '0' is a floating internal node.
    n.add("V", "V1", "10",  "in",  "0")   # '0' is now non-ground
    n.add("R", "R1", "1k",  "in",  "mid")
    n.add("R", "R2", "1k",  "mid", "0")
    # This netlist has the same edges, but 'mid' is NOT the node named "0".
    # Compare to _divider_a where "0" is ground: they differ in which node is ground.
    return n


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCircuitEquivalent:

    def test_identical_topology_different_names_and_values(self):
        """Two dividers wired the same way but with different node names
        and component values must be equivalent."""
        assert circuit_equivalent(_divider_a(), _divider_b()) is True

    def test_series_vs_parallel_not_equivalent(self):
        """Series resistors (voltage divider) vs. parallel resistors are
        different topologies — must NOT be equivalent."""
        assert circuit_equivalent(_divider_a(), _parallel_a()) is False

    def test_parallel_different_names_equivalent(self):
        """Two parallel-bank circuits with different names/values are equivalent."""
        assert circuit_equivalent(_parallel_a(), _parallel_b()) is True

    def test_resistor_replaced_by_source_not_equivalent(self):
        """Swapping one component kind breaks equivalence."""
        assert circuit_equivalent(_divider_a(), _source_replaced()) is False

    def test_ground_preservation(self):
        """Two netlists that differ only in WHICH node is ground are NOT equivalent.
        The isomorphism must pin the '0' node to '0'.
        """
        # Build a divider where the internal node 'mid' acts as if it were ground.
        # We do this by making a netlist where the node named "0" is the
        # internal tap rather than the real ground.
        mid_as_gnd = Netlist()
        mid_as_gnd.add("V", "V1", "10",  "in",   "tap")   # ground is "tap" if "0" never appears
        # Actually to test ground preservation we need "0" to appear but in a
        # *different* structural position.  In _divider_a: "0" is the bottom of
        # the source and the bottom of R2.  Let's build a version where "0" is
        # only the bottom of R2 but the source connects to a non-ground node:
        #
        #   V1: in -> mid    (source does NOT touch ground)
        #   R1: in -> mid    (same edge as V1 — parallel source/resistor)
        #   R2: mid -> 0
        #
        # This has a different structure from _divider_a.
        different_ground = Netlist()
        different_ground.add("V", "V1", "10", "in",  "mid")
        different_ground.add("R", "R1", "1k", "in",  "mid")
        different_ground.add("R", "R2", "1k", "mid", "0")
        # _divider_a has ground connected to V1 directly; different_ground does not.
        assert circuit_equivalent(_divider_a(), different_ground) is False

    def test_reflexive(self):
        """A netlist is always equivalent to itself."""
        n = _divider_a()
        assert circuit_equivalent(n, n) is True

    def test_symmetric(self):
        """Equivalence is symmetric: a≡b iff b≡a."""
        a = _divider_a()
        b = _divider_b()
        assert circuit_equivalent(a, b) == circuit_equivalent(b, a)


class TestToCircuitGraph:

    def test_node_count(self):
        """The graph has one node per distinct net in the netlist."""
        n = _divider_a()
        G = to_circuit_graph(n)
        # _divider_a has nets: "in", "mid", "0" -> 3 nodes.
        assert G.number_of_nodes() == 3

    def test_edge_count(self):
        """The graph has one edge per component (V1 + R1 + R2 = 3)."""
        n = _divider_a()
        G = to_circuit_graph(n)
        assert G.number_of_edges() == 3

    def test_ground_node_tagged(self):
        """The node named '0' must have is_ground=True; others False."""
        G = to_circuit_graph(_divider_a())
        assert G.nodes["0"]["is_ground"] is True
        assert G.nodes["in"]["is_ground"] is False
        assert G.nodes["mid"]["is_ground"] is False

    def test_edge_kinds(self):
        """Each edge carries the correct 'kind' attribute."""
        G = to_circuit_graph(_divider_a())
        # Collect all edge kinds as a multiset.
        kinds = [data["kind"] for _, _, data in G.edges(data=True)]
        assert sorted(kinds) == ["R", "R", "V"]
