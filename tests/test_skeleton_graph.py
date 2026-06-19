"""Tests for vision/skeleton_graph.py — written BEFORE the implementation (TDD).

Every mask here is drawn by hand with numpy slices, small enough to check the
expected node/edge counts on paper. Coordinates are (x, y) with top-left origin,
matching the rest of the project.

The contract these tests pin down (the redesign's foundation):
  * degree-1 skeleton pixels become ENDPOINT nodes (where erasure cut a wire —
    i.e. where a component terminal attached),
  * degree-3+ pixels cluster into BRANCH nodes (real wire junctions),
  * degree-2 pixels become the EDGES connecting them,
  * tiny whisker spurs can be pruned at the graph level,
  * closed rings with no endpoints still appear in the graph (LOOP node),
  * corners and T-junctions produce CLEAN counts (no phantom branch nodes from
    diagonal pixel contact — the classic skeleton-graph pitfall).
"""

import numpy as np
import pytest

from vision.skeleton_graph import build_skeleton_graph, endpoint_positions


def blank(h: int = 20, w: int = 20) -> np.ndarray:
    return np.zeros((h, w), dtype=bool)


def kinds(g):
    """Count node kinds, e.g. {'endpoint': 2, 'branch': 1}."""
    out: dict = {}
    for _, data in g.nodes(data=True):
        out[data["kind"]] = out.get(data["kind"], 0) + 1
    return out


def n_components(g):
    import networkx as nx
    return nx.number_connected_components(g)


class TestBasicShapes:
    def test_straight_line(self):
        """A horizontal 9-px line: 2 endpoints, 1 edge, no branches."""
        m = blank()
        m[5, 2:11] = True
        g = build_skeleton_graph(m, prune_len=0)

        assert kinds(g) == {"endpoint": 2}
        assert g.number_of_edges() == 1
        assert n_components(g) == 1

        pos = sorted(endpoint_positions(g).values())
        assert pos == [(2, 5), (10, 5)]

        (_, _, data), = g.edges(data=True)
        assert data["length"] == 7

    def test_l_bend_is_one_clean_path(self):
        """An L: corner pixels must NOT create phantom branch nodes."""
        m = blank()
        m[5, 2:7] = True
        m[5:10, 6] = True
        g = build_skeleton_graph(m, prune_len=0)

        assert kinds(g) == {"endpoint": 2}
        assert g.number_of_edges() == 1
        assert n_components(g) == 1
        pos = sorted(endpoint_positions(g).values())
        assert pos == [(2, 5), (6, 9)]

    def test_diagonal_line(self):
        """A 45-degree line is a single path with 2 endpoints."""
        m = blank()
        for i in range(2, 9):
            m[i, i] = True
        g = build_skeleton_graph(m, prune_len=0)

        assert kinds(g) == {"endpoint": 2}
        assert g.number_of_edges() == 1
        pos = sorted(endpoint_positions(g).values())
        assert pos == [(2, 2), (8, 8)]

    def test_t_junction(self):
        """A T: 3 endpoints, exactly 1 branch node, 3 edges."""
        m = blank()
        m[8, 2:15] = True
        m[2:9, 8] = True
        g = build_skeleton_graph(m, prune_len=0)

        assert kinds(g) == {"endpoint": 3, "branch": 1}
        assert g.number_of_edges() == 3
        assert n_components(g) == 1

    def test_plus_junction(self):
        """A +: 4 endpoints, 1 branch node, 4 edges."""
        m = blank()
        m[8, 2:15] = True
        m[2:15, 8] = True
        g = build_skeleton_graph(m, prune_len=0)

        assert kinds(g) == {"endpoint": 4, "branch": 1}
        assert g.number_of_edges() == 4
        assert n_components(g) == 1


class TestComponentsAndLoops:
    def test_two_separate_lines(self):
        """Disconnected wires are separate graph components."""
        m = blank()
        m[3, 2:9] = True
        m[12, 5:15] = True
        g = build_skeleton_graph(m, prune_len=0)

        assert kinds(g) == {"endpoint": 4}
        assert g.number_of_edges() == 2
        assert n_components(g) == 2

    def test_closed_ring_still_appears(self):
        """A closed rectangle has no endpoints/branches but must not vanish:
        it gets a LOOP node so connectivity queries still see it."""
        m = blank()
        m[4, 4:13] = True
        m[10, 4:13] = True
        m[4:11, 4] = True
        m[4:11, 12] = True
        g = build_skeleton_graph(m, prune_len=0)

        assert g.number_of_nodes() >= 1
        assert "loop" in kinds(g)
        assert n_components(g) == 1


class TestPruning:
    def make_whiskered_line(self):
        """A long line with a 3-px whisker hanging off its middle."""
        m = blank()
        m[8, 2:17] = True
        m[5:8, 9] = True
        return m

    def test_no_pruning_keeps_whisker(self):
        g = build_skeleton_graph(self.make_whiskered_line(), prune_len=0)
        assert kinds(g)["endpoint"] == 3
        assert kinds(g).get("branch", 0) == 1

    def test_pruning_removes_whisker_only(self):
        g = build_skeleton_graph(self.make_whiskered_line(), prune_len=4)
        pos = sorted(endpoint_positions(g).values())
        assert pos == [(2, 8), (16, 8)]
        assert n_components(g) == 1

    def test_pruning_never_deletes_a_standalone_short_wire(self):
        """A short isolated segment is a real wire, not a whisker — pruning
        only removes spurs that hang off a larger structure."""
        m = blank()
        m[5, 6:10] = True
        g = build_skeleton_graph(m, prune_len=4)
        assert kinds(g) == {"endpoint": 2}
        assert g.number_of_edges() == 1
