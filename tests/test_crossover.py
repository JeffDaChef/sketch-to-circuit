"""Tests for crossover handling — lifting the no-crossing-wires constraint.

Two tiers: (1) the threading algorithm on bare skeleton graphs (fast, exact), and
(2) end-to-end on a rendered circuit where two nets' wires cross — including the
control that WITHOUT the crossover box the extractor gets it wrong, which is what
proves the feature is doing the work.
"""

import random

import networkx as nx
import numpy as np
import pytest

from data_collection.extra_layouts import crossover_circuit
from solver.equivalence import circuit_equivalent
from vision.skeleton_graph import build_skeleton_graph
from vision.wire_extraction import extract_netlist, thread_crossovers


# --- tier 1: the threading algorithm on a bare skeleton ----------------------

def _plus_graph():
    """Skeleton graph of a clean '+' crossing (one degree-4 node, 4 endpoints)."""
    m = np.zeros((60, 60), bool)
    m[30, 5:55] = True       # horizontal wire
    m[5:55, 30] = True       # vertical wire
    return build_skeleton_graph(m, prune_len=0)


def test_plain_crossing_is_one_component_until_threaded():
    g = _plus_graph()
    assert nx.number_connected_components(g) == 1          # the bug: the wires are fused
    n = thread_crossovers(g, [{"kind": "crossover", "bbox": [20, 20, 40, 40]}])
    assert n == 1
    assert nx.number_connected_components(g) == 2          # threaded apart


def test_threading_pairs_collinear_wires():
    # After threading, each component must hold one straight wire: the vertical
    # endpoints together, the horizontal endpoints together — not a mixed pair.
    g = _plus_graph()
    thread_crossovers(g, [{"kind": "crossover", "bbox": [20, 20, 40, 40]}])
    comps = []
    for comp in nx.connected_components(g):
        eps = sorted(tuple(map(int, g.nodes[k]["pos"]))
                     for k in comp if g.nodes[k]["kind"] == "endpoint")
        comps.append(eps)
    assert [(30, 5), (30, 54)] in comps                    # vertical wire (same x)
    assert [(5, 30), (54, 30)] in comps                    # horizontal wire (same y)


def test_threading_is_graceful_when_no_crossing_node():
    # A crossover box over empty space (or a hop-drawn crossover whose wires never
    # touched) has no degree-4 node: nothing is threaded, graph unchanged.
    g = _plus_graph()
    before = g.number_of_nodes()
    n = thread_crossovers(g, [{"kind": "crossover", "bbox": [0, 0, 5, 5]}])
    assert n == 0 and g.number_of_nodes() == before


# --- tier 2: end-to-end on a rendered crossover circuit ----------------------

def test_crossover_circuit_extracts_correctly():
    img, comps, truth = crossover_circuit(random.Random(0))
    extracted = extract_netlist(img, comps)
    assert circuit_equivalent(extracted, truth)


def test_without_crossover_box_it_gets_it_wrong():
    # The control: strip the crossover box and the crossing fuses n1 and n2,
    # shorting R1 — so the extraction must NOT match. This is what proves the
    # crossover handling (not something else) is responsible for the success.
    img, comps, truth = crossover_circuit(random.Random(0))
    no_box = [c for c in comps if c["kind"] != "crossover"]
    assert not circuit_equivalent(extract_netlist(img, no_box), truth)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_crossover_generalizes_across_jitter(seed):
    # The crossing handling shouldn't depend on exact pixel placement.
    img, comps, truth = crossover_circuit(random.Random(seed))
    assert circuit_equivalent(extract_netlist(img, comps), truth)
