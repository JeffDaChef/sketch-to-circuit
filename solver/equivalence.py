"""Circuit equivalence checker based on graph isomorphism.

WHY THIS EXISTS
---------------
When the vision pipeline extracts a netlist from a circuit image it cannot know
the generator's internal node names ("n1", "top", …) — it makes up its own.
Likewise it ignores component *values* because the image may not show them
clearly enough.  So the only meaningful question is: does the extracted netlist
have the **same electrical topology** as the ground-truth netlist?

We model each netlist as a NetworkX MultiGraph where:
  * nodes  = electrical nets (the node named "0" is special: it's ground)
  * edges  = components; each edge carries a ``kind`` attribute ("R", "V", …)

Two circuits are equivalent when those graphs are isomorphic under a mapping
that (a) preserves ground / non-ground nodes and (b) matches edge kinds.

This is also used as the Phase-4 extraction metric: run
``circuit_equivalent(extracted, ground_truth)`` and count how many seeds pass.
"""

from __future__ import annotations

import networkx as nx
from networkx.algorithms import isomorphism

from solver.netlist import GROUND, Netlist


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def to_circuit_graph(net: Netlist) -> nx.MultiGraph:
    """Convert a Netlist to a NetworkX MultiGraph suitable for isomorphism tests.

    Each electrical net becomes a node tagged with ``is_ground=True`` if it is
    the SPICE ground net ("0"), ``False`` otherwise.
    Each component becomes an edge tagged with ``kind`` = its one-letter code
    ("R", "V", …).  Values are intentionally ignored.
    """
    G: nx.MultiGraph = nx.MultiGraph()

    # Collect all node names first so isolated nodes still appear.
    all_nodes = {n for c in net.components for n in c.nodes}
    for name in all_nodes:
        G.add_node(name, is_ground=(name == GROUND))

    for comp in net.components:
        # The edge represents the component; kind is the only info we keep.
        G.add_edge(comp.nodes[0], comp.nodes[1], kind=comp.kind)

    return G


def circuit_equivalent(net_a: Netlist, net_b: Netlist) -> bool:
    """Return True when net_a and net_b have the same electrical topology.

    "Same topology" means the circuit graphs are isomorphic under a node
    mapping that:
      - maps the ground node of net_a to the ground node of net_b and
        non-ground to non-ground  (``is_ground`` node attribute must match)
      - matches every edge by its ``kind`` code (so resistors can only map to
        resistors, voltage sources to voltage sources, etc.)

    Component *values* and *names* are deliberately ignored — the extraction
    pipeline cannot reliably read values from a sketch.

    Examples
    --------
    >>> from solver.netlist import Netlist
    >>> a = Netlist(); a.add("V","V1","5","top","0"); a.add("R","R1","1k","top","0")
    >>> b = Netlist(); b.add("V","Vsrc","9","vcc","0"); b.add("R","R1","2k","vcc","0")
    >>> circuit_equivalent(a, b)
    True
    """
    Ga = to_circuit_graph(net_a)
    Gb = to_circuit_graph(net_b)

    # node_match: both endpoints' is_ground flag must agree.
    node_match = isomorphism.categorical_node_match("is_ground", False)

    # edge_match: every edge (component) must match by kind.
    # categorical_multiedge_match handles multigraphs where two nodes can be
    # connected by more than one component (e.g. two resistors in parallel).
    edge_match = isomorphism.categorical_multiedge_match("kind", None)

    matcher = isomorphism.GraphMatcher(Ga, Gb,
                                       node_match=node_match,
                                       edge_match=edge_match)
    return matcher.is_isomorphic()
