"""Skeleton graph — turning a 1-px wire skeleton into a real graph.

WHY THIS EXISTS
---------------
The first wire extractor treated leftover wires as "blobs" and matched component
terminals to whatever blob happened to be *near* — which worked on two circuit
layouts and fell apart on a third (see ROADMAP.md "FINDING"). The redesign needs
a faithful description of the wires themselves: where each wire STARTS and ENDS,
where wires MEET, and which path connects what.

That description is a graph, built from the skeleton:

  * a skeleton pixel with ONE neighbour is the tip of a wire — an **endpoint**
    node. Crucially: erasing a component's body *cuts* the wires that entered
    it, and every cut leaves an endpoint exactly where the wire attached to the
    component. Endpoints are therefore "places a terminal might connect" — and
    they are sparse, unlike "any wire pixel anywhere".
  * a pixel with THREE OR MORE neighbours is where wires meet — a **branch**
    node (adjacent branch pixels are merged into one node).
  * runs of two-neighbour pixels between nodes are the **edges** (the wires).
  * a closed ring has no endpoints or branches at all; it still gets one
    **loop** node so connectivity queries can see it.

THE DIAGONAL-CONTACT PITFALL (why the adjacency rule below exists)
------------------------------------------------------------------
With plain 8-connectivity, the pixels around a corner or a T-junction touch
each other diagonally as well as through the junction pixel, which inflates
their neighbour counts and conjures phantom branch nodes out of a simple bend.
The fix is REDUCED adjacency: orthogonal neighbours always count; a diagonal
neighbour counts only if neither of the two orthogonal pixels it shares with us
is itself ink. (If one is ink, the connection already flows through it, so the
diagonal shortcut is redundant.) With this rule an L-bend is a clean 2-endpoint
path and a T-junction has exactly one branch pixel — see the unit tests.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

_ORTHO = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _neighbours(mask: np.ndarray, x: int, y: int) -> list[tuple[int, int]]:
    """Reduced-adjacency neighbours of skeleton pixel (x, y).

    Orthogonal ink neighbours always count.  A diagonal ink neighbour counts
    only when BOTH shared orthogonal pixels are empty — otherwise the
    connection already exists through the filled orthogonal pixel and counting
    the diagonal too would double-connect the corner.
    """
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for dx, dy in _ORTHO:
        nx_, ny_ = x + dx, y + dy
        if 0 <= nx_ < w and 0 <= ny_ < h and mask[ny_, nx_]:
            out.append((nx_, ny_))
    for dx, dy in _DIAG:
        nx_, ny_ = x + dx, y + dy
        if not (0 <= nx_ < w and 0 <= ny_ < h and mask[ny_, nx_]):
            continue
        side_a = (0 <= x + dx < w) and mask[y, x + dx]
        side_b = (0 <= y + dy < h) and mask[y + dy, x]
        if not side_a and not side_b:
            out.append((nx_, ny_))
    return out


def build_skeleton_graph(mask: np.ndarray, prune_len: int = 4) -> nx.MultiGraph:
    """Build a graph of a 1-px skeleton mask.

    Parameters
    ----------
    mask:
        Boolean HxW array, True where the skeleton is. Should already be
        skeletonised (1 px wide); thicker input degrades the node typing.
    prune_len:
        Whisker-spur threshold: an endpoint whose single edge is this many
        interior pixels or fewer, AND hangs off a larger structure, is removed.
        0 disables pruning. Pruning happens at the GRAPH level, so it can never
        split a wire or delete a standalone segment.

    Returns
    -------
    networkx.MultiGraph with:
      * node attrs:  pos=(x, y), kind in {"endpoint", "branch", "loop"}
      * edge attrs:  length (count of interior pixels), pixels (their (x,y) list)
    """
    ys, xs = np.where(mask)
    pixels = list(zip(xs.tolist(), ys.tolist()))

    nbr: dict[tuple[int, int], list[tuple[int, int]]] = {
        p: _neighbours(mask, p[0], p[1]) for p in pixels
    }
    endpoint_px = [p for p in pixels if len(nbr[p]) <= 1]
    branch_px = {p for p in pixels if len(nbr[p]) >= 3}

    g = nx.MultiGraph()
    node_of: dict[tuple[int, int], int] = {}
    next_id = [0]

    def _new_node(kind: str, pos: tuple[float, float]) -> int:
        nid = next_id[0]
        next_id[0] += 1
        g.add_node(nid, kind=kind, pos=pos)
        return nid

    for p in endpoint_px:
        node_of[p] = _new_node("endpoint", p)

    unvisited = set(branch_px)
    while unvisited:
        seed = unvisited.pop()
        cluster = [seed]
        frontier = [seed]
        while frontier:
            cx, cy = frontier.pop()
            for dx, dy in _ORTHO + _DIAG:
                q = (cx + dx, cy + dy)
                if q in unvisited:
                    unvisited.discard(q)
                    cluster.append(q)
                    frontier.append(q)
        centroid = (
            float(np.mean([c[0] for c in cluster])),
            float(np.mean([c[1] for c in cluster])),
        )
        nid = _new_node("branch", centroid)
        for c in cluster:
            node_of[c] = nid

    visited_interior: set[tuple[int, int]] = set()
    direct_seen: set[frozenset] = set()

    for start_px, start_node in node_of.items():
        for q in nbr[start_px]:
            if q in node_of:
                key = frozenset((start_px, q))
                if node_of[q] != start_node and key not in direct_seen:
                    direct_seen.add(key)
                    g.add_edge(start_node, node_of[q], length=0, pixels=[])
                continue
            if q in visited_interior:
                continue
            path = [q]
            prev, cur = start_px, q
            while True:
                visited_interior.add(cur)
                nxt = [r for r in nbr[cur] if r != prev]
                if not nxt:
                    tip = _new_node("endpoint", cur)
                    node_of[cur] = tip
                    g.add_edge(start_node, tip,
                               length=len(path) - 1, pixels=path[:-1])
                    break
                step = nxt[0]
                if step in node_of:
                    g.add_edge(start_node, node_of[step],
                               length=len(path), pixels=path)
                    break
                prev, cur = cur, step
                path.append(cur)

    leftovers = [p for p in pixels
                 if p not in node_of and p not in visited_interior]
    leftover_set = set(leftovers)
    while leftover_set:
        seed = leftover_set.pop()
        ring = [seed]
        frontier = [seed]
        while frontier:
            cur = frontier.pop()
            for r in nbr[cur]:
                if r in leftover_set:
                    leftover_set.discard(r)
                    ring.append(r)
                    frontier.append(r)
        nid = _new_node("loop", seed)
        g.add_edge(nid, nid, length=len(ring) - 1, pixels=ring[1:])

    if prune_len > 0:
        changed = True
        while changed:
            changed = False
            for node in list(g.nodes):
                if g.nodes[node]["kind"] != "endpoint" or g.degree(node) != 1:
                    continue
                (_, other, data), = g.edges(node, data=True)
                if other == node:
                    continue
                if data["length"] <= prune_len and g.degree(other) >= 2:
                    g.remove_node(node)
                    changed = True

    return g


def endpoint_positions(g: nx.MultiGraph) -> dict[int, tuple[int, int]]:
    """Map of node id -> (x, y) for every endpoint node in the graph."""
    return {n: data["pos"] for n, data in g.nodes(data=True)
            if data["kind"] == "endpoint"}
