"""Wire extraction — recovering electrical connectivity from a circuit image.

WHY THIS EXISTS
---------------
After the YOLO detector finds each component's bounding box we still need to
know HOW those components are wired together.  That is, we need the *netlist*:
"R1 connects net n1 to net n2; V1 connects n1 to 0 (ground); …"

This module recovers the netlist purely from pixels.  It is the second design:
the first one matched terminals to nearby wire "blobs" and needed a pile of
layout-specific fallbacks that did not survive new circuit layouts (frozen for
comparison in wire_extraction_baseline.py; full story in ROADMAP.md and
docs/wire_extraction_redesign_plan.md).

THE CORE IDEA
-------------
Erasing a component's bounding box *cuts* every wire that entered it — and each
cut leaves a skeleton ENDPOINT exactly where the wire attached to the component.
So instead of asking "what wire pixels are near this terminal?" (everything is
near a rail), we build a real graph of the wires (vision/skeleton_graph.py) and
ask "which CUT ENDPOINTS did this component's erasure create?"  Endpoints are
sparse and meaningful; proximity to them is evidence, not coincidence.

THE PIPELINE
------------
 1. Binarise the image (Otsu) to an ink mask.
 2. Erase every component's expanded bounding box; remember the erased
    rectangles as labelled REGIONS (overlapping boxes merge into one region —
    which is itself information: their components touch).
 3. Skeletonise the leftover wires and build the skeleton graph.  Each
    connected piece of the graph is one candidate net; each endpoint is a spot
    where some erasure cut a wire.
 4. Infer every component's terminal points from its bounding-box geometry
    (tall box → pins at top/bottom mid-edges, wide box → left/right).
 5. Connect terminals to nets using FOUR rules, ordered strongest evidence
    first; a weaker rule only fires when the stronger ones found nothing:
      (1) FACE BAND — endpoints lying along the terminal's face of the erased
          box, in the same region.  (The plain case: the wire was cut right
          where the pin is.)
      (2) JUNCTION DOTS — a junction's erasure also cuts wires; everything
          around the dot (cut endpoints + the nearest terminal of each
          touching component) is one electrical node.  This is the drawing
          convention doing its job.
      (3) TOUCHING TERMINALS — two still-unconnected terminals of different
          components, very close together inside one merged region, are
          connected (their joining wire was swallowed whole by the merged
          erasure — e.g. a source lead meeting a resistor lead directly).
      (4) REGION RESCUE — a terminal still unconnected (and not on a junction)
          takes the nearest endpoint of its own region, within a generous cap.
          This recovers leads that attach far outside their symbol's box
          (a voltage source's lead can sit ~90 px out because the box is
          inflated by its text label).
 6. Union-find merges everything; the group holding a ground symbol becomes
    net "0"; emit the Netlist (placeholder values — reading values is a
    different module's job).

Every rule is stated in terms of what erasure does to wires.  None of them
mention "vertical", "above the ground symbol", or any template's layout.
"""

from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np
from PIL import Image
from scipy.ndimage import label as nd_label
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

from solver.netlist import GROUND, KINDS, Netlist
from vision.skeleton_graph import build_skeleton_graph

# ---------------------------------------------------------------------------
# Kind mapping: human-readable component name -> SPICE one-letter code.
# "ground" and "junction" are NOT circuit components; they get special
# treatment (ground marks net "0"; junctions merge nearby terminals).
# Anything else (e.g. "text") is erased from the image but produces nothing.
# ---------------------------------------------------------------------------
KIND_MAP: dict[str, str] = {
    "resistor":       "R",
    "voltage source": "V",
    "capacitor":      "C",
    "diode":          "D",
    # "switch" would be "S" but Netlist.KINDS doesn't include it yet.
}

_SUPPORTED_KINDS = set(KINDS.keys())   # {"R", "C", "V", "I", "D"}

# Placeholder values written into the netlist (ignored by the equivalence
# check; value READING is a separate, later module).
_PLACEHOLDERS: dict[str, str] = {
    "R": "1k", "V": "1", "C": "1u", "I": "1m", "D": "1",
}


# ---------------------------------------------------------------------------
# Union-Find (path-compressed)
# ---------------------------------------------------------------------------

class _UF:
    """Minimal union-find with path compression."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        px, py = self.find(x), self.find(y)
        if px != py:
            self._parent[px] = py


# ---------------------------------------------------------------------------
# Image-level helpers
# ---------------------------------------------------------------------------

def _to_ink_mask(image: np.ndarray) -> np.ndarray:
    """Boolean mask, True where ink is (ink = darker than Otsu threshold)."""
    if image.ndim == 3:
        gray = rgb2gray(image)
    else:
        gray = image.astype(float)
        if gray.max() > 1.0:
            gray /= 255.0
    return gray < threshold_otsu(gray)


def _erase_components(ink: np.ndarray, components: list[dict],
                      margin: int) -> np.ndarray:
    """Zero out every component's margin-expanded bounding box.

    A "crossover" is NOT erased: it isn't a component but a wire feature (two
    wires crossing without connecting), and we want its crossing pixels to stay so
    the skeleton graph forms the node we later thread through.
    """
    erased = ink.copy()
    h, w = erased.shape
    for comp in components:
        if comp["kind"] == "crossover":
            continue
        xmin, ymin, xmax, ymax = comp["bbox"]
        r0, r1 = max(0, int(ymin) - margin), min(h, int(ymax) + margin)
        c0, c1 = max(0, int(xmin) - margin), min(w, int(xmax) + margin)
        erased[r0:r1, c0:c1] = False
    return erased


def _region_map(shape: tuple[int, int], components: list[dict],
                margin: int) -> np.ndarray:
    """Label the union of all expanded bounding boxes into connected regions.

    Overlapping/touching boxes merge into ONE region — deliberately: when two
    components' erasures merge, the wire evidence between them is gone, and
    "same region" is the licence the weaker matching rules need.
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for comp in components:
        if comp["kind"] == "crossover":
            continue                      # not a component; see _erase_components
        xmin, ymin, xmax, ymax = comp["bbox"]
        r0, r1 = max(0, int(ymin) - margin), min(h, int(ymax) + margin)
        c0, c1 = max(0, int(xmin) - margin), min(w, int(xmax) + margin)
        mask[r0:r1, c0:c1] = True
    labeled, _ = nd_label(mask, structure=np.ones((3, 3), dtype=int))
    return labeled


def _regions_near(regions: np.ndarray, x: float, y: float,
                  win: int = 5) -> set[int]:
    """Region ids found within a small window of (x, y).

    Endpoints sit just OUTSIDE the erased rectangles that cut them, so we look
    in a window rather than at the exact pixel.  A point may be near two
    regions at once; we return all of them.
    """
    h, w = regions.shape
    x0, x1 = max(0, int(x) - win), min(w, int(x) + win + 1)
    y0, y1 = max(0, int(y) - win), min(h, int(y) + win + 1)
    vals = regions[y0:y1, x0:x1]
    return set(int(v) for v in np.unique(vals) if v > 0)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _terminal_points(bbox: list[float]) -> tuple[tuple[float, float],
                                                 tuple[float, float]]:
    """The two pin points of a 2-terminal component, from bbox geometry.

    A component's leads leave through the two ends of its LONGER axis: a tall
    box pins at top/bottom mid-edges, a wide box at left/right.  Convention:
    t0 = top (vertical) or left (horizontal); t1 = the opposite end.
    """
    xmin, ymin, xmax, ymax = bbox
    cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    if (ymax - ymin) >= (xmax - xmin):
        return (cx, ymin), (cx, ymax)
    return (xmin, cy), (xmax, cy)


def _bbox_border_dist(px: float, py: float, bbox: list[float]) -> float:
    """Distance from a point to a bbox border: negative inside, 0 on it."""
    xmin, ymin, xmax, ymax = bbox
    if xmin <= px <= xmax and ymin <= py <= ymax:
        return -min(px - xmin, xmax - px, py - ymin, ymax - py)
    nearest_x = max(xmin, min(xmax, px))
    nearest_y = max(ymin, min(ymax, py))
    return math.hypot(px - nearest_x, py - nearest_y)


def _in_face_band(ep: tuple[float, float], bbox: list[float],
                  side: str, margin: int, radius: float) -> bool:
    """Is endpoint `ep` in the band along one face of the ERASED bbox?

    The band is the face line of the margin-expanded box, thickened by
    `radius` in the face's normal direction and widened by `radius` along it —
    i.e. "the strip where this face's erasure would have cut a wire".
    """
    x, y = ep
    xmin, ymin, xmax, ymax = bbox
    ex0, ey0 = xmin - margin, ymin - margin
    ex1, ey1 = xmax + margin, ymax + margin
    vertical = (ymax - ymin) >= (xmax - xmin)
    if vertical:
        face_y = ey0 if side == "t0" else ey1
        return abs(y - face_y) <= radius and (ex0 - radius) <= x <= (ex1 + radius)
    face_x = ex0 if side == "t0" else ex1
    return abs(x - face_x) <= radius and (ey0 - radius) <= y <= (ey1 + radius)


# ---------------------------------------------------------------------------
# Crossover threading (lifting the no-crossing-wires constraint)
# ---------------------------------------------------------------------------

def _point_in_bbox(pos: tuple[float, float], bbox: list[float], pad: float = 0.0) -> bool:
    x, y = pos
    return (bbox[0] - pad) <= x <= (bbox[2] + pad) and (bbox[1] - pad) <= y <= (bbox[3] + pad)


def _thread_one(graph: nx.MultiGraph, node: int) -> bool:
    """Split one degree-4 crossing node into two collinear pass-throughs.

    Two wires crossing (a plain '+') skeletonise to a single branch node of
    degree 4, which wrongly fuses the two wires into one electrical net. We undo
    that: pair the four incident edges by direction — each wire continues roughly
    straight, so the two most *opposite* edges belong to the same wire — and
    rebuild the node as two separate nodes, one per wire. The crossing is then a
    pass-over, not a connection.
    """
    inc = list(graph.edges(node, keys=True, data=True))    # (node, other, key, data)
    if len(inc) != 4 or any(v == node for _, v, _, _ in inc):
        return False                                       # not a clean 4-way / has a self-loop
    npos = graph.nodes[node]["pos"]

    def direction(other: int) -> tuple[float, float]:
        op = graph.nodes[other]["pos"]
        dx, dy = op[0] - npos[0], op[1] - npos[1]
        n = math.hypot(dx, dy) or 1.0
        return dx / n, dy / n

    dirs = [direction(e[1]) for e in inc]
    # Pair edge 0 with whichever of the others points most nearly opposite to it
    # (smallest, i.e. most negative, dot product); the remaining two are the pair.
    rest = [1, 2, 3]
    j = min(rest, key=lambda k: dirs[0][0] * dirs[k][0] + dirs[0][1] * dirs[k][1])
    rest.remove(j)
    pairs = ((inc[0], inc[j]), (inc[rest[0]], inc[rest[1]]))

    base = max(graph.nodes) + 1
    graph.remove_node(node)                                # drops the 4 fused edges
    for offset, (e1, e2) in enumerate(pairs):
        new = base + offset
        graph.add_node(new, kind="branch", pos=npos)
        for _, v, _, data in (e1, e2):
            graph.add_edge(new, v, length=data.get("length", 0), pixels=data.get("pixels", []))
    return True


def thread_crossovers(graph: nx.MultiGraph, crossovers: list[dict]) -> int:
    """Thread every detected crossover in `graph`; return how many were split.

    `crossovers` are component-style dicts with kind "crossover" and a bbox. For
    each, any degree-4 branch node sitting inside the box is split. Anything that
    isn't a clean 4-way (e.g. a 'hop'-drawn crossover whose wires never touched,
    so the skeleton already keeps them apart) is left alone — graceful by design.
    """
    threaded = 0
    for cx in crossovers:
        bbox = cx["bbox"]
        targets = [n for n, d in graph.nodes(data=True)
                   if d["kind"] == "branch" and _point_in_bbox(d["pos"], bbox)]
        for node in targets:
            if graph.has_node(node) and _thread_one(graph, node):
                threaded += 1
    return threaded


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_netlist(
    image: np.ndarray | Image.Image,
    components: list[dict],
    *,
    match_radius: int | None = None,
    debug: bool = False,
) -> Netlist | tuple[Netlist, dict[str, Any]]:
    """Recover a Netlist from a circuit image and component bounding boxes.

    Parameters
    ----------
    image:
        HxW or HxWx3 numpy array, or a PIL Image.
    components:
        List of dicts like
        ``{"name": "R1", "kind": "resistor", "value": "10k", "bbox": [x0,y0,x1,y1]}``
        (pixel coords, top-left origin) — the synthetic ground-truth format,
        later the YOLO detection format.
    match_radius:
        Close-range matching distance (face bands, junction snapping).
        Default: max(6, 1% of the image diagonal).
    debug:
        If True, return ``(netlist, info)`` where ``info`` holds the skeleton
        graph, endpoints, regions and per-terminal net assignments — the
        payload vision/debug_viz.py renders into a picture.
    """
    if isinstance(image, Image.Image):
        image = np.asarray(image)
    h, w = image.shape[:2]

    # All matching distances scale with COMPONENT size, not image size: the
    # geometry that matters (lead lengths, symbol spacing) is set by how big
    # the circuit elements are drawn, and the same circuit can be rendered at
    # many pixels-per-element. S = the median long side of the component
    # boxes ~= one element's drawn length.
    sides = [max(c["bbox"][2] - c["bbox"][0], c["bbox"][3] - c["bbox"][1])
             for c in components if c["kind"] in KIND_MAP]
    scale = float(np.median(sides)) if sides else 0.1 * math.hypot(w, h)

    if match_radius is None:
        match_radius = max(6, int(0.08 * scale))

    _MARGIN = 4                                   # erase margin (px)
    _JUNCTION_TOUCH = 5                           # junction-to-bbox contact (px)
    _TT_CAP = max(40, int(0.35 * scale))          # touching-terminals cap
    _TT_TIGHT = max(16, int(0.20 * scale))        # ...when one side has wire evidence
    _RESCUE_CAP = max(80, int(0.60 * scale))      # region-rescue cap

    # --- steps 1-3: ink -> erased -> skeleton graph -----------------------
    ink = _to_ink_mask(image)
    wire_only = _erase_components(ink, components, _MARGIN)
    skel = skeletonize(wire_only)
    graph = build_skeleton_graph(skel, prune_len=4)
    regions = _region_map((h, w), components, _MARGIN)

    # Lift the no-crossing-wires constraint: where the detector marked a crossover,
    # split the fused degree-4 node so the two wires stay separate nets. Done
    # BEFORE connected-components, which is what turns the graph into nets.
    thread_crossovers(graph, [c for c in components if c["kind"] == "crossover"])

    # Each connected piece of the skeleton graph is one candidate net.
    net_of_node: dict[int, int] = {}
    for net_id, nodes in enumerate(nx.connected_components(graph)):
        for n in nodes:
            net_of_node[n] = net_id

    # All cut endpoints, with their net and nearby region(s).
    endpoints: list[dict] = []
    for n, data in graph.nodes(data=True):
        if data["kind"] != "endpoint":
            continue
        pos = data["pos"]
        endpoints.append({
            "pos": pos,
            "net": net_of_node[n],
            "regions": _regions_near(regions, pos[0], pos[1]),
        })

    # --- step 4: terminal records ------------------------------------------
    two_terminal = [c for c in components if c["kind"] in KIND_MAP]
    junctions = [c for c in components if c["kind"] == "junction"]
    gnd_symbols = [c for c in components if c["kind"] == "ground"]

    uf = _UF()

    def _net_key(net_id: int) -> str:
        return f"wire_{net_id}"

    # One record per electrical terminal (2 per component, 1 per ground).
    records: list[dict] = []
    for comp in two_terminal:
        p0, p1 = _terminal_points(comp["bbox"])
        for side, point in (("t0", p0), ("t1", p1)):
            records.append({
                "key": f"term_{comp['name']}_{side}",
                "comp": comp, "side": side, "point": point,
                "region": _regions_near(regions, point[0], point[1]),
                "claimed": False, "on_junction": False, "touch_matched": False,
            })
    for comp in gnd_symbols:
        xmin, ymin, xmax, ymax = comp["bbox"]
        centre = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)
        records.append({
            "key": f"term_{comp['name']}_g",
            "comp": comp, "side": "g", "point": centre,
            "region": _regions_near(regions, centre[0], centre[1]),
            "claimed": False, "on_junction": False,
        })
    for rec in records:
        uf.find(rec["key"])

    # --- step 5, rule (1): FACE BAND ---------------------------------------
    # Endpoints lying along this terminal's face of the erased box, cut by the
    # same region this component lives in: the wire attached right here.
    for rec in records:
        bbox = rec["comp"]["bbox"]
        for ep in endpoints:
            if not (rec["region"] & ep["regions"]):
                continue
            if rec["side"] == "g":
                # Ground has one lead but its symbol's orientation varies, so
                # accept a cut anywhere snug around the box.
                hit = _bbox_border_dist(ep["pos"][0], ep["pos"][1],
                                        bbox) <= _MARGIN + match_radius
            else:
                hit = _in_face_band(ep["pos"], bbox, rec["side"],
                                    _MARGIN, match_radius)
            if hit:
                uf.union(rec["key"], _net_key(ep["net"]))
                rec["claimed"] = True

    # --- step 5, rule (2): JUNCTION DOTS ------------------------------------
    # A junction dot's erasure cut the wires running through it; everything
    # around the dot is one node: the cut endpoints, plus — for each component
    # the dot touches — that component's nearest terminal.
    for junc in junctions:
        jb = junc["bbox"]
        jc = ((jb[0] + jb[2]) / 2.0, (jb[1] + jb[3]) / 2.0)
        jkey = f"junc_{junc['name']}"
        uf.find(jkey)
        for ep in endpoints:
            if _bbox_border_dist(ep["pos"][0], ep["pos"][1],
                                 jb) <= _MARGIN + match_radius:
                uf.union(jkey, _net_key(ep["net"]))
        # Nearest terminal of each touching component joins the junction node.
        by_comp: dict[str, list[dict]] = {}
        for rec in records:
            by_comp.setdefault(rec["comp"]["name"], []).append(rec)
        for name, recs in by_comp.items():
            bbox = recs[0]["comp"]["bbox"]
            if _bbox_border_dist(jc[0], jc[1], bbox) <= _JUNCTION_TOUCH:
                nearest = min(recs, key=lambda r: math.hypot(
                    jc[0] - r["point"][0], jc[1] - r["point"][1]))
                uf.union(jkey, nearest["key"])
                nearest["on_junction"] = True

    # --- step 5, rule (3): TOUCHING TERMINALS -------------------------------
    # Two terminals of different components, close together inside one merged
    # region: their joining wire was swallowed by the erasure, so adjacency IS
    # the connection evidence.  Two tiers:
    #   * both still unresolved -> connect within the normal cap;
    #   * one already wire-matched -> connect only at TIGHT range (2x match
    #     radius): terminals that practically touch are connected no matter
    #     what, but a resolved terminal shouldn't pull in merely-nearby ones.
    #     (Found by poking: a ground symbol pressed against a source's foot
    #     must inherit its net even after the source matched a wire.)
    open_recs = [r for r in records if not r["on_junction"]]
    for i, a in enumerate(open_recs):
        for b in open_recs[i + 1:]:
            if a["comp"]["name"] == b["comp"]["name"]:
                continue                      # never short one component
            if a["claimed"] and b["claimed"]:
                continue                      # both already have wire evidence
            if not (a["region"] & b["region"]):
                continue
            d = math.hypot(a["point"][0] - b["point"][0],
                           a["point"][1] - b["point"][1])
            cap = _TT_CAP if not (a["claimed"] or b["claimed"]) \
                else _TT_TIGHT
            if d <= cap:
                uf.union(a["key"], b["key"])
                a["touch_matched"] = b["touch_matched"] = True

    # --- step 5, rule (4): REGION RESCUE ------------------------------------
    # A terminal with no wire evidence and no junction takes the single
    # nearest endpoint of its own region, within a generous cap.  Region-
    # scoped + endpoints-only keeps this from grabbing a passing rail.
    for rec in records:
        if rec["claimed"] or rec["on_junction"]:
            continue
        best, best_d = None, _RESCUE_CAP
        for ep in endpoints:
            if not (rec["region"] & ep["regions"]):
                continue
            d = math.hypot(rec["point"][0] - ep["pos"][0],
                           rec["point"][1] - ep["pos"][1])
            if d < best_d:
                best, best_d = ep, d
        if best is not None:
            uf.union(rec["key"], _net_key(best["net"]))
            rec["claimed"] = True

    # --- step 6: name nets, ground last, build the Netlist ------------------
    gnd_roots = {uf.find(f"term_{g['name']}_g") for g in gnd_symbols}

    net_names: dict[str, str] = {}
    _counter = [0]

    def _name_of(key: str) -> str:
        root = uf.find(key)
        if root in gnd_roots:
            return GROUND
        if root not in net_names:
            _counter[0] += 1
            net_names[root] = f"n{_counter[0]}"
        return net_names[root]

    result = Netlist()
    terminal_nets: dict[str, str] = {}
    for comp in two_terminal:
        code = KIND_MAP[comp["kind"]]
        if code not in _SUPPORTED_KINDS:
            continue
        n0 = _name_of(f"term_{comp['name']}_t0")
        n1 = _name_of(f"term_{comp['name']}_t1")
        terminal_nets[f"{comp['name']}_t0"] = n0
        terminal_nets[f"{comp['name']}_t1"] = n1
        result.add(code, comp["name"], _PLACEHOLDERS.get(code, "1"), n0, n1)

    if debug:
        labeled, _ = nd_label(skel, structure=np.ones((3, 3), dtype=int))
        info: dict[str, Any] = {
            "labeled": labeled,            # skeleton pixels by connected wire
            "graph": graph,                # the full skeleton graph
            "endpoints": endpoints,        # cut points with nets + regions
            "regions": regions,            # erased-rectangle region labels
            "records": records,            # per-terminal matching outcomes
            "terminal_nets": terminal_nets,
            "match_radius": match_radius,
            "erase_margin": _MARGIN,
        }
        return result, info
    return result
