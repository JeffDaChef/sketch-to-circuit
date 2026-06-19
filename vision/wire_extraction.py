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

KIND_MAP: dict[str, str] = {
    "resistor":       "R",
    "voltage source": "V",
    "capacitor":      "C",
    "diode":          "D",
}

_SUPPORTED_KINDS = set(KINDS.keys())

_PLACEHOLDERS: dict[str, str] = {
    "R": "1k", "V": "1", "C": "1u", "I": "1m", "D": "1",
}



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
            continue
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



def _point_in_bbox(pos: tuple[float, float], bbox: list[float], pad: float = 0.0) -> bool:
    x, y = pos
    return (bbox[0] - pad) <= x <= (bbox[2] + pad) and (bbox[1] - pad) <= y <= (bbox[3] + pad)


_COLLINEAR_DOT = -0.5


def _thread_cluster(graph: nx.MultiGraph, cluster: list[int]) -> bool:
    """Thread the crossing formed by a cluster of branch nodes into two wires.

    Two wires crossing fuse them into one electrical net. The crossing shows up in
    the skeleton as either ONE degree-4 node (a clean right-angle '+') or, when the
    wires meet at a shallow angle, TWO adjacent degree-3 nodes joined by a tiny
    edge. Handling both, we treat the whole in-box cluster as the crossing: collect
    the edges leaving it (ignoring edges internal to the cluster), and — since each
    wire continues roughly straight — pair the two that point most nearly OPPOSITE
    as one wire, the other two as the second wire. The cluster is then rebuilt as
    two separate nodes, so connected-components sees two nets.

    Returns False (leaving the graph untouched) when the cluster doesn't look like a
    clean two-wire crossing: not exactly four edges leave it, or a chosen pair isn't
    actually collinear (a 'hop'-drawn crossover already apart, a 3-way junction, or a
    messy multi-node tangle). Graceful by design — better fused than mis-split.
    """
    members = set(cluster)
    pts = [graph.nodes[n]["pos"] for n in cluster]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)

    external = []
    for n in cluster:
        for _, other, _, data in graph.edges(n, keys=True, data=True):
            if other not in members:
                external.append((other, data.get("length", 0), data.get("pixels", [])))
    if len(external) != 4:
        return False

    def direction(other: int) -> tuple[float, float]:
        ox, oy = graph.nodes[other]["pos"]
        dx, dy = ox - cx, oy - cy
        mag = math.hypot(dx, dy) or 1.0
        return dx / mag, dy / mag

    dirs = [direction(e[0]) for e in external]
    rest = [1, 2, 3]
    j = min(rest, key=lambda k: dirs[0][0] * dirs[k][0] + dirs[0][1] * dirs[k][1])
    rest.remove(j)
    pairs = [(0, j), (rest[0], rest[1])]

    for a, b in pairs:
        if dirs[a][0] * dirs[b][0] + dirs[a][1] * dirs[b][1] > _COLLINEAR_DOT:
            return False

    base = max(graph.nodes) + 1
    for n in cluster:
        graph.remove_node(n)
    for offset, (a, b) in enumerate(pairs):
        new = base + offset
        graph.add_node(new, kind="branch", pos=(cx, cy))
        for idx in (a, b):
            other, length, pixels = external[idx]
            graph.add_edge(new, other, length=length, pixels=pixels)
    return True


def thread_crossovers(graph: nx.MultiGraph, crossovers: list[dict]) -> int:
    """Thread every detected crossover in `graph`; return how many were split.

    `crossovers` are component-style dicts with kind "crossover" and a bbox. The
    branch nodes inside each box are treated as one crossing cluster and threaded
    (handles both the clean degree-4 '+' and the shallow-angle two-degree-3 case).
    Anything that doesn't look like a clean two-wire crossing is left alone.
    """
    threaded = 0
    for cx in crossovers:
        cluster = [n for n, d in graph.nodes(data=True)
                   if d["kind"] == "branch" and _point_in_bbox(d["pos"], cx["bbox"])]
        if cluster and _thread_cluster(graph, cluster):
            threaded += 1
    return threaded



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

    sides = [max(c["bbox"][2] - c["bbox"][0], c["bbox"][3] - c["bbox"][1])
             for c in components if c["kind"] in KIND_MAP]
    scale = float(np.median(sides)) if sides else 0.1 * math.hypot(w, h)

    if match_radius is None:
        match_radius = max(6, int(0.08 * scale))

    _MARGIN = 4
    _JUNCTION_TOUCH = 5
    _TT_CAP = max(40, int(0.35 * scale))
    _TT_TIGHT = max(16, int(0.20 * scale))
    _RESCUE_CAP = max(80, int(0.60 * scale))

    ink = _to_ink_mask(image)
    wire_only = _erase_components(ink, components, _MARGIN)
    skel = skeletonize(wire_only)
    graph = build_skeleton_graph(skel, prune_len=4)
    regions = _region_map((h, w), components, _MARGIN)

    thread_crossovers(graph, [c for c in components if c["kind"] == "crossover"])

    net_of_node: dict[int, int] = {}
    for net_id, nodes in enumerate(nx.connected_components(graph)):
        for n in nodes:
            net_of_node[n] = net_id

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

    two_terminal = [c for c in components if c["kind"] in KIND_MAP]
    junctions = [c for c in components if c["kind"] == "junction"]
    gnd_symbols = [c for c in components if c["kind"] == "ground"]

    uf = _UF()

    def _net_key(net_id: int) -> str:
        return f"wire_{net_id}"

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

    for rec in records:
        bbox = rec["comp"]["bbox"]
        for ep in endpoints:
            if not (rec["region"] & ep["regions"]):
                continue
            if rec["side"] == "g":
                hit = _bbox_border_dist(ep["pos"][0], ep["pos"][1],
                                        bbox) <= _MARGIN + match_radius
            else:
                hit = _in_face_band(ep["pos"], bbox, rec["side"],
                                    _MARGIN, match_radius)
            if hit:
                uf.union(rec["key"], _net_key(ep["net"]))
                rec["claimed"] = True

    for junc in junctions:
        jb = junc["bbox"]
        jc = ((jb[0] + jb[2]) / 2.0, (jb[1] + jb[3]) / 2.0)
        jkey = f"junc_{junc['name']}"
        uf.find(jkey)
        for ep in endpoints:
            if _bbox_border_dist(ep["pos"][0], ep["pos"][1],
                                 jb) <= _MARGIN + match_radius:
                uf.union(jkey, _net_key(ep["net"]))
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

    open_recs = [r for r in records if not r["on_junction"]]
    for i, a in enumerate(open_recs):
        for b in open_recs[i + 1:]:
            if a["comp"]["name"] == b["comp"]["name"]:
                continue
            if a["claimed"] and b["claimed"]:
                continue
            if not (a["region"] & b["region"]):
                continue
            d = math.hypot(a["point"][0] - b["point"][0],
                           a["point"][1] - b["point"][1])
            cap = _TT_CAP if not (a["claimed"] or b["claimed"]) \
                else _TT_TIGHT
            if d <= cap:
                uf.union(a["key"], b["key"])
                a["touch_matched"] = b["touch_matched"] = True

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
            "labeled": labeled,
            "graph": graph,
            "endpoints": endpoints,
            "regions": regions,
            "records": records,
            "terminal_nets": terminal_nets,
            "match_radius": match_radius,
            "erase_margin": _MARGIN,
        }
        return result, info
    return result
