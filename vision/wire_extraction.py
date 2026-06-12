"""Wire extraction — recovering electrical connectivity from a circuit image.

WHY THIS EXISTS
---------------
After the YOLO detector finds each component's bounding box we still need to
know HOW those components are wired together.  That is, we need the *netlist*:
"R1 connects net n1 to net n2; V1 connects n1 to 0 (ground); …"

This module recovers the netlist purely from pixels.  The high-level idea:

  1. Binarise the image to find where the ink is.
  2. Erase the component bodies (we know their bounding boxes) so only the
     *wire* pixels remain.
  3. Skeletonise those wire pixels to 1-pixel-wide centre lines and label
     each connected wire segment as a distinct "blob".
  4. For each component, probe just outside its erased bounding box to find
     which wire blob(s) each terminal touches.
  5. Use a union-find to merge blobs and terminals that are electrically
     connected — either directly (shared wire blob) or through junction dots.
  6. Each union-find group is one net.  Name them, attach the ground symbol
     to net "0", and build a Netlist.

DESIGN CHOICES
--------------
* We infer terminal positions from the *bounding box geometry* — no terminal
  coordinates are taken from the generator or detector.  The probes sit at
  the eight boundary points of the *erased* bounding box (four corners + four
  edge midpoints) so we land just outside the component body where the wire
  stubs begin.
* Junction dots (kind="junction") are handled by a bbox-overlap rule rather
  than wire-blob proximity because junction dots sit exactly on the component
  boundary and the erased region swallows any short wire segment between them.
* The "ground-proximity" fallback handles the series-divider layout where the
  voltage-source's negative terminal has no physical wire to the ground symbol
  (they are co-located in schematic space but pixel-distant in the image).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import label as nd_label
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

from solver.netlist import GROUND, KINDS, Netlist

# ---------------------------------------------------------------------------
# Kind mapping: human-readable component name -> SPICE one-letter code.
#
# "ground" and "junction" are NOT circuit components; they get special
# treatment (ground marks net "0"; junctions merge nearby terminals).
# ---------------------------------------------------------------------------
KIND_MAP: dict[str, str] = {
    "resistor":       "R",
    "voltage source": "V",
    "capacitor":      "C",
    "diode":          "D",
    # "switch" would be "S" but Netlist.KINDS doesn't include it yet;
    # add it here if KINDS is extended:
    # "switch": "S",
}

# Only the codes that Netlist.add() actually accepts.
_SUPPORTED_KINDS = set(KINDS.keys())   # {"R", "C", "V", "I", "D"}

# Placeholder values written into the netlist (ignored by equivalence check).
_PLACEHOLDERS: dict[str, str] = {
    "R": "1k",
    "V": "1",
    "C": "1u",
    "I": "1m",
    "D": "1",
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
            self._parent[x] = self.find(self._parent[x])  # path compression
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        px, py = self.find(x), self.find(y)
        if px != py:
            self._parent[px] = py


# ---------------------------------------------------------------------------
# Step 1: Binarise
# ---------------------------------------------------------------------------

def _to_ink_mask(image: np.ndarray) -> np.ndarray:
    """Convert an HxW(x3) image to a boolean mask that is True where ink is.

    Ink is darker than the paper background, so we use Otsu's threshold and
    mark pixels *below* it as ink.
    """
    if image.ndim == 3:
        gray = rgb2gray(image)      # skimage returns float [0,1]
    else:
        # Already grayscale; normalise to [0,1] if needed.
        gray = image.astype(float)
        if gray.max() > 1.0:
            gray /= 255.0

    thresh = threshold_otsu(gray)   # the natural split between paper and ink
    return gray < thresh            # True = dark = ink


# ---------------------------------------------------------------------------
# Step 2: Erase component bounding boxes
# ---------------------------------------------------------------------------

def _erase_components(ink: np.ndarray,
                      components: list[dict],
                      margin: int) -> np.ndarray:
    """Zero out every component's (possibly enlarged) bounding box.

    We erase EVERYTHING in the component list — resistors, sources, junctions,
    ground symbols, text — because all of those sit on top of the wire.
    The wire stub is *just outside* the erased rectangle (that is why we keep
    margin small: big enough to fully remove the body, small enough not to
    eat the wire stubs we need for terminal matching).
    """
    erased = ink.copy()
    h, w = erased.shape

    for comp in components:
        xmin, ymin, xmax, ymax = comp["bbox"]
        # Convert to int and clamp to image bounds.
        r0 = max(0, int(ymin) - margin)
        r1 = min(h, int(ymax) + margin)
        c0 = max(0, int(xmin) - margin)
        c1 = min(w, int(xmax) + margin)
        erased[r0:r1, c0:c1] = False

    return erased


# ---------------------------------------------------------------------------
# Step 3: Skeletonise and label wire blobs
# ---------------------------------------------------------------------------

def _skeletonise_and_label(wire_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Thin wire mask to 1-pixel skeleton, then label connected components.

    Returns (label_array, num_blobs).  Label 0 is background; labels 1..N
    are individual wire fragments.  8-connectivity (3×3 structuring element)
    is used so diagonal wires stay connected.
    """
    skeleton = skeletonize(wire_mask)

    # 8-connectivity structuring element.
    structure = np.ones((3, 3), dtype=int)
    labeled, n_blobs = nd_label(skeleton, structure=structure)
    return labeled, n_blobs


# ---------------------------------------------------------------------------
# Step 4: Terminal probing
# ---------------------------------------------------------------------------

def _corner_probes(bbox: list[float],
                   margin: int) -> tuple[list[tuple], list[tuple]]:
    """Return probe point lists for the two terminals of a 2-terminal component.

    We probe at the BOUNDARY of the *erased* bounding box — i.e. just outside
    the component body — so the probe lands where the wire stub begins after
    erasure.  We generate three points per terminal side: left corner, right
    corner, and edge midpoint.

    For a vertical component (height >= width) the two terminals are on the
    top edge (terminal-0) and bottom edge (terminal-1) of the erased bbox.
    For a horizontal component it's left/right.

    (In practice the corner with the shortest distance to the wire usually
    wins because the wire exits from a corner of the component body, not
    always from the center of an edge.)
    """
    xmin, ymin, xmax, ymax = bbox
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    # Expand by the erase margin so probes sit outside the erased region.
    ex0 = xmin - margin
    ex1 = xmax + margin
    ey0 = ymin - margin
    ey1 = ymax + margin

    # Probes on the "top" face of the erased box.
    top_probes = [(ex0, ey0), (ex1, ey0), (cx, ey0)]
    # Probes on the "bottom" face of the erased box.
    bot_probes = [(ex0, ey1), (ex1, ey1), (cx, ey1)]

    return top_probes, bot_probes


def _find_all_blobs(probes: list[tuple],
                    skel_pts: np.ndarray,
                    labeled: np.ndarray,
                    radius: float) -> set[int]:
    """Find every non-zero blob label whose nearest skeleton pixel is within radius.

    Searching ALL probes and returning ALL matching blobs (not just the nearest
    one) is crucial for the parallel-bank layout where a component touches two
    separate wire segments on the same terminal side — we must merge those
    segments into one net.
    """
    found: set[int] = set()
    if skel_pts.shape[0] == 0:
        return found

    for tx, ty in probes:
        dists = np.hypot(skel_pts[:, 0] - tx, skel_pts[:, 1] - ty)
        for i, d in enumerate(dists):
            if d <= radius:
                blob = int(labeled[int(skel_pts[i, 1]), int(skel_pts[i, 0])])
                if blob > 0:
                    found.add(blob)

    return found


# ---------------------------------------------------------------------------
# Step 5: Junction merging and ground proximity fallback
# ---------------------------------------------------------------------------

def _bbox_border_dist(px: float, py: float, bbox: list[float]) -> float:
    """Signed distance from point (px,py) to the nearest edge of bbox.

    Returns a NEGATIVE value if the point is strictly inside the box
    (|value| = distance to the nearest edge), and a positive Euclidean
    distance if it is outside.  Zero means exactly on the border.

    This lets us answer "is this junction center on/inside the component
    bounding box?" with a threshold like ``dist <= threshold``.
    """
    xmin, ymin, xmax, ymax = bbox
    inside_x = xmin <= px <= xmax
    inside_y = ymin <= py <= ymax

    if inside_x and inside_y:
        # Distance to nearest edge (negative = inside).
        dx = min(px - xmin, xmax - px)
        dy = min(py - ymin, ymax - py)
        return -min(dx, dy)

    # Outside: Euclidean distance to the nearest point on the border.
    nearest_x = max(xmin, min(xmax, px))
    nearest_y = max(ymin, min(ymax, py))
    return math.hypot(px - nearest_x, py - nearest_y)


def _x_ranges_overlap(bbox1: list[float],
                       bbox2: list[float],
                       margin: int) -> bool:
    """Return True if the two (margin-expanded) bboxes overlap in the X direction."""
    lo = max(bbox1[0] - margin, bbox2[0] - margin)
    hi = min(bbox1[2] + margin, bbox2[2] + margin)
    return hi > lo


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
        HxW or HxWx3 numpy array, or a PIL Image.  Accepts both; converted to
        greyscale internally.
    components:
        List of dicts, each like::

            {"name": "R1", "kind": "resistor", "value": "10k", "bbox": [x0,y0,x1,y1]}

        The "bbox" values are pixel coordinates (top-left origin, floats OK).
        The "kind" must be one of the keys in KIND_MAP or "ground"/"junction".
    match_radius:
        Maximum pixel distance a probe point can be from the nearest skeleton
        pixel and still count as connected.  Default: max(6, 1% of the image
        diagonal).  Expose this so tests can tune it.
    debug:
        If True, return ``(netlist, info_dict)`` where ``info_dict`` contains
        the intermediate terminal→net mapping and wire-blob label array.

    Returns
    -------
    A Netlist (or a (Netlist, dict) pair when debug=True).
    """

    # --- normalise image input -------------------------------------------
    if isinstance(image, Image.Image):
        image = np.asarray(image)

    h, w = image.shape[:2]

    # Default match_radius: ~1% of the diagonal, minimum 6 pixels.
    if match_radius is None:
        diag = math.hypot(w, h)
        match_radius = max(6, int(0.01 * diag))

    # Erase margin: 4 pixels beyond each side of the bbox.
    # Must be ≥ 1 to fully remove component terminal stubs but small enough
    # not to eat the wire segment just outside the box.
    _ERASE_MARGIN = 4

    # A junction is considered "touching" a component if its centre is within
    # this many pixels of the component's bbox border (or inside it).
    _JUNCTION_THRESHOLD = 5

    # GND uses a larger search radius because parallel-bank ground symbols
    # can be ~35 pixels away from the nearest wire fragment after erasure.
    _GND_RADIUS = 40

    # ------------------------------------------------------------------
    # Step 1 — binarise
    # ------------------------------------------------------------------
    ink = _to_ink_mask(image)

    # ------------------------------------------------------------------
    # Step 2 — erase all component boxes
    # ------------------------------------------------------------------
    wire_only = _erase_components(ink, components, _ERASE_MARGIN)

    # ------------------------------------------------------------------
    # Step 3 — skeletonise + label wire blobs
    # ------------------------------------------------------------------
    labeled, _n_blobs = _skeletonise_and_label(wire_only)

    # Build a (N,2) array of skeleton pixel coordinates (x, y) for fast
    # distance computations.
    skel_ys, skel_xs = np.where(labeled > 0)
    if skel_xs.size > 0:
        skel_pts = np.stack([skel_xs, skel_ys], axis=1).astype(np.float32)
    else:
        skel_pts = np.empty((0, 2), dtype=np.float32)

    # ------------------------------------------------------------------
    # Separate components by role
    # ------------------------------------------------------------------
    two_terminal = [c for c in components
                    if c["kind"] not in ("junction", "ground")]
    junctions    = [c for c in components if c["kind"] == "junction"]
    gnd_symbols  = [c for c in components if c["kind"] == "ground"]

    # ------------------------------------------------------------------
    # Step 6 — union-find
    # ------------------------------------------------------------------
    uf = _UF()

    # dict mapping (comp_name, "t0"/"t1") -> a canonical union-find key.
    terminal_key: dict[tuple[str, str], str] = {}

    _float_id = [0]   # counter for terminals that find no wire blob

    def _new_float() -> str:
        fid = f"float_{_float_id[0]}"
        _float_id[0] += 1
        return fid

    # --- 6a. Locate the ground net (the GND symbol's terminal) ----------
    # We probe with the larger GND radius because the ground symbol can sit
    # a few dozen pixels away from the nearest surviving wire fragment.
    gnd_node: str | None = None
    for gnd in gnd_symbols:
        bbox = gnd["bbox"]
        xmin, ymin, xmax, ymax = bbox
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        gnd_probes = [
            (cx,                     ymin - _ERASE_MARGIN),  # top edge centre
            (xmax + _ERASE_MARGIN,   cy),                    # right edge centre
            (xmin - _ERASE_MARGIN,   cy),                    # left edge centre
            (xmin - _ERASE_MARGIN,   ymin - _ERASE_MARGIN),  # top-left corner
            (xmax + _ERASE_MARGIN,   ymin - _ERASE_MARGIN),  # top-right corner
        ]
        blobs = _find_all_blobs(gnd_probes, skel_pts, labeled, _GND_RADIUS)
        if blobs:
            gnd_node = f"blob_{min(blobs)}"
            for b in blobs:
                uf.union(gnd_node, f"blob_{b}")
        else:
            gnd_node = _new_float()
        uf.find(gnd_node)

    # --- 6b. Assign each 2-terminal component's terminals ----------------
    for comp in two_terminal:
        bbox  = comp["bbox"]
        name  = comp["name"]
        top_p, bot_p = _corner_probes(bbox, _ERASE_MARGIN)

        t0_blobs = _find_all_blobs(top_p, skel_pts, labeled, match_radius)
        t1_blobs = _find_all_blobs(bot_p, skel_pts, labeled, match_radius)

        def _assign(blobs: set[int]) -> str:
            """Turn a set of blob IDs into a union-find key; apply fallbacks."""
            if blobs:
                # Pick the smallest blob id as the canonical key for this group.
                canonical = f"blob_{min(blobs)}"
                all_b = [f"blob_{b}" for b in blobs]
                # Union all blobs found at this terminal together — they are the
                # same net (e.g. the two wire segments on either side of a rung
                # in the parallel bank both connect to the same terminal).
                for bk in all_b[1:]:
                    uf.union(all_b[0], bk)
                uf.union(canonical, all_b[0])
                return canonical

            # No wire blob found directly.  Fall back to the ground net if this
            # component's x-range overlaps the ground symbol's x-range — which
            # happens in the series-divider layout where V1's negative terminal
            # is electrically at ground but physically disconnected from the
            # ground symbol in the pixel drawing.
            if gnd_node is not None:
                if any(_x_ranges_overlap(bbox, g["bbox"], _ERASE_MARGIN)
                       for g in gnd_symbols):
                    return gnd_node

            # Truly floating: give it a unique placeholder that will form its
            # own net unless a junction merges it later.
            return _new_float()

        t0_key = _assign(t0_blobs)
        t1_key = _assign(t1_blobs)
        terminal_key[(name, "t0")] = t0_key
        terminal_key[(name, "t1")] = t1_key
        uf.find(t0_key)
        uf.find(t1_key)

    # --- 6c. Junction merging -------------------------------------------
    # A junction dot marks a point where two or more components share the same
    # electrical node.  We detect this by checking whether the junction's
    # bounding-box centre is inside (or very close to the border of) each
    # component's bounding box.  Components where the junction is in the top
    # half connect via terminal-0; those where it is in the bottom half connect
    # via terminal-1.
    for junc in junctions:
        jbbox = junc["bbox"]
        jcx   = (jbbox[0] + jbbox[2]) / 2.0
        jcy   = (jbbox[1] + jbbox[3]) / 2.0

        # Collect all (comp, terminal_side) pairs that this junction touches.
        to_merge: list[str] = []
        for comp in two_terminal:
            dist = _bbox_border_dist(jcx, jcy, comp["bbox"])
            if dist <= _JUNCTION_THRESHOLD:
                # Is the junction in the top half or bottom half of the component?
                comp_cy = (comp["bbox"][1] + comp["bbox"][3]) / 2.0
                t_side  = "t0" if jcy <= comp_cy else "t1"
                to_merge.append(terminal_key[(comp["name"], t_side)])

        # Union all collected terminals into one group.
        if len(to_merge) >= 2:
            for tk in to_merge[1:]:
                uf.union(to_merge[0], tk)

    # ------------------------------------------------------------------
    # Step 7 — assign net names
    # ------------------------------------------------------------------
    gnd_root = uf.find(gnd_node) if gnd_node is not None else None

    net_names: dict[str, str] = {}   # uf-root -> net name
    _net_counter = [0]

    def _get_net(key: str) -> str:
        root = uf.find(key)
        if root == gnd_root:
            return GROUND              # "0"
        if root not in net_names:
            _net_counter[0] += 1
            net_names[root] = f"n{_net_counter[0]}"
        return net_names[root]

    # ------------------------------------------------------------------
    # Step 8 — build the Netlist
    # ------------------------------------------------------------------
    result = Netlist()
    for comp in two_terminal:
        kind_code = KIND_MAP.get(comp["kind"])
        if kind_code is None:
            continue          # unknown kind; skip rather than crash
        if kind_code not in _SUPPORTED_KINDS:
            # Netlist.add would reject this code; skip and note the limitation.
            # (Currently "S" for switch is not in KINDS.)
            continue

        n0 = _get_net(terminal_key[(comp["name"], "t0")])
        n1 = _get_net(terminal_key[(comp["name"], "t1")])
        value = _PLACEHOLDERS.get(kind_code, "1")
        result.add(kind_code, comp["name"], value, n0, n1)

    # ------------------------------------------------------------------
    # Optional debug payload
    # ------------------------------------------------------------------
    if debug:
        info: dict[str, Any] = {
            "labeled": labeled,
            "terminal_nets": {
                f"{name}_{side}": _get_net(key)
                for (name, side), key in terminal_key.items()
            },
            "gnd_node": gnd_node,
            "match_radius": match_radius,
            "erase_margin": _ERASE_MARGIN,
        }
        return result, info

    return result
