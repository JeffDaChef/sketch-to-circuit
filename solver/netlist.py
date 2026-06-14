"""Circuit netlist data structures.

A *netlist* describes a circuit as a parts list with connectivity:
"R1 is a 10k-ohm resistor connected between net 'n1' and net 'n2'."
A *net* is one electrically-connected blob of wire; every terminal touching
that blob is at the same voltage. Following SPICE convention, the ground net
is always named "0".

This module is the shared language of the project: the vision pipeline
produces Netlist objects, the MNA solver consumes them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# Component kinds supported by the data model. The DC solver (mna.py) handles
# R/V/I; C is an open circuit at DC; D is approximated later (fixed LED drop).
KINDS = {
    "R": "resistor",
    "C": "capacitor",
    "L": "inductor",
    "V": "voltage source",
    "I": "current source",
    "D": "diode/LED",
}

GROUND = "0"


class NetlistError(Exception):
    """Raised for malformed netlists or unparseable values."""


# --- value parsing -----------------------------------------------------------

# Engineering-notation multipliers. Case matters only for m (milli) vs M (mega):
# SPICE itself uses "MEG" for mega, but handwritten "1M" on a resistor almost
# always means megaohm, so we accept both M and MEG as 1e6.
_MULTIPLIERS = {
    "T": 1e12,
    "G": 1e9,
    "MEG": 1e6,
    "M": 1e6,
    "k": 1e3,
    "K": 1e3,
    "m": 1e-3,
    "u": 1e-6,
    "µ": 1e-6,
    "n": 1e-9,
    "p": 1e-12,
}

_UNIT_SUFFIX = re.compile(r"(ohms?|Ω|[VvAaFf])$")
_PLAIN = re.compile(r"^(\d+\.?\d*|\.\d+)(MEG|meg|[TGkKmMuµnp])?$")
_INFIX = re.compile(r"^(\d+)(MEG|meg|[TGkKmMuµnp])(\d+)$")


def parse_value(text: str) -> float:
    """Turn a written component value like '10k', '4u7', or '5V' into a number.

    Handles plain numbers ('470'), engineering suffixes ('10k' -> 10000,
    '100n' -> 1e-7), trailing units which are ignored ('5V', '100nF', '2.2kΩ'),
    and European infix notation where the multiplier replaces the decimal
    point ('4u7' -> 4.7e-6, '2k2' -> 2200).
    """
    token = text.strip().replace(" ", "")
    if not token:
        raise NetlistError("empty value")

    # Strip a trailing unit (V, A, F, ohm...) -- it names the quantity, not the size.
    # Only strip a single letter if there's still a digit or multiplier before it,
    # so we don't wreck pure numbers.
    stripped = _UNIT_SUFFIX.sub("", token)
    if stripped and (stripped[-1].isdigit() or stripped[-1] in _MULTIPLIERS or
                     stripped[-3:].upper() == "MEG"):
        token = stripped

    # Plain Python numbers first: this covers integers, decimals, a leading sign,
    # and scientific notation ('1e6', '-2.2e-3') — exactly what to_spice() emits
    # with %g, so to_spice -> from_spice round-trips. (NaN/inf parse here too but
    # are rejected later by Component, which requires finite values.)
    try:
        return float(token)
    except ValueError:
        pass

    m = _INFIX.match(token)
    if m:
        whole, mult, frac = m.groups()
        return float(f"{whole}.{frac}") * _MULTIPLIERS[mult.upper() if len(mult) > 1 else mult]

    m = _PLAIN.match(token)
    if m:
        number, mult = m.groups()
        scale = 1.0
        if mult:
            scale = _MULTIPLIERS[mult.upper() if len(mult) > 1 else mult]
        return float(number) * scale

    raise NetlistError(f"can't parse component value: {text!r}")


# --- the data structures -----------------------------------------------------


@dataclass
class Component:
    """One circuit element: its kind, name, value, and the two nets it touches.

    Node order matters for sources: nodes[0] is the + terminal of a voltage
    source; current from a current source flows out of nodes[0] through the
    source into nodes[1] (i.e. it pushes current INTO the circuit at nodes[1]).
    """

    kind: str       # one of KINDS: 'R', 'C', 'V', 'I', 'D'
    name: str       # e.g. 'R1' -- unique within a netlist
    value: float    # ohms, farads, volts, or amps depending on kind
    nodes: tuple[str, str]

    def __post_init__(self):
        if self.kind not in KINDS:
            raise NetlistError(f"unknown component kind {self.kind!r} for {self.name}")
        if len(self.nodes) != 2:
            raise NetlistError(f"{self.name}: components have exactly 2 nodes")
        # A value must be a real, finite number (NaN/inf would poison the matrix).
        if not math.isfinite(self.value):
            raise NetlistError(f"{self.name}: value must be finite, got {self.value!r}")
        # Passive parts (R/C/L) must be strictly positive: a zero resistance is a
        # short (model it as a merged node) and would divide-by-zero in the solver;
        # negatives are unphysical. Sources (V/I) and the diode placeholder may be
        # any finite value (a -5 V rail is legitimate).
        if self.kind in ("R", "C", "L") and self.value <= 0:
            raise NetlistError(
                f"{self.name}: {KINDS[self.kind]} value must be positive, got {self.value!r} "
                "(a zero/negative R/C/L isn't physical; a 0 Ω 'resistor' is just a wire)"
            )


@dataclass
class Netlist:
    """A whole circuit: a list of components. Ground is the net named '0'."""

    components: list[Component] = field(default_factory=list)

    def add(self, kind: str, name: str, value: float | str, node_a: str, node_b: str) -> Component:
        """Add a component; value may be a number or a string like '10k'."""
        if any(c.name == name for c in self.components):
            raise NetlistError(f"duplicate component name {name!r}")
        if isinstance(value, str):
            value = parse_value(value)
        comp = Component(kind, name, float(value), (str(node_a), str(node_b)))
        self.components.append(comp)
        return comp

    def node_names(self) -> list[str]:
        """All net names in the circuit except ground, sorted."""
        names = {n for c in self.components for n in c.nodes}
        names.discard(GROUND)
        return sorted(names)

    def has_ground(self) -> bool:
        return any(GROUND in c.nodes for c in self.components)

    # --- SPICE text format ---

    def to_spice(self, title: str = "sketch-to-circuit netlist") -> str:
        """Emit standard SPICE netlist text (title line, components, .end)."""
        lines = [f"* {title}"]
        for c in self.components:
            lines.append(f"{c.name} {c.nodes[0]} {c.nodes[1]} {c.value:.6g}")
        lines.append(".end")
        return "\n".join(lines) + "\n"

    @classmethod
    def from_spice(cls, text: str) -> "Netlist":
        """Parse SPICE-style text produced by to_spice (or written by hand)."""
        netlist = cls()
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("*") or line.lower() == ".end":
                continue
            parts = line.split()
            if len(parts) != 4:
                raise NetlistError(f"can't parse netlist line: {raw!r}")
            name, node_a, node_b, value = parts
            kind = name[0].upper()
            netlist.add(kind, name, parse_value(value), node_a, node_b)
        return netlist
