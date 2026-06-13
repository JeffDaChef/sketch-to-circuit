"""Validate our from-scratch MNA solver against ngspice.

WHY THIS EXISTS
---------------
`solver/mna.py` is a circuit solver we wrote by hand. The most credible thing we
can say about it is *not* "trust me, the math is right" — it's "it agrees with
ngspice, the industry-standard simulator, to six decimal places on every circuit
I tried." This module is the machine that produces that sentence (and the table
behind it).

HOW IT WORKS
------------
For a given Netlist we:
  1. write a SPICE *deck* (a text description ngspice understands), asking it to
     do a DC operating-point analysis and print every node voltage and every
     voltage-source current;
  2. run `ngspice -b <deck>` as a subprocess and capture its text output;
  3. parse that output back into plain numbers;
  4. run our own solver on the same circuit and diff the two answers.

SCOPE: the linear DC subset — resistors (R), voltage sources (V), current
sources (I) — which is exactly what our solver supports. Capacitors are open at
DC; diodes need the (not-yet-built) nonlinear solver, so circuits containing them
are out of scope for this comparison.

SIGN CONVENTION: our solver already reports voltage-source current the same way
ngspice does (current *into* the + terminal is negative when the source is
delivering power — see test_mna.py). So currents are compared directly, no flip.

NOTE ON AVAILABILITY: ngspice needs a one-time admin install on this Mac, so it
may not be present. Everything except the actual subprocess call is unit-tested
without it; `ngspice_available()` lets callers (and the test suite) skip the live
run cleanly until ngspice is installed.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from solver.mna import SolveResult, solve
from solver.netlist import GROUND, Netlist

# Components ngspice can compare against our linear DC solver. A circuit using
# anything else (diodes especially) is outside what mna.py can solve, so we
# refuse it rather than silently produce a meaningless comparison.
_SUPPORTED_KINDS = {"R", "V", "I", "C"}  # C is allowed (open at DC) but ignored by both


class NgspiceError(Exception):
    """Raised when ngspice is missing, errors out, or returns unparseable output."""


# --- discovering ngspice -----------------------------------------------------

def ngspice_available() -> bool:
    """True if an `ngspice` executable is on PATH (so the live run can happen)."""
    return shutil.which("ngspice") is not None


# --- step 1: build the SPICE deck -------------------------------------------

def build_deck(netlist: Netlist, title: str = "sketch-to-circuit validation") -> str:
    """Write the SPICE text we hand to ngspice for a DC operating-point check.

    The deck is: a title line (SPICE always treats line 1 as the title), one line
    per component, then a `.control` block that runs `op` (the DC operating point)
    and prints each unknown on its own line so the output is trivial to parse.

    We print one quantity per `print` statement on purpose: asking ngspice to
    print several vectors at once produces a multi-column table that is fiddly to
    parse, whereas a lone vector prints as a clean ``name = value`` line.
    """
    for c in netlist.components:
        if c.kind not in _SUPPORTED_KINDS:
            raise NgspiceError(
                f"{c.name}: component kind {c.kind!r} is outside the linear DC "
                "subset this harness compares (no diode/nonlinear support yet)"
            )
    if not netlist.has_ground():
        raise NgspiceError("circuit has no ground (net '0'); ngspice needs a 0 node")

    lines = [f"* {title}"]
    for c in netlist.components:
        # Same one-line-per-part format as Netlist.to_spice(): name n+ n- value.
        lines.append(f"{c.name} {c.nodes[0]} {c.nodes[1]} {c.value:.10g}")

    lines.append(".control")
    lines.append("op")
    for node in netlist.node_names():          # non-ground nodes only; V(0) is 0 by definition
        lines.append(f"print v({node})")
    for c in netlist.components:
        if c.kind == "V":
            lines.append(f"print i({c.name})")
    lines.append(".endc")
    lines.append(".end")
    return "\n".join(lines) + "\n"


# --- step 2: run ngspice -----------------------------------------------------

def run_ngspice(deck: str, timeout: float = 30.0) -> str:
    """Run `ngspice -b` on a deck and return its raw stdout text.

    Raises NgspiceError if ngspice isn't installed or exits non-zero.
    """
    if not ngspice_available():
        raise NgspiceError(
            "ngspice not found on PATH — install it (one-time admin step) to run "
            "the live comparison. The deck-building and parsing logic is testable "
            "without it."
        )
    with tempfile.TemporaryDirectory() as tmp:
        deck_path = Path(tmp) / "circuit.cir"
        deck_path.write_text(deck)
        try:
            proc = subprocess.run(
                ["ngspice", "-b", str(deck_path)],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired as err:
            raise NgspiceError(f"ngspice timed out after {timeout}s") from err
    if proc.returncode != 0:
        raise NgspiceError(
            f"ngspice exited with code {proc.returncode}:\n{proc.stderr or proc.stdout}"
        )
    return proc.stdout


# --- step 3: parse ngspice's output ------------------------------------------

# A number in ngspice output: optional sign, digits/decimal, optional exponent.
_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
# A printed result line: "<lhs> = <number>", ignoring surrounding noise/units.
_RESULT_LINE = re.compile(rf"^\s*(\S+)\s*=\s*({_NUMBER})\b")


def parse_ngspice_output(text: str, netlist: Netlist) -> tuple[dict[str, float], dict[str, float]]:
    """Pull node voltages and source currents out of ngspice's text output.

    Returns ``(voltages, currents)`` keyed by *our* original node and source
    names. ngspice lowercases vector names, so we match case-insensitively and
    map back. Lines that aren't ``name = number`` (banners, timing, warnings) are
    ignored, which is what lets this survive ngspice's chatty output.
    """
    # Look-up tables from a normalized (lowercased) name back to our exact name.
    node_lookup = {n.lower(): n for n in netlist.node_names()}
    vsrc_lookup = {c.name.lower(): c.name for c in netlist.components if c.kind == "V"}

    voltages: dict[str, float] = {}
    currents: dict[str, float] = {}

    for line in text.splitlines():
        m = _RESULT_LINE.match(line)
        if not m:
            continue
        lhs, num = m.group(1).lower().strip(), float(m.group(2))

        v_match = re.fullmatch(r"v\((.+)\)", lhs)
        if v_match and v_match.group(1) in node_lookup:
            voltages[node_lookup[v_match.group(1)]] = num
            continue

        # Source current can appear as "i(v1)" or as the raw branch vector "v1#branch".
        i_match = re.fullmatch(r"i\((.+)\)", lhs)
        branch_match = re.fullmatch(r"(.+)#branch", lhs)
        src = (i_match.group(1) if i_match else
               branch_match.group(1) if branch_match else None)
        if src is not None and src in vsrc_lookup:
            currents[vsrc_lookup[src]] = num

    return voltages, currents


# --- step 4: compare ----------------------------------------------------------

@dataclass
class ComparisonReport:
    """The diff between our solver and ngspice for one circuit."""

    title: str
    agrees: bool
    voltage_errors: dict[str, float] = field(default_factory=dict)   # node -> |ours - ngspice|
    current_errors: dict[str, float] = field(default_factory=dict)   # source -> |ours - ngspice|
    max_voltage_error: float = 0.0
    max_current_error: float = 0.0
    ours: SolveResult | None = None
    ngspice_voltages: dict[str, float] = field(default_factory=dict)
    ngspice_currents: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        mark = "OK " if self.agrees else "XX "
        out = [f"[{mark}] {self.title}",
               f"      max ΔV = {self.max_voltage_error:.2e} V,  "
               f"max ΔI = {self.max_current_error:.2e} A"]
        if not self.agrees:
            for node, err in sorted(self.voltage_errors.items()):
                if err:
                    out.append(f"      V({node}): ours={self.ours.node_voltages[node]:+.6g} "
                               f"ngspice={self.ngspice_voltages.get(node, float('nan')):+.6g}")
        return "\n".join(out)


def compare(
    netlist: Netlist,
    title: str = "circuit",
    *,
    rel_tol: float = 1e-4,
    abs_tol_v: float = 1e-6,
    abs_tol_i: float = 1e-9,
    runner=run_ngspice,
) -> ComparisonReport:
    """Solve `netlist` both ways and report whether the answers agree.

    Tolerances are loose enough to absorb ngspice's ~7-significant-figure printed
    output, tight enough that a real solver bug would show. ``runner`` is the
    function that turns a deck into ngspice text; it's injectable so tests can
    feed canned output without ngspice installed.
    """
    ours = solve(netlist)
    deck = build_deck(netlist, title)
    ngspice_v, ngspice_i = parse_ngspice_output(runner(deck), netlist)

    voltage_errors: dict[str, float] = {}
    current_errors: dict[str, float] = {}
    agrees = True

    for node in netlist.node_names():
        if node not in ngspice_v:
            raise NgspiceError(f"ngspice output missing node voltage v({node})")
        err = abs(ours.node_voltages[node] - ngspice_v[node])
        voltage_errors[node] = err
        if not math.isclose(ours.node_voltages[node], ngspice_v[node],
                            rel_tol=rel_tol, abs_tol=abs_tol_v):
            agrees = False

    for name, ours_i in ours.source_currents.items():
        if name not in ngspice_i:
            raise NgspiceError(f"ngspice output missing source current i({name})")
        err = abs(ours_i - ngspice_i[name])
        current_errors[name] = err
        if not math.isclose(ours_i, ngspice_i[name], rel_tol=rel_tol, abs_tol=abs_tol_i):
            agrees = False

    return ComparisonReport(
        title=title,
        agrees=agrees,
        voltage_errors=voltage_errors,
        current_errors=current_errors,
        max_voltage_error=max(voltage_errors.values(), default=0.0),
        max_current_error=max(current_errors.values(), default=0.0),
        ours=ours,
        ngspice_voltages=ngspice_v,
        ngspice_currents=ngspice_i,
    )


def validate_suite(cases: list[tuple[str, Netlist]], runner=run_ngspice) -> list[ComparisonReport]:
    """Compare a whole list of ``(title, netlist)`` cases and return the reports."""
    return [compare(net, title, runner=runner) for title, net in cases]


# --- a small built-in suite + CLI --------------------------------------------

def default_suite() -> list[tuple[str, Netlist]]:
    """A handful of hand-checkable circuits to validate against ngspice.

    These mirror the circuits in tests/test_mna.py whose answers we worked out on
    paper, so a passing run means: paper == our solver == ngspice.
    """
    cases: list[tuple[str, Netlist]] = []

    n = Netlist()
    n.add("V", "V1", "10", "in", "0"); n.add("R", "R1", "1k", "in", "mid"); n.add("R", "R2", "1k", "mid", "0")
    cases.append(("voltage divider (equal)", n))

    n = Netlist()
    n.add("V", "V1", "10", "a", "0"); n.add("R", "R1", "1k", "a", "b"); n.add("R", "R2", "3k", "b", "0")
    cases.append(("voltage divider (1k/3k)", n))

    n = Netlist()
    n.add("V", "V1", "6", "a", "0"); n.add("R", "R1", "1k", "a", "0"); n.add("R", "R2", "1k", "a", "0")
    cases.append(("parallel resistors", n))

    n = Netlist()
    n.add("I", "I1", "1m", "0", "a"); n.add("R", "R1", "1k", "a", "0")
    cases.append(("current source + 1k", n))

    return cases


def main() -> int:
    """Run the default suite against ngspice and print a pass/fail table."""
    if not ngspice_available():
        print("ngspice is not installed — install it (one-time admin step) to run "
              "the live comparison.\nThe harness is ready; this command will work "
              "the moment ngspice is on PATH.")
        return 1
    reports = validate_suite(default_suite())
    print("ngspice vs. our MNA solver\n" + "=" * 40)
    for r in reports:
        print(r)
    passed = sum(r.agrees for r in reports)
    print("=" * 40)
    print(f"{passed}/{len(reports)} circuits agree with ngspice")
    return 0 if passed == len(reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
