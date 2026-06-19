"""Export a Netlist to a portable SPICE deck real tools can open and run.

WHY THIS EXISTS
---------------
`Netlist.to_spice()` writes a deliberately minimal format for our *own* round-trip
(four fields per line, no models, no analysis). This module writes the richer
thing an outside engineer expects: a complete `.cir` file that **ngspice, LTspice,
or KiCad** can load and simulate without edits. That bridges the toy to the real
engineering ecosystem — an extracted hand-drawn circuit can leave this project and
open in the same software a professional uses.

What it adds over the minimal format:
  * **Diodes get a real `.model`.** SPICE diodes reference a named model
    (`D1 anode cathode DMODEL1`) with a separate `.model DMODEL1 D(Is=... N=...)`
    line. Identical models are de-duplicated and shared.
  * **Values in engineering notation** (`10000` → `10k`, `1e-4` → `100u`) so the
    file reads the way a schematic would. Note SPICE uses `MEG` for mega and `m`
    for milli — we honour that (never bare `M`).
  * **An optional analysis directive** (`.op`, `.tran 10u 20m`, `.dc V1 0 5 0.1`)
    so the deck is immediately runnable, not just a parts list.

R / V / I / C lines stay in the same `name n+ n- value` shape as `to_spice`, so the
non-diode part of the output is still valid for our own parser too.
"""

from __future__ import annotations

from solver.netlist import GROUND, Netlist
from solver.nonlinear import SILICON, DiodeModel

_ENG_SUFFIXES = [
    (1e12, "T"), (1e9, "G"), (1e6, "MEG"), (1e3, "k"),
    (1.0, ""), (1e-3, "m"), (1e-6, "u"), (1e-9, "n"), (1e-12, "p"),
]


def format_value(value: float) -> str:
    """Render a component value in engineering notation (e.g. 10000 -> '10k').

    Chosen so that parse_value(format_value(x)) == x. Values outside the suffix
    range fall back to plain %g scientific notation.
    """
    if value == 0:
        return "0"
    av = abs(value)
    for scale, suffix in _ENG_SUFFIXES:
        if av >= scale:
            return f"{value / scale:g}{suffix}"
    return f"{value:g}"


def _component_line(comp, diode_model_name: dict[str, str]) -> str:
    """One SPICE element line for a component."""
    a, b = comp.nodes
    if comp.kind == "D":
        return f"{comp.name} {a} {b} {diode_model_name[comp.name]}"
    return f"{comp.name} {a} {b} {format_value(comp.value)}"


def export_spice(
    netlist: Netlist,
    *,
    title: str = "sketch-to-circuit export",
    analysis: str | None = None,
    models: dict[str, DiodeModel] | None = None,
    default_model: DiodeModel = SILICON,
) -> str:
    """Build a complete, runnable SPICE deck for `netlist`.

    `analysis` is an optional raw SPICE directive that makes the deck self-running,
    e.g. ".op" (operating point), ".tran 10u 20m" (transient), or
    ".dc V1 0 5 0.1" (DC sweep). `models` maps a diode name to its DiodeModel;
    diodes not listed use `default_model`. Returns the deck as text.
    """
    models = dict(models or {})

    diode_model_name: dict[str, str] = {}
    model_decls: dict[str, DiodeModel] = {}
    seen: dict[tuple[float, float], str] = {}
    for comp in netlist.components:
        if comp.kind != "D":
            continue
        model = models.get(comp.name, default_model)
        key = (model.I_s, model.n)
        if key not in seen:
            name = f"DMODEL{len(seen) + 1}"
            seen[key] = name
            model_decls[name] = model
        diode_model_name[comp.name] = seen[key]

    lines = [f"* {title}"]
    lines += [_component_line(c, diode_model_name) for c in netlist.components]
    for name, model in model_decls.items():
        lines.append(f".model {name} D(Is={model.I_s:g} N={model.n:g})")
    if analysis:
        lines.append(analysis if analysis.startswith(".") else f".{analysis}")
    lines.append(".end")

    if not netlist.has_ground():
        lines.insert(1, "* WARNING: no ground (node '0') — this deck will not simulate as-is")
    return "\n".join(lines) + "\n"


def write_spice_file(netlist: Netlist, path: str, **kwargs) -> None:
    """Write `export_spice(...)` to a .cir file ready to open in ngspice/KiCad."""
    from pathlib import Path
    Path(path).write_text(export_spice(netlist, **kwargs))


def _demo() -> int:
    """Print an exported deck for an LED circuit so the format is visible."""
    from solver.nonlinear import LED
    n = Netlist()
    n.add("V", "V1", "5", "in", "0")
    n.add("R", "R1", "220", "in", "a")
    n.add("D", "LED1", 0.0, "a", "0")
    print(export_spice(n, title="LED + resistor", analysis=".op", models={"LED1": LED}))
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
