"""Non-linear DC solving — real diodes via Newton-Raphson.

WHAT THIS ADDS
--------------
The linear solver (`mna.py`) refuses diodes, and the original project plan was to
fake a diode as a fixed 2 V drop. This module does the real thing: it solves
circuits containing diodes/LEDs using the actual **Shockley diode equation**

    I = I_s · (exp(V / (n·V_t)) − 1)

where V is the voltage across the diode, I_s the saturation current, n the
ideality factor, and V_t ≈ 0.0259 V the thermal voltage at room temperature.
That exponential is what makes a diode non-linear — there's no single resistance
that describes it — so a one-shot linear solve can't find the answer.

THE METHOD: NEWTON-RAPHSON (same companion-model trick as transient.py)
----------------------------------------------------------------------
Newton-Raphson finds where a curve crosses zero by repeatedly drawing the
tangent line and jumping to where the tangent hits zero. Applied to a circuit:
at a guessed diode voltage V0 we replace the diode by its **tangent** — a
straight line with slope g = dI/dV at V0. A straight line through current/voltage
is exactly a resistor (slope g) in parallel with a current source (the offset) —
the diode's "companion model", just like a capacitor's in transient.py. So each
iteration becomes an ordinary R/V/I circuit we hand to the existing, ngspice-
validated `solve()`. We read the new diode voltages, rebuild the tangents, and
repeat until the voltages stop moving. No new linear algebra.

    g    = (I_s / (n·V_t)) · exp(V0 / (n·V_t))     # slope of the I-V curve at V0
    I0   = I_s · (exp(V0 / (n·V_t)) − 1)           # diode current at V0
    I_eq = I0 − g·V0                               # the parallel current source

VOLTAGE LIMITING: the exponential is brutal — a guess that overshoots by half a
volt changes the current by a factor of ~e^20. Left alone, Newton-Raphson can
diverge or overflow. Real SPICE uses a scheme called `pnjlim`; we use the same
*idea* in a simpler, explainable form: never let a diode's voltage jump up by
more than a small step per iteration. Near the answer the natural step is tiny,
so this only bites during the wild early iterations.

SCOPE: DC operating point with diodes/LEDs (forward and reverse). No reverse
breakdown / Zener modelling. Diode `value` in the netlist is currently unused —
the model comes from `DiodeModel` (default silicon; an LED preset is provided).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from solver.mna import SolveResult, solve
from solver.netlist import Netlist


class NonlinearError(Exception):
    """Raised when Newton-Raphson fails to converge within the iteration budget."""


@dataclass(frozen=True)
class DiodeModel:
    """Shockley diode parameters. Defaults model a small silicon diode (~0.65 V)."""

    I_s: float = 1e-14        # saturation current (A)
    n: float = 1.0            # ideality factor
    V_t: float = 0.025852     # thermal voltage at ~300 K (V)

    @property
    def vte(self) -> float:
        """n·V_t — the scale that sets how sharply the exponential turns on."""
        return self.n * self.V_t

    def current(self, v: float) -> float:
        """Diode current at voltage v across it (anode − cathode)."""
        return self.I_s * (math.exp(min(v / self.vte, 80.0)) - 1.0)

    def conductance(self, v: float) -> float:
        """Slope dI/dV of the I-V curve at v — the companion-model resistance's 1/R."""
        return (self.I_s / self.vte) * math.exp(min(v / self.vte, 80.0))


# A light-emitting diode turns on much higher (~1.8 V): smaller I_s, larger n.
LED = DiodeModel(I_s=1e-17, n=2.0)
SILICON = DiodeModel()


def _linearized_netlist(netlist: Netlist, vd: dict[str, float],
                        models: dict[str, DiodeModel]) -> Netlist:
    """Build one Newton iteration's R/V/I circuit: each diode → resistor ∥ source.

    `vd[name]` is the current guess for that diode's voltage; the diode is
    replaced by its tangent there (conductance g and offset current I_eq).
    """
    nl = Netlist()
    for c in netlist.components:
        if c.kind == "D":
            model = models[c.name]
            v0 = vd[c.name]
            g = model.conductance(v0)
            i_eq = model.current(v0) - g * v0          # offset so the line passes through (v0, I0)
            nl.add("R", f"{c.name}__Rd", 1.0 / g, c.nodes[0], c.nodes[1])
            nl.add("I", f"{c.name}__Id", i_eq, c.nodes[0], c.nodes[1])
        else:
            nl.add(c.kind, c.name, c.value, c.nodes[0], c.nodes[1])
    return nl


def solve_nonlinear(
    netlist: Netlist,
    *,
    models: dict[str, DiodeModel] | None = None,
    default_model: DiodeModel = SILICON,
    max_iter: int = 200,
    tol: float = 1e-9,
    max_step: float = 0.1,
) -> SolveResult:
    """DC solve for a circuit that may contain diodes, via Newton-Raphson.

    `models` optionally maps a diode name to its DiodeModel; any diode not listed
    uses `default_model`. Returns the same SolveResult shape as the linear solver,
    with each diode's final current included in `branch_currents`.
    """
    diodes = [c for c in netlist.components if c.kind == "D"]
    if not diodes:
        return solve(netlist)                          # nothing non-linear: one linear solve

    models = dict(models or {})
    for d in diodes:
        models.setdefault(d.name, default_model)

    # Start every diode at a modest forward guess; limiting handles the rest.
    vd = {d.name: 0.6 for d in diodes}

    last: SolveResult | None = None
    for _ in range(max_iter):
        last = solve(_linearized_netlist(netlist, vd, models))
        new_vd = {d.name: last.node_voltages[d.nodes[0]] - last.node_voltages[d.nodes[1]]
                  for d in diodes}

        # Voltage limiting: cap how far any diode voltage may *rise* per step, the
        # only direction the exponential can run away in.
        moved = 0.0
        for name in vd:
            step = new_vd[name] - vd[name]
            if step > max_step:
                new_vd[name] = vd[name] + max_step
            moved = max(moved, abs(new_vd[name] - vd[name]))
            vd[name] = new_vd[name]

        if moved < tol:
            break
    else:
        raise NonlinearError(
            f"Newton-Raphson did not converge in {max_iter} iterations "
            f"(last voltage move {moved:.2e} V > tol {tol:g})"
        )

    # Fold the final diode currents into the result's branch currents.
    for d in diodes:
        last.branch_currents[d.name] = models[d.name].current(vd[d.name])
    return last


def _demo() -> int:
    """Solve a 5 V / 1 kΩ / diode circuit and an LED circuit; print the operating points."""
    n = Netlist()
    n.add("V", "V1", "5", "in", "0")
    n.add("R", "R1", "1k", "in", "mid")
    n.add("D", "D1", 0.0, "mid", "0")                  # anode at 'mid', cathode at ground
    r = solve_nonlinear(n)
    vd = r.voltage("mid") - r.voltage("0")
    print("Silicon diode + 1k from 5V:")
    print(f"  V across diode = {vd:.3f} V   (rule of thumb ~0.7 V)")
    print(f"  current        = {r.branch_currents['D1']*1e3:.3f} mA")

    n2 = Netlist()
    n2.add("V", "V1", "5", "in", "0")
    n2.add("R", "R1", "220", "in", "mid")
    n2.add("D", "LED1", 0.0, "mid", "0")
    r2 = solve_nonlinear(n2, models={"LED1": LED})
    print("\nLED + 220Ω from 5V:")
    print(f"  V across LED = {r2.voltage('mid'):.3f} V   (LEDs glow ~1.8-2 V)")
    print(f"  current      = {r2.branch_currents['LED1']*1e3:.3f} mA")
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
