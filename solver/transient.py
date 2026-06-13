"""Transient (time-domain) circuit simulation — watch a capacitor charge.

WHAT THIS ADDS
--------------
`solver/mna.py` answers "where does this circuit settle?" (one DC snapshot).
This module answers "how does it get there?" — the voltage at every node as a
function of *time*. The headline demo: connect a battery to a resistor and a
capacitor and watch the capacitor's voltage curve upward as it charges (the
classic RC exponential), instead of just reporting its final value.

THE TRICK: COMPANION MODELS (so we reuse the DC solver, not rewrite it)
----------------------------------------------------------------------
A capacitor's defining law is i = C·dv/dt — it involves a *rate of change*, which
the DC solver can't express. The standard fix is **numerical integration**: chop
time into small steps of size h, and approximate the derivative across one step.

Using **backward-Euler**, dv/dt at the new time ≈ (v_new − v_old) / h, so

    i_new = C·(v_new − v_old)/h = (C/h)·v_new − (C/h)·v_old.

Read that as a circuit: the first term is a plain resistor of conductance C/h
(i.e. resistance h/C); the second is a constant current source set by *last
step's* voltage. So at each time step a capacitor becomes **a resistor in
parallel with a current source** — its "companion model", made of parts the DC
solver already handles.

TWO EXTRAS THAT COMPOSE CLEANLY
-------------------------------
* Non-linear parts (diodes): once a capacitor is a resistor+source for the step,
  the step is just an R/V/I/D circuit. If any diodes are present we solve each
  step with `solve_nonlinear` (Newton-Raphson) instead of the plain linear solve
  — the time loop becomes SPICE's outer loop, Newton-Raphson the inner loop. This
  is what makes a *rectifier with a smoothing capacitor* simulable.
* Time-varying sources: pass `sources={"V1": lambda t: ...}` and the named V/I
  source takes that value at each instant (a `sine()` helper is provided). Without
  this every source is held at its constant DC value.

INITIAL CONDITIONS
------------------
At t = 0 each capacitor holds its initial voltage (default 0 V — an uncharged
cap, which behaves as a short circuit at the first instant). We find the starting
node voltages by solving the circuit once with each capacitor pinned to its
initial voltage (modelled as a voltage source), then begin stepping.

SCOPE (v1): capacitors (C), optionally with diodes and/or time-varying sources.
Inductors are a natural follow-up. Backward-Euler is rock-solid stable (it never
invents oscillations), which is why it's the default; trapezoidal (more accurate,
can ring) is an easy future swap via the `method` hook.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from solver.mna import solve
from solver.netlist import Netlist
from solver.nonlinear import DiodeModel, solve_nonlinear


class TransientError(Exception):
    """Raised for invalid transient runs (bad step size, unsupported parts...)."""


@dataclass
class TransientResult:
    """The time-domain answer: a time axis plus a voltage curve per node/cap."""

    times: list[float]                                  # seconds, length = #samples
    node_voltages: dict[str, list[float]]               # node name -> voltage at each time
    capacitor_voltages: dict[str, list[float]]          # cap name -> voltage across it

    def series(self, node: str) -> list[float]:
        """The full voltage-vs-time series for one node."""
        return self.node_voltages[node]

    def final(self) -> dict[str, float]:
        """The last (settled) voltage at every node — should match a DC solve."""
        return {n: v[-1] for n, v in self.node_voltages.items()}

    def __str__(self) -> str:
        span = f"{self.times[0]:g}..{self.times[-1]:g} s, {len(self.times)} samples"
        lines = [f"Transient: {span}", "Final node voltages:"]
        for n in sorted(self.final()):
            lines.append(f"  V({n}) = {self.final()[n]:+.4f} V")
        return "\n".join(lines)


# --- time-varying source helpers ---------------------------------------------

def sine(amplitude: float, freq_hz: float, offset: float = 0.0, phase: float = 0.0) -> Callable[[float], float]:
    """A sine waveform v(t) = offset + amplitude·sin(2π·freq·t + phase), for `sources`."""
    w = 2.0 * math.pi * freq_hz
    return lambda t: offset + amplitude * math.sin(w * t + phase)


# --- building the per-step circuits (the companion-model machinery) ----------

def _with_sources(nl: Netlist, c, overrides: dict[str, float]) -> None:
    """Add component `c` to `nl`, swapping in a time-varying value if overridden."""
    value = overrides[c.name] if (c.kind in ("V", "I") and c.name in overrides) else c.value
    nl.add(c.kind, c.name, value, c.nodes[0], c.nodes[1])


def _ic_netlist(netlist: Netlist, initial: dict[str, float], overrides: dict[str, float]) -> Netlist:
    """The t=0 circuit: each capacitor pinned to its initial voltage (a source).

    An uncharged capacitor (0 V) becomes a 0 V source — i.e. a short — which is
    exactly how a fresh capacitor behaves the instant the circuit switches on.
    """
    nl = Netlist()
    for c in netlist.components:
        if c.kind == "C":
            nl.add("V", f"{c.name}__Vic", initial.get(c.name, 0.0), c.nodes[0], c.nodes[1])
        else:
            _with_sources(nl, c, overrides)
    return nl


def _companion_netlist(netlist: Netlist, cap_v: dict[str, float], dt: float,
                       overrides: dict[str, float]) -> Netlist:
    """One backward-Euler step's circuit: each capacitor → resistor ∥ current source.

    R_comp = dt/C (conductance C/dt); the parallel current source carries
    −(C/dt)·v_prev so the pair reproduces i = (C/dt)(v_new − v_prev). v_prev is
    the capacitor's voltage from the previous step (cap.nodes[0] − cap.nodes[1]).
    Non-capacitor parts (including diodes) pass through for the step's solver.
    """
    nl = Netlist()
    for c in netlist.components:
        if c.kind == "C":
            r_comp = dt / c.value                       # resistance h/C
            i_comp = -(c.value / dt) * cap_v[c.name]     # current source value, sign per netlist.py
            nl.add("R", f"{c.name}__Rc", r_comp, c.nodes[0], c.nodes[1])
            nl.add("I", f"{c.name}__Ic", i_comp, c.nodes[0], c.nodes[1])
        else:
            _with_sources(nl, c, overrides)
    return nl


# --- the time-stepping loop --------------------------------------------------

def solve_transient(
    netlist: Netlist,
    t_stop: float,
    dt: float,
    *,
    initial_conditions: dict[str, float] | None = None,
    sources: dict[str, Callable[[float], float]] | None = None,
    models: dict[str, DiodeModel] | None = None,
    method: str = "backward-euler",
) -> TransientResult:
    """March the circuit through time from 0 to `t_stop` in steps of `dt`.

    `initial_conditions` maps a capacitor name to its starting voltage (default
    0). `sources` maps a V/I source name to a function of time (e.g. `sine(...)`);
    unlisted sources stay at their DC value. `models` maps a diode name to its
    DiodeModel (only relevant when the circuit contains diodes, which trigger a
    Newton-Raphson solve at each step). Returns a TransientResult with the voltage
    curve for every node and capacitor. `method` is fixed to backward-Euler for
    now; the hook is here so trapezoidal can slot in later.
    """
    if method != "backward-euler":
        raise TransientError(f"only 'backward-euler' is implemented, not {method!r}")
    if dt <= 0 or t_stop <= 0 or dt > t_stop:
        raise TransientError("need 0 < dt <= t_stop")
    initial = dict(initial_conditions or {})
    sources = dict(sources or {})
    caps = [c for c in netlist.components if c.kind == "C"]
    cap_names = {c.name for c in caps}
    for name in initial:
        if name not in cap_names:
            raise TransientError(f"initial condition for unknown capacitor {name!r}")
    source_names = {c.name for c in netlist.components if c.kind in ("V", "I")}
    for name in sources:
        if name not in source_names:
            raise TransientError(f"time-varying value for unknown source {name!r}")

    # Diodes present -> each step is a Newton-Raphson solve; else a plain linear one.
    has_diodes = any(c.kind == "D" for c in netlist.components)
    def step_solve(nl: Netlist):
        return solve_nonlinear(nl, models=models) if has_diodes else solve(nl)

    def overrides(t: float) -> dict[str, float]:
        return {name: fn(t) for name, fn in sources.items()}

    # t = 0 snapshot: solve with caps pinned to their initial voltages.
    ic = step_solve(_ic_netlist(netlist, initial, overrides(0.0)))
    cap_v = {c.name: initial.get(c.name, 0.0) for c in caps}

    times = [0.0]
    node_voltages: dict[str, list[float]] = {n: [v] for n, v in ic.node_voltages.items()}
    capacitor_voltages: dict[str, list[float]] = {c.name: [cap_v[c.name]] for c in caps}

    # Step forward. round() guards against floating-point drift in the step count.
    n_steps = int(round(t_stop / dt))
    t = 0.0
    for _ in range(n_steps):
        t += dt
        step = step_solve(_companion_netlist(netlist, cap_v, dt, overrides(t)))
        # Update each capacitor's remembered voltage from the new node voltages,
        # then record the whole snapshot.
        for c in caps:
            cap_v[c.name] = step.node_voltages[c.nodes[0]] - step.node_voltages[c.nodes[1]]
        times.append(t)
        for n in node_voltages:
            node_voltages[n].append(step.node_voltages[n])
        for c in caps:
            capacitor_voltages[c.name].append(cap_v[c.name])

    return TransientResult(times, node_voltages, capacitor_voltages)


# --- a plotting helper + demos -----------------------------------------------

def save_plot(result: TransientResult, nodes: list[str], path: str, title: str = "Transient response") -> None:
    """Save a voltage-vs-time PNG for the given nodes (the demo artifact)."""
    import matplotlib
    matplotlib.use("Agg")                               # headless: no display needed
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for n in nodes:
        ax.plot(result.times, result.series(n), label=f"V({n})")
    ax.set_xlabel("time (s)"); ax.set_ylabel("voltage (V)")
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _rc_demo() -> None:
    """Charge a clean RC: 5 V into 1 kΩ then 1 mF to ground (τ = 1 s)."""
    n = Netlist()
    n.add("V", "V1", "5", "in", "0")
    n.add("R", "R1", "1k", "in", "mid")
    n.add("C", "C1", 1e-3, "mid", "0")                  # τ = R·C = 1000 · 1e-3 = 1 s
    result = solve_transient(n, t_stop=5.0, dt=0.01)
    tau_idx = min(range(len(result.times)), key=lambda i: abs(result.times[i] - 1.0))
    print("RC charging (τ = 1 s):")
    print(f"  V(mid) at t=τ: {result.series('mid')[tau_idx]:.3f} V  (theory 3.16 V)")
    save_plot(result, ["mid"], "rc_charging.png", title="RC charging (τ = 1 s)")
    print("  saved rc_charging.png")


def _rectifier_demo() -> None:
    """Half-wave rectifier with a smoothing capacitor — diodes + caps + time.

    A 5 V, 60 Hz sine drives a diode into a capacitor and load resistor. The cap
    charges near each positive peak and slowly discharges through the load between
    peaks, leaving a (mostly) DC output with a little ripple — the canonical
    'transient + non-linear diode' circuit, impossible before this lever.
    """
    n = Netlist()
    n.add("V", "V1", 0.0, "ac", "0")                    # value overridden by the sine below
    n.add("D", "D1", 0.0, "ac", "out")                  # anode 'ac', cathode 'out'
    n.add("C", "C1", 100e-6, "out", "0")                # smoothing cap, RC = 0.1 s >> 1/60 s
    n.add("R", "R1", "1k", "out", "0")                  # load
    period = 1.0 / 60.0
    result = solve_transient(
        n, t_stop=5 * period, dt=period / 200,
        sources={"V1": sine(amplitude=5.0, freq_hz=60.0)},
    )
    out = result.series("out")
    settled = out[len(out) // 2:]                       # ignore initial charge-up transient
    ripple = max(settled) - min(settled)
    print("\nHalf-wave rectifier (5 V, 60 Hz, 100 µF, 1 kΩ):")
    print(f"  peak output : {max(out):.3f} V  (≈ 5 V − one diode drop)")
    print(f"  ripple (settled): {ripple*1e3:.1f} mV")
    save_plot(result, ["ac", "out"], "rectifier.png", title="Half-wave rectifier + smoothing cap")
    print("  saved rectifier.png")


def _demo() -> int:
    _rc_demo()
    _rectifier_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
