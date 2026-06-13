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
parallel with a current source** — its "companion model" — both made of parts
the DC solver already handles. We build that ordinary R/V/I circuit, call the
existing (ngspice-validated) `solve()`, read the new node voltages, update each
capacitor's remembered voltage, and step forward. No new linear algebra.

INITIAL CONDITIONS
------------------
At t = 0 each capacitor holds its initial voltage (default 0 V — an uncharged
cap, which behaves as a short circuit at the first instant). We find the starting
node voltages by solving the circuit once with each capacitor pinned to its
initial voltage (modelled as a voltage source), then begin stepping.

SCOPE (v1): capacitors (C) with constant DC sources — exactly the RC-charging
story. Sources are held at their DC value across the whole run (a step input is
just "uncharged cap + constant source", which is the demo). Inductors and
time-varying sources are natural follow-ups. Backward-Euler is rock-solid stable
(it never invents oscillations), which is why it's the default; trapezoidal
(more accurate, can ring) is an easy future swap via the `method` hook.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from solver.mna import solve
from solver.netlist import Netlist


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


# --- building the per-step circuits (the companion-model machinery) ----------

def _ic_netlist(netlist: Netlist, initial: dict[str, float]) -> Netlist:
    """The t=0 circuit: each capacitor pinned to its initial voltage (a source).

    An uncharged capacitor (0 V) becomes a 0 V source — i.e. a short — which is
    exactly how a fresh capacitor behaves the instant the circuit switches on.
    """
    nl = Netlist()
    for c in netlist.components:
        if c.kind == "C":
            nl.add("V", f"{c.name}__Vic", initial.get(c.name, 0.0), c.nodes[0], c.nodes[1])
        else:
            nl.add(c.kind, c.name, c.value, c.nodes[0], c.nodes[1])
    return nl


def _companion_netlist(netlist: Netlist, cap_v: dict[str, float], dt: float) -> Netlist:
    """One backward-Euler step's circuit: each capacitor → resistor ∥ current source.

    R_comp = dt/C (conductance C/dt); the parallel current source carries
    −(C/dt)·v_prev so the pair reproduces i = (C/dt)(v_new − v_prev). v_prev is
    the capacitor's voltage from the previous step (cap.nodes[0] − cap.nodes[1]).
    """
    nl = Netlist()
    for c in netlist.components:
        if c.kind == "C":
            r_comp = dt / c.value                       # resistance h/C
            i_comp = -(c.value / dt) * cap_v[c.name]     # current source value, sign per netlist.py
            nl.add("R", f"{c.name}__Rc", r_comp, c.nodes[0], c.nodes[1])
            nl.add("I", f"{c.name}__Ic", i_comp, c.nodes[0], c.nodes[1])
        else:
            nl.add(c.kind, c.name, c.value, c.nodes[0], c.nodes[1])
    return nl


# --- the time-stepping loop --------------------------------------------------

def solve_transient(
    netlist: Netlist,
    t_stop: float,
    dt: float,
    *,
    initial_conditions: dict[str, float] | None = None,
    method: str = "backward-euler",
) -> TransientResult:
    """March the circuit through time from 0 to `t_stop` in steps of `dt`.

    `initial_conditions` maps a capacitor name to its starting voltage (default
    0). Returns a TransientResult with the voltage curve for every node and
    capacitor. `method` is fixed to backward-Euler for now; the hook is here so
    trapezoidal can slot in later.
    """
    if method != "backward-euler":
        raise TransientError(f"only 'backward-euler' is implemented, not {method!r}")
    if dt <= 0 or t_stop <= 0 or dt > t_stop:
        raise TransientError("need 0 < dt <= t_stop")
    initial = dict(initial_conditions or {})
    caps = [c for c in netlist.components if c.kind == "C"]
    for name in initial:
        if name not in {c.name for c in caps}:
            raise TransientError(f"initial condition for unknown capacitor {name!r}")

    # t = 0 snapshot: solve with caps pinned to their initial voltages.
    ic = solve(_ic_netlist(netlist, initial))
    cap_v = {c.name: initial.get(c.name, 0.0) for c in caps}

    times = [0.0]
    node_voltages: dict[str, list[float]] = {n: [v] for n, v in ic.node_voltages.items()}
    capacitor_voltages: dict[str, list[float]] = {c.name: [cap_v[c.name]] for c in caps}

    # Step forward. round() guards against floating-point drift in the step count.
    n_steps = int(round(t_stop / dt))
    t = 0.0
    for _ in range(n_steps):
        t += dt
        step = solve(_companion_netlist(netlist, cap_v, dt))
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


# --- a plotting helper + RC demo ---------------------------------------------

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


def _rc_demo() -> int:
    """Charge a 1 µF... no — a clean RC: 5 V into 1 kΩ then 1 mF to ground (τ=1 s)."""
    n = Netlist()
    n.add("V", "V1", "5", "in", "0")
    n.add("R", "R1", "1k", "in", "mid")
    n.add("C", "C1", 1e-3, "mid", "0")                  # τ = R·C = 1000 · 1e-3 = 1 s
    result = solve_transient(n, t_stop=5.0, dt=0.01)

    tau_idx = min(range(len(result.times)), key=lambda i: abs(result.times[i] - 1.0))
    v_tau = result.series("mid")[tau_idx]
    print(result)
    print(f"\nRC time constant τ = 1.0 s. Theory says V(mid) reaches 63.2% of 5 V = "
          f"3.16 V at t=τ.\nSimulated V(mid) at t=1.0 s: {v_tau:.3f} V")
    out = "rc_charging.png"
    save_plot(result, ["mid"], out, title="RC charging (τ = 1 s)")
    print(f"Saved curve to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_rc_demo())
