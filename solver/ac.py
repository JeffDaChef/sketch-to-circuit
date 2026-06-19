"""AC (frequency-domain) analysis — how a circuit responds to each frequency.

WHAT THIS ADDS
--------------
The DC solver answers "where does it settle?", the transient solver "how does it
get there over time?". This answers the third classic question: **"how does the
circuit treat each frequency?"** — which is what makes a filter a filter. Drive a
circuit with a sine of frequency f and, once settled, every voltage is also a sine
of frequency f with some amplitude and phase shift. AC analysis computes that
amplitude-and-phase response across a sweep of frequencies — the basis of Bode
plots, filter cutoffs, and resonance.

THE IDEA: PHASORS turn calculus into algebra. A sinusoid is represented by one
complex number (a *phasor*) carrying its amplitude and phase. Under that
representation a capacitor and inductor stop being calculus (i = C·dv/dt) and
become plain complex "resistors" called impedances:

    resistor   Z = R                  (admittance Y = 1/R)
    capacitor  Z = 1/(jωC)            (Y = jωC)      — blocks DC, passes high f
    inductor   Z = jωL                (Y = 1/(jωL))  — passes DC, blocks high f

with ω = 2πf. So AC analysis is just the SAME Modified Nodal Analysis as the DC
solver, run with COMPLEX admittances — np.linalg.solve handles complex matrices
directly. (At ω→0 the cap's admittance →0, an open, and the inductor's →∞, a short,
exactly matching how the DC solver treats them. We therefore sweep f>0.)

SCOPE: linear AC — R, C, L, and sources. Diodes are non-linear; true AC analysis
of a diode circuit means linearising around a DC operating point (small-signal),
which is a later step, so we refuse diodes here with a clear error.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np

from solver.netlist import GROUND, Netlist
from solver.nonlinear import SILICON, DiodeModel, solve_nonlinear


class ACError(Exception):
    """Raised for invalid AC runs (non-positive frequency, unsupported parts...)."""


@dataclass
class ACResult:
    """The phasor solution at ONE frequency: complex voltage at every node."""

    freq: float
    node_voltages: dict[str, complex]
    source_currents: dict[str, complex]

    def gain(self, node: str) -> complex:
        """Complex phasor at `node` (its amplitude is |.|, phase is its angle)."""
        return self.node_voltages[node]


def solve_ac(netlist: Netlist, freq: float) -> ACResult:
    """Solve the circuit at a single frequency `freq` (Hz) using complex MNA.

    Source ``value``s are treated as AC phasor amplitudes (real, phase 0). Returns
    complex node voltages and voltage-source currents.
    """
    if freq <= 0:
        raise ACError("AC frequency must be > 0 (a capacitor is an open / inductor a "
                      "short at DC — use the DC solver for f=0)")
    for c in netlist.components:
        if c.kind == "D":
            raise ACError(f"{c.name}: diodes are non-linear; AC needs a small-signal "
                          "linearisation around a DC operating point (not built yet)")
    if not netlist.has_ground():
        raise ACError("circuit has no ground (net '0') to reference voltages against")

    w = 2.0 * math.pi * freq
    nodes = netlist.node_names()
    node_index = {name: i for i, name in enumerate(nodes)}
    n = len(nodes)
    vsources = [c for c in netlist.components if c.kind == "V"]
    vsource_index = {c.name: n + k for k, c in enumerate(vsources)}
    size = n + len(vsources)

    A = np.zeros((size, size), dtype=complex)
    z = np.zeros(size, dtype=complex)

    for c in netlist.components:
        if c.kind == "R":
            y = 1.0 / c.value
        elif c.kind == "C":
            y = 1j * w * c.value
        elif c.kind == "L":
            y = 1.0 / (1j * w * c.value)
        elif c.kind == "I":
            a, b = c.nodes
            if a != GROUND:
                z[node_index[a]] -= c.value
            if b != GROUND:
                z[node_index[b]] += c.value
            continue
        else:
            continue
        a, b = c.nodes
        if a != GROUND:
            A[node_index[a], node_index[a]] += y
        if b != GROUND:
            A[node_index[b], node_index[b]] += y
        if a != GROUND and b != GROUND:
            A[node_index[a], node_index[b]] -= y
            A[node_index[b], node_index[a]] -= y

    for c in vsources:
        s = vsource_index[c.name]
        p, q = c.nodes
        if p != GROUND:
            A[node_index[p], s] += 1
            A[s, node_index[p]] += 1
        if q != GROUND:
            A[node_index[q], s] -= 1
            A[s, node_index[q]] -= 1
        z[s] = c.value

    try:
        x = np.linalg.solve(A, z)
    except np.linalg.LinAlgError as err:
        raise ACError("circuit is unsolvable at this frequency (singular matrix) — "
                      "a floating node or a source/reactance loop") from err

    node_voltages: dict[str, complex] = {GROUND: 0j}
    for name, i in node_index.items():
        node_voltages[name] = complex(x[i])
    source_currents = {c.name: complex(x[vsource_index[c.name]]) for c in vsources}
    return ACResult(freq, node_voltages, source_currents)



def logspace(f_start: float, f_stop: float, points_per_decade: int = 20) -> list[float]:
    """Frequencies spread evenly on a log axis (the natural axis for Bode plots)."""
    if f_start <= 0 or f_stop <= f_start:
        raise ACError("need 0 < f_start < f_stop")
    decades = math.log10(f_stop / f_start)
    n = max(2, int(round(decades * points_per_decade)) + 1)
    return [f_start * 10 ** (decades * i / (n - 1)) for i in range(n)]


def transfer_function(netlist: Netlist, freqs, output_node: str, input_source: str) -> dict:
    """Sweep `freqs` and return the transfer function H(f) = V(output)/V(input).

    `input_source` is the name of the driving voltage source; H is normalised by
    its phasor so the result is dimensionless gain. Returns a dict with ``freq``,
    complex ``H``, ``mag``, ``mag_db`` (20·log10|H|) and ``phase_deg``.
    """
    src = next((c for c in netlist.components if c.name == input_source), None)
    if src is None or src.kind != "V":
        raise ACError(f"input_source {input_source!r} is not a voltage source in the netlist")
    v_in = complex(src.value)
    if v_in == 0:
        raise ACError(f"input source {input_source!r} has zero amplitude")

    freq_list, H = list(freqs), []
    for f in freq_list:
        H.append(solve_ac(netlist, f).node_voltages[output_node] / v_in)
    mag = [abs(h) for h in H]
    return {
        "freq": freq_list,
        "H": H,
        "mag": mag,
        "mag_db": [20.0 * math.log10(m) if m > 0 else -np.inf for m in mag],
        "phase_deg": [math.degrees(cmath.phase(h)) for h in H],
    }



def _diode_resistances(netlist: Netlist, models, default_model: DiodeModel):
    """Solve the DC operating point and return {diode_name: small-signal r_d}.

    A diode's small-signal resistance is r_d = 1/(dI/dV) at its bias point, which
    the DiodeModel already gives as 1/conductance(V_D). At room temperature this is
    the classic r_d ≈ n·V_t/I_D ("26 mV / I_D"). A reverse-biased diode has a
    near-zero slope, so r_d is capped at a large finite value (effectively open).
    """
    res = solve_nonlinear(netlist, models=models, default_model=default_model)
    models = dict(models or {})
    rds: dict[str, float] = {}
    for c in netlist.components:
        if c.kind == "D":
            m = models.get(c.name, default_model)
            vd = res.node_voltages[c.nodes[0]] - res.node_voltages[c.nodes[1]]
            rds[c.name] = 1.0 / max(m.conductance(vd), 1e-12)
    return rds, res


def operating_point(netlist: Netlist, models=None, default_model: DiodeModel = SILICON) -> dict:
    """DC bias point → {diode_name: (V_D, r_d)} (voltage across, small-signal resistance)."""
    rds, res = _diode_resistances(netlist, models, default_model)
    return {c.name: (res.node_voltages[c.nodes[0]] - res.node_voltages[c.nodes[1]], rds[c.name])
            for c in netlist.components if c.kind == "D"}


def small_signal_ac(netlist: Netlist, freq: float, models=None,
                    default_model: DiodeModel = SILICON) -> ACResult:
    """AC at one frequency with each diode replaced by its small-signal r_d.

    Building block: linearise around the DC operating point, then run the linear
    AC solver with the netlist's source values as AC phasors. (For a normalised
    transfer function with the bias sources zeroed for AC, use
    small_signal_transfer_function.)
    """
    rds, _ = _diode_resistances(netlist, models, default_model)
    lin = Netlist()
    for c in netlist.components:
        if c.kind == "D":
            lin.add("R", f"{c.name}__rd", rds[c.name], c.nodes[0], c.nodes[1])
        else:
            lin.add(c.kind, c.name, c.value, c.nodes[0], c.nodes[1])
    return solve_ac(lin, freq)


def small_signal_transfer_function(netlist: Netlist, freqs, output_node: str,
                                   input_source: str, models=None,
                                   default_model: DiodeModel = SILICON) -> dict:
    """H(f) = V(output)/V(input) for a circuit with diodes, linearised at the bias.

    The DC operating point is found from the netlist's source values (the bias);
    then for the AC sweep the input source is driven at amplitude 1 and every OTHER
    independent source is set to its AC value of zero — because a DC supply is an AC
    ground (a fixed voltage source → short, a fixed current source → open). This is
    the standard small-signal AC procedure, and it makes a diode a *bias-tunable*
    element: r_d depends on the DC current, so e.g. a diode + capacitor is a filter
    whose cutoff you set with a bias current.
    """
    src = next((c for c in netlist.components if c.name == input_source), None)
    if src is None or src.kind != "V":
        raise ACError(f"input_source {input_source!r} is not a voltage source in the netlist")
    rds, _ = _diode_resistances(netlist, models, default_model)

    ac_net = Netlist()
    for c in netlist.components:
        if c.kind == "D":
            ac_net.add("R", f"{c.name}__rd", rds[c.name], c.nodes[0], c.nodes[1])
        elif c.kind == "V":
            ac_net.add("V", c.name, 1.0 if c.name == input_source else 0.0, c.nodes[0], c.nodes[1])
        elif c.kind == "I":
            ac_net.add("I", c.name, 0.0, c.nodes[0], c.nodes[1])
        else:
            ac_net.add(c.kind, c.name, c.value, c.nodes[0], c.nodes[1])

    freq_list = list(freqs)
    H = [solve_ac(ac_net, f).node_voltages[output_node] for f in freq_list]
    mag = [abs(h) for h in H]
    return {
        "freq": freq_list,
        "H": H,
        "mag": mag,
        "mag_db": [20.0 * math.log10(m) if m > 0 else -np.inf for m in mag],
        "phase_deg": [math.degrees(cmath.phase(h)) for h in H],
    }


def save_bode_plot(sweep: dict, path: str, title: str = "Bode plot") -> None:
    """Save a two-panel Bode plot (magnitude in dB and phase) from a sweep dict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax_mag.semilogx(sweep["freq"], sweep["mag_db"])
    ax_mag.set_ylabel("magnitude (dB)"); ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.set_title(title)
    ax_ph.semilogx(sweep["freq"], sweep["phase_deg"])
    ax_ph.set_ylabel("phase (deg)"); ax_ph.set_xlabel("frequency (Hz)")
    ax_ph.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def _demo() -> int:
    """RC low-pass + series-RLC band-pass, checked against their textbook formulas."""
    R, C = 1000.0, 1e-6
    fc = 1.0 / (2 * math.pi * R * C)
    lp = Netlist()
    lp.add("V", "Vin", 1, "in", "0"); lp.add("R", "R1", R, "in", "out"); lp.add("C", "C1", C, "out", "0")
    at_fc = solve_ac(lp, fc).gain("out")
    print("RC low-pass filter:")
    print(f"  cutoff f_c = {fc:.1f} Hz")
    print(f"  |H(f_c)| = {abs(at_fc):.4f} (theory 0.7071),  phase = {math.degrees(cmath.phase(at_fc)):+.1f}° (theory −45°)")
    save_bode_plot(transfer_function(lp, logspace(fc/100, fc*100), "out", "Vin"),
                   "bode_lowpass.png", title=f"RC low-pass (f_c ≈ {fc:.0f} Hz)")
    print("  saved bode_lowpass.png")

    L, C2 = 10e-3, 1e-6
    f0 = 1.0 / (2 * math.pi * math.sqrt(L * C2))
    bp = Netlist()
    bp.add("V", "Vin", 1, "in", "0"); bp.add("L", "L1", L, "in", "a")
    bp.add("C", "C1", C2, "a", "b"); bp.add("R", "R1", 100.0, "b", "0")
    peak = abs(solve_ac(bp, f0).gain("b"))
    off = abs(solve_ac(bp, f0 * 10).gain("b"))
    print("\nSeries-RLC band-pass (output across R):")
    print(f"  resonance f_0 = {f0:.1f} Hz")
    print(f"  |H(f_0)| = {peak:.4f} (theory 1.0),  |H(10·f_0)| = {off:.4f} (rolled off)")
    save_bode_plot(transfer_function(bp, logspace(f0/100, f0*100), "b", "Vin"),
                   "bode_bandpass.png", title=f"Series-RLC band-pass (f_0 ≈ {f0:.0f} Hz)")
    print("  saved bode_bandpass.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print("\nSmall-signal: bias-tunable RC low-pass (diode r_d + C, C = 1 µF):")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ibias in (1e-4, 1e-3):
        net = Netlist()
        net.add("V", "vin", 0.0, "in", "0")
        net.add("D", "D1", 0.0, "in", "mid")
        net.add("C", "C1", 1e-6, "mid", "0")
        net.add("I", "Ibias", ibias, "mid", "0")
        vd, rd = operating_point(net)["D1"]
        fc = 1.0 / (2 * math.pi * rd * 1e-6)
        sweep = small_signal_transfer_function(net, logspace(10, 1e6), "mid", "vin")
        ax.semilogx(sweep["freq"], sweep["mag_db"], label=f"I_bias={ibias*1e3:.1f} mA  (r_d≈{rd:.0f}Ω, f_c≈{fc:.0f} Hz)")
        print(f"  I_bias={ibias*1e3:>4.1f} mA -> r_d = {rd:6.1f} Ω -> cutoff f_c = {fc:6.0f} Hz")
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("magnitude (dB)")
    ax.set_title("Bias-tunable low-pass: a diode's r_d sets the cutoff")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig("bode_diode_tunable.png", dpi=110); plt.close(fig)
    print("  saved bode_diode_tunable.png (cutoff shifts ~10x with 10x bias)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
