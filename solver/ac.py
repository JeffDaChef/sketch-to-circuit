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


class ACError(Exception):
    """Raised for invalid AC runs (non-positive frequency, unsupported parts...)."""


@dataclass
class ACResult:
    """The phasor solution at ONE frequency: complex voltage at every node."""

    freq: float
    node_voltages: dict[str, complex]      # net name -> complex phasor (volts)
    source_currents: dict[str, complex]    # voltage-source name -> complex phasor (amps)

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

    # R, C and L all stamp identically — as a complex admittance Y between their
    # two nodes (this is the whole elegance of the phasor view).
    for c in netlist.components:
        if c.kind == "R":
            y = 1.0 / c.value
        elif c.kind == "C":
            y = 1j * w * c.value
        elif c.kind == "L":
            y = 1.0 / (1j * w * c.value)
        elif c.kind == "I":
            a, b = c.nodes                          # current-source phasor -> RHS
            if a != GROUND:
                z[node_index[a]] -= c.value
            if b != GROUND:
                z[node_index[b]] += c.value
            continue
        else:                                        # V handled below; D already rejected
            continue
        a, b = c.nodes
        if a != GROUND:
            A[node_index[a], node_index[a]] += y
        if b != GROUND:
            A[node_index[b], node_index[b]] += y
        if a != GROUND and b != GROUND:
            A[node_index[a], node_index[b]] -= y
            A[node_index[b], node_index[a]] -= y

    # Voltage sources: the same +1/-1 augmentation as DC MNA (the "modified" part).
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


# --- frequency sweep + transfer function -------------------------------------

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
    # RC low-pass: cutoff f_c = 1/(2πRC); at f_c, |H| = 1/√2 (−3 dB), phase −45°.
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

    # Series RLC band-pass (output across R): peaks at f_0 = 1/(2π√(LC)), |H|=1 there.
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
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
