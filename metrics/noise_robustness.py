"""Noise-robustness study: how does extraction accuracy degrade as images get worse?

WHY THIS EXISTS
---------------
On the clean synthetic images we control, extraction scores 100%. Real hand-drawn
photographs will be blurry, grainy, speckled, and unevenly lit. We can't test on
real photos yet (that needs the trained detector), but we *can* predict — with
numbers — what will break, by deliberately corrupting the synthetic images at
increasing severity and watching the accuracy fall. The output is an
**accuracy-vs-severity curve** per corruption type: an honest, quantitative
forecast of the pipeline's weak points, and a Phase-2 to-do list ranked by impact.

This is the "noise-robustness study" from ROADMAP.md, and it doubles as honest
limitations material for the writeup: reporting *where and how* something breaks
is the most credible engineering move there is.

WHAT WE CORRUPT (and why these)
-------------------------------
* **Gaussian blur** — an out-of-focus or low-resolution photo.
* **Gaussian noise** — sensor grain / poor lighting.
* **Speckle (salt-and-pepper)** — dust, JPEG specks, pen spatter.

Each corruption keeps the ground-truth component boxes valid (no geometry change),
so the only thing under test is the wire tracing. We don't include rotation (it
would move the axis-aligned boxes) here.

DETERMINISM
-----------
Every base circuit gets a fixed integer seed. A corruption that needs randomness
regenerates its random field from that seed, so (a) results are perfectly
reproducible and (b) a higher severity acts on the *same* random field as a lower
one — a fair *nested* sweep (more severity = strictly more corruption) rather than
independent draws per level. Note this makes the corruption nested, not the
*accuracy curve* strictly monotonic: the extractor isn't monotonic in its input,
so a slightly worse image can occasionally happen to extract correctly. The curves
trend downward but may wiggle, especially at small sample counts.
"""

from __future__ import annotations

import argparse
import random
import signal
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from skimage.filters import gaussian

from data_collection.synthetic import generate_one
from solver.equivalence import circuit_equivalent
from solver.netlist import Netlist
from vision.wire_extraction import extract_netlist



def blur(image: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Gaussian blur with standard deviation `sigma` pixels (deterministic)."""
    if sigma <= 0:
        return image
    out = gaussian(image, sigma=sigma, channel_axis=-1)
    return (out * 255.0).astype(np.uint8)


def gaussian_noise(image: np.ndarray, std: float, seed: int) -> np.ndarray:
    """Add zero-mean Gaussian noise of standard deviation `std` (in 0-255 units)."""
    if std <= 0:
        return image
    field = np.random.default_rng(seed).standard_normal(image.shape)
    return np.clip(image + std * field, 0, 255).astype(np.uint8)


def speckle(image: np.ndarray, amount: float, seed: int) -> np.ndarray:
    """Salt-and-pepper: fraction `amount` of pixels forced to black or white.

    Uses a fixed uniform field per seed, so a larger `amount` corrupts a strict
    superset of the pixels a smaller one did — a fair nested sweep (the accuracy
    curve still reflects the extractor's own, not-strictly-monotonic, sensitivity).
    """
    if amount <= 0:
        return image
    field = np.random.default_rng(seed).random(image.shape[:2])
    out = image.copy()
    out[field < amount / 2] = 0
    out[field > 1 - amount / 2] = 255
    return out


@dataclass(frozen=True)
class Corruption:
    name: str
    fn: object
    levels: tuple[float, ...]
    label: str


CORRUPTIONS = [
    Corruption("blur", blur, (0.0, 1.0, 2.0, 3.0, 4.0, 5.0), "Gaussian blur (σ px, ≤5)"),
    Corruption("noise", gaussian_noise, (0.0, 15.0, 30.0, 45.0, 60.0, 75.0), "Gaussian noise (σ/255, ≤75)"),
    Corruption("speckle", speckle, (0.0, 0.004, 0.008, 0.012, 0.016, 0.020), "Speckle (fraction, ≤2%)"),
]



@dataclass
class BaseCircuit:
    image: np.ndarray
    components: list
    truth: Netlist
    n_components: int
    seed: int


def make_base_set(count: int, seed: int = 0) -> list[BaseCircuit]:
    """Render `count` synthetic circuits once; corruptions are applied to these."""
    rng = random.Random(seed)
    bases: list[BaseCircuit] = []
    for i in range(count):
        img, gt = generate_one(rng)
        truth = Netlist.from_spice(gt["netlist_spice"])
        bases.append(BaseCircuit(
            image=np.asarray(img).copy(),
            components=gt["components"],
            truth=truth,
            n_components=len(truth.components),
            seed=i,
        ))
    return bases


@contextmanager
def _time_limit(seconds: float):
    """Raise TimeoutError if the wrapped block runs longer than `seconds`.

    Heavily corrupted images (e.g. speckle) blow the skeleton graph up to thousands
    of nodes, so a single extraction can take pathologically long. We cap it: an
    image we can't process in a few seconds is, for our purposes, a failure — which
    is itself an honest result, not something to hide. Uses SIGALRM, so it only arms
    in the main thread (a no-op elsewhere, e.g. under some test runners).
    """
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"extraction exceeded {seconds:g}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _is_correct(base: BaseCircuit, image: np.ndarray, timeout: float = 8.0) -> bool:
    """Run extraction on one (corrupted) image; a crash OR timeout counts as wrong."""
    try:
        with _time_limit(timeout):
            extracted = extract_netlist(image, base.components)
        return circuit_equivalent(extracted, base.truth)
    except Exception:
        return False


def score(bases: list[BaseCircuit], corruption: Corruption, severity: float) -> tuple[int, int]:
    """Return (correct, total) over the base set at one corruption severity."""
    correct = sum(_is_correct(b, corruption.fn(b.image, severity, b.seed)) for b in bases)
    return correct, len(bases)



def run_study(count: int = 80, seed: int = 0, bases: list[BaseCircuit] | None = None,
              corruptions: list[Corruption] | None = None) -> dict:
    """Sweep every corruption over its severity levels; return accuracy curves."""
    bases = bases if bases is not None else make_base_set(count, seed)
    out: dict[str, dict] = {}
    for c in (corruptions if corruptions is not None else CORRUPTIONS):
        accs = []
        for sev in c.levels:
            correct, total = score(bases, c, sev)
            accs.append(100.0 * correct / total if total else 0.0)
        out[c.name] = {"levels": list(c.levels), "accuracy": accs, "label": c.label}
    return out


def accuracy_by_complexity(bases: list[BaseCircuit], corruption: Corruption,
                           severity: float) -> dict[int, list[int]]:
    """At one corruption severity, break accuracy down by component count.

    The hypothesis is that bigger circuits (more wires to trace) fail first under
    noise. Treat this as exploratory: our circuits only span 3-5 components and the
    per-bucket counts are small, so a clean monotonic 'difficulty curve' may not
    show up until there's both more size range and a larger sample.
    """
    by_n: dict[int, list[int]] = {}
    for b in bases:
        ok = _is_correct(b, corruption.fn(b.image, severity, b.seed))
        bucket = by_n.setdefault(b.n_components, [0, 0])
        bucket[1] += 1
        if ok:
            bucket[0] += 1
    return by_n


def save_plot(study: dict, path: str) -> None:
    """Save the accuracy-vs-severity curves (x = relative severity 0..1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, data in study.items():
        levels = data["levels"]
        span = levels[-1] - levels[0] or 1.0
        rel = [(lv - levels[0]) / span for lv in levels]
        ax.plot(rel, data["accuracy"], marker="o", label=data["label"])
    ax.set_xlabel("relative corruption severity  (0 = clean, 1 = max)")
    ax.set_ylabel("extraction accuracy (%)")
    ax.set_title("Wire-extraction robustness to image corruption")
    ax.set_ylim(-3, 103)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _print_study(study: dict) -> None:
    for name, data in study.items():
        print(f"\n{data['label']}")
        print(f"  {'severity':>10}  {'accuracy':>9}")
        print(f"  {'-'*10}  {'-'*9}")
        for lv, acc in zip(data["levels"], data["accuracy"]):
            print(f"  {lv:>10g}  {acc:>8.1f}%")


def main() -> None:
    p = argparse.ArgumentParser(description="Noise-robustness study for wire extraction.")
    p.add_argument("--count", type=int, default=80, help="circuits per severity level (default 80)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", default="noise_robustness.png", help="output PNG path")
    args = p.parse_args()

    print(f"Rendering {args.count} base circuits (seed={args.seed}) …")
    bases = make_base_set(args.count, args.seed)
    print(f"Clean baseline: {sum(_is_correct(b, b.image) for b in bases)}/{len(bases)} correct")

    study = run_study(bases=bases)
    _print_study(study)

    sev = 45.0
    noise = next(c for c in CORRUPTIONS if c.name == "noise")
    print(f"\nAccuracy by circuit complexity under Gaussian noise σ={sev:g}:")
    print(f"  {'# components':>12}  {'accuracy':>9}")
    print(f"  {'-'*12}  {'-'*9}")
    by_n = accuracy_by_complexity(bases, noise, sev)
    for n in sorted(by_n):
        c, t = by_n[n]
        print(f"  {n:>12}  {100.0*c/t:>8.1f}%  ({c}/{t})")

    save_plot(study, args.plot)
    print(f"\nSaved curves to {args.plot}")


if __name__ == "__main__":
    main()
