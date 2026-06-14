#!/usr/bin/env bash
#
# reproduce.sh — regenerate every headline number and figure in one command.
#
# Why this exists: a reviewer (or future me) should be able to reproduce the
# results in EXPLAINED.md / the writeup from a clean checkout, exactly. This runs
# the test suite, the metrics (with their fixed seeds), and the solver demos, in
# order, and writes the figures (rc_charging.png, rectifier.png,
# noise_robustness.png, extractor_ablation.png).
#
# Usage:  ./reproduce.sh        (from the repo root)
#
# Determinism: PYTHONHASHSEED is fixed so dict/set ordering can't drift; every
# metric already takes an explicit --seed. The numbers below should match the
# tables in EXPLAINED.md to the digit.

set -u
cd "$(dirname "$0")"
export PYTHONHASHSEED=0

PY=.venv/bin/python
if [ ! -x "$PY" ]; then
    echo "error: $PY not found. Create the venv and 'uv pip install -r requirements.txt' first." >&2
    exit 1
fi

hr() { printf '\n========== %s ==========\n' "$1"; }

hr "1/6  Test suite"
$PY -m pytest -q

hr "2/6  End-to-end extraction accuracy (200 circuits, seed 0)"
$PY -m metrics.extraction_accuracy --count 200 --seed 0

hr "3/7  Extractor ablation: blob-proximity vs skeleton-graph (30 seeds × 5 layouts)"
$PY -m metrics.extractor_ablation --seeds 30

hr "4/7  Difficulty curve: extraction accuracy vs circuit size"
$PY -m metrics.difficulty --seeds 12 --max-size 12

hr "5/7  Noise-robustness study (80 circuits/level, seed 0) — takes a few minutes"
$PY -m metrics.noise_robustness --count 80 --seed 0

hr "6/7  Solver demos (transient, nonlinear diodes, AC/Bode, SPICE export)"
$PY -m solver.transient
$PY -m solver.nonlinear
$PY -m solver.ac
$PY -m solver.spice_export

hr "7/7  ngspice validation (skips cleanly if ngspice isn't installed)"
$PY -m solver.ngspice_validation || echo "(ngspice not installed — validation skipped; install it to run this step)"

hr "Done"
echo "Figures: rc_charging.png, rlc_ringing.png, rectifier.png, bode_lowpass.png, bode_bandpass.png, bode_diode_tunable.png, noise_robustness.png, extractor_ablation.png, difficulty_curve.png"
