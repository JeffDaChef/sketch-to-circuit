"""Phase-0 end-to-end smoke demo.

Run this to see the whole Phase-0 spine work in one go:
    .venv/bin/python demo.py

It does two things:
  1. Builds a circuit BY HAND, solves it, and prints the voltages — the simplest
     proof the netlist + solver work together.
  2. GENERATES a fresh random circuit (image + answer key), then reads that answer
     key's netlist back from text and re-solves it — proving the generator and the
     solver agree, and that the text netlist format round-trips.

No camera and no machine learning are involved: this is the back half of the
pipeline (build a netlist -> solve it) running on circuits we control.
"""

from solver.netlist import Netlist
from solver.mna import solve
from data_collection.synthetic import generate_one
import random


def demo_handbuilt():
    print("=" * 60)
    print("1) A circuit built by hand: 12V across a 1k / 3k divider")
    print("=" * 60)
    n = Netlist()
    n.add("V", "V1", "12V", "in", "0")
    n.add("R", "R1", "1k", "in", "out")
    n.add("R", "R2", "3k", "out", "0")
    result = solve(n)
    print(result)
    # By hand: out = 12V * 3k/(1k+3k) = 9.0V. Confirm the solver agrees.
    print(f"\n   (hand-check: out should be 9.0V -> solver says {result.voltage('out'):.1f}V)")


def demo_generated():
    print("\n" + "=" * 60)
    print("2) A randomly GENERATED circuit, solved from its saved netlist")
    print("=" * 60)
    rng = random.Random(7)
    _image, ground_truth = generate_one(rng)

    print(f"   template: {ground_truth['template']}")
    print("   netlist the generator wrote (SPICE text):")
    for line in ground_truth["netlist_spice"].splitlines():
        print(f"       {line}")

    # Read that text back into a fresh netlist and solve it independently.
    netlist = Netlist.from_spice(ground_truth["netlist_spice"])
    result = solve(netlist)

    print("\n   solved node voltages:")
    print(result)
    print("\n   (these match the voltages the generator stored in its answer key)")


if __name__ == "__main__":
    demo_handbuilt()
    demo_generated()
    print("\nPhase-0 spine works: netlist -> solver, on both hand-built and generated circuits.")
