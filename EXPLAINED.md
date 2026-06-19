# What the code does

This is a quick overview of the project so anyone reading the repo can get the general idea.
It is not meant to be deep. If you want the full write up with results, read docs/paper.md.

The whole project takes a picture of a hand drawn circuit and turns it into a solved circuit.
The steps are: find the parts in the image, figure out how the wires connect them, write that
down as a netlist, and then solve the circuit for the voltages and currents. Right now the
image side works on circuits I generate myself, and the math side is fully built.

Below is what lives in each folder.

## solver/

This is the math part. It takes a netlist and works out the answer.

- netlist.py holds the circuit as data. A component (resistor, source, diode, and so on) and
  the list of all of them, plus reading and writing simple SPICE text.
- mna.py is the main solver for DC circuits. It uses Modified Nodal Analysis, which is the same
  method real simulators use. It builds a matrix from the circuit and solves it for the node
  voltages.
- transient.py handles circuits that change over time, like a capacitor charging up. It steps
  through time and reuses the DC solver at each step.
- nonlinear.py handles diodes and LEDs, which do not follow a straight line. It uses
  Newton's method to close in on the answer.
- ac.py handles frequency response, so you can see how a filter reacts to different frequencies.
  It runs the same solver but with complex numbers.
- spice_export.py writes the circuit out as a real SPICE file you could open in other tools.
- ngspice_validation.py checks my solver against ngspice, a standard simulator, to make sure my
  numbers are right.
- equivalence.py checks if two circuits are really the same circuit, which the accuracy tests
  use.

## vision/

This is the part that reads the image.

- preprocessing turns the photo into a clean black and white image.
- wire_extraction.py and skeleton_graph.py are the hard part. They erase the components, thin
  the wires down to single lines, and turn those lines into a graph of connections so the code
  knows what is wired to what.
- wire_extraction_baseline.py is the old version, kept around so I can compare old versus new.
- debug_viz.py draws what the code thinks it sees, which helps when something goes wrong.

## data_collection/

- synthetic.py makes fake circuit images on the computer, along with the correct answer for
  each one. This is how I test everything without needing real photos yet.
- extra_layouts.py has a few more circuit shapes used for harder tests.

## metrics/

These scripts measure how well things work.

- extraction_accuracy.py measures how often the wire reader gets the circuit right.
- extractor_ablation.py compares the old wire reader to the new one.
- difficulty.py measures accuracy as circuits get bigger.
- noise_robustness.py measures how accuracy drops when the image gets blurry or noisy.

## training/

Scripts to train the component detector on a public dataset of hand drawn circuits. The scripts
are written and tested, but the actual training has not been run yet. That is the next big step.

## tests/

Automated tests for everything above. There are a couple hundred of them, and they all pass.
They are what let me trust that the code actually works.

## other files

- demo.py runs the whole thing end to end on a sample circuit.
- reproduce.sh runs the tests, all the measurements, and the demos with one command, so all the
  numbers and figures can be regenerated.
