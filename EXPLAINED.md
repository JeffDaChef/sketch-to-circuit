# EXPLAINED.md — the whole project, in plain English

This is the project's "textbook." It explains **how the system is built and how to think
while building it** — not the syntax. You won't find code here. Each file is described
twice over: first *where it sits in the bigger machine and what connects to it*, then a
walk through its logic as a sequence of plain-English moves ("now I'm doing this, because…").
When one file leans on another, it's called out explicitly, because the most important
thing to understand is **how the pieces talk to each other.**

Read it top to bottom and you'll understand the reasoning behind every part we've built.

---

## The big picture: the conveyor belt

The finished system is a conveyor belt. A photo of a hand-drawn circuit goes in one end,
and each station transforms it into something closer to a solved, annotated circuit:

1. **Camera** grabs a frame of the paper.
2. **Flatten & clean** the image (correct the camera angle, turn it into crisp black-on-white).
3. **Detect components** — find the resistors, batteries, etc. *(this is the trained AI part)*.
4. **Trace the wires** connecting those components.
5. **Build the netlist** — write the circuit down as data.
6. **Read the values** off the handwriting ("10k", "5V").
7. **Solve the circuit** — compute the voltage at every junction *(the math engine)*.
8. **Overlay the answers** back onto the live video, and explain the circuit in English.

We are deliberately building the **back half first** (stations 5 and 7), using fake
computer-drawn circuits where we already know the right answer. That lets us build and
*prove* the hard math today, with no camera and no AI training. When the trained detector
arrives later, it plugs into a back end that already works.

**What's built so far (Phase 0):** the netlist (station 5), the solver (station 7), and a
generator that produces fake circuits to test them with. Those three are what this
document currently covers.

---

## Stage 0 — Setting up the workshop

Before any project code, the computer needed the right tools. None of this is part of the
circuit system; it's the workbench everything else is built on.

- **The terminal** is a way to drive the computer by typing commands instead of clicking.
  Every setup step was one typed command.
- **Apple's built-in Python was too old**, and you should never install project tools into
  the system's own copy (it can break the operating system). So we needed a fresh, private
  Python.
- **Homebrew** is the usual Mac tool for installing developer software, but installing *it*
  needs an administrator password, and this Mac account isn't an administrator. So we used
  **uv** instead — a fast tool that installs everything inside your own user folder, no
  admin needed. With one command it downloaded a private **Python 3.12** and built a
  **virtual environment** (a sealed toolbox of libraries that belongs to this project
  alone, so it can never clash with anything else on the machine).
- Into that toolbox we installed five libraries: **schemdraw** (draws circuit diagrams),
  **numpy** (fast matrix math — the engine of the solver), **networkx** (graph math, for a
  later phase), **matplotlib** (the rendering engine schemdraw draws onto), and **pytest**
  (runs our tests).
- **git** is a save-point system. Every working chunk gets "committed" — frozen as a named
  snapshot you can always return to. The chain of commits is a real, honest record of the
  work building up over time. A `.gitignore` file lists junk that shouldn't be saved (the
  toolbox, generated images, hidden Mac files).

**The repo layout:** `solver/` holds the circuit math, `data_collection/` holds the fake-
circuit generator, `tests/` holds the checks, `docs/` holds the drawing-rules spec, and
the folders `vision/`, `training/`, `ui/` are empty placeholders for later phases.

---

## File 1 — `solver/netlist.py` — describing a circuit as data

**Where this fits in the pipeline:** Station 5. It defines the *shared language* the whole
project speaks — the way any circuit is written down as data.

**What feeds in / what it feeds:** Later, the wire-tracer (station 4) will *produce* these
netlist objects from a photo. Right now, the generator (File 3) and our tests produce them
by hand. Everything downstream — above all the solver (File 2) — *consumes* them. So this
file is the hinge the whole project turns on.

**The core idea first:** a circuit is just a parts list plus wiring. Each part says what it
is, what it's worth, and which two *nets* it touches. A "net" is one connected blob of
wire — every point on it is electrically the same, so it gets a single name. Ground (the
reference point we measure all voltages against) is always the net named "0". Where things
are drawn on the page doesn't matter electrically; only what connects to what.

**How it's built, move by move:**

- First I'm pulling in a couple of small built-in tools: one for **pattern-matching text**
  (I'll need it to read written values like "10k"), and one shorthand for making tidy
  **labeled records** without busywork.

- Now I'm writing down the **list of component types** the project allows — resistor,
  capacitor, voltage source, current source, diode — as a lookup from a one-letter code to
  a human name. Anything not on this list will be rejected later. I'm also fixing the name
  of the **ground net as "0"**, once, so the rest of the code refers to a single shared
  constant instead of scattering "0" everywhere.

- Now I'm defining my **own kind of error** for circuit problems. When something is wrong —
  a value that can't be read, an unknown component — I'll raise this. Having a custom,
  clearly-named error means the tests and the rest of the program can recognize *our*
  circuit problems specifically, and not confuse them with unrelated computer errors.

- Now I'm setting up the data I need to **read written values**. There's a table saying
  what each engineering letter multiplies by — "k" means times a thousand, "u" means
  divide by a million, and so on. And there are three text patterns I prepare in advance
  and reuse: one that spots a trailing unit like "V" or "F" at the end, one that matches a
  plain value such as "10k", and one that matches the European style where the letter sits
  in the middle and stands in for the decimal point, like "4u7" meaning 4.7 millionths.

- Now I'm writing the **value reader** itself — the single most reused little routine in
  the project. It runs in a deliberate order. First it cleans the text by removing spaces.
  If nothing is left, that's an error and it stops. Then it tries to chop off a trailing
  unit, but only if doing so still leaves a sensible number, so it never mangles a plain
  digit. Then it tests the "4u7" middle-letter shape first; if that matches, it rebuilds
  the number around a decimal point and scales it by the letter's multiplier. Otherwise it
  tests the ordinary "10k" shape and scales that. If neither shape fits, it gives up loudly
  with a clear error rather than guessing. *This same reader will later be the safety net
  behind the handwriting-recognition station (station 6), which is why it's careful and
  lives here in the shared language.*

- Now I'm defining what **one component** looks like as a record: its type, its name like
  "R1", its value as a plain number, and the pair of net names its two ends touch. The
  order of those two ends carries meaning for sources — the first one is the "+" terminal.
  The moment a component is created it checks itself: the type must be on the allowed list,
  and there must be exactly two ends. If not, it raises the custom error immediately, so
  bad data can't travel any further into the system.

- Now I'm defining the **whole circuit** as a container that mainly holds a list of those
  components, and giving it the handful of things a circuit needs to be able to do:
  - **Add a component.** It refuses a duplicate name, and if the value arrived as text like
    "10k" it quietly runs it through the value reader from earlier so the stored value is
    always a real number. This is the one method the rest of the project calls to build a
    circuit up piece by piece.
  - **List the nodes.** It gathers every net name mentioned by any component, drops ground,
    and returns the rest in order. *The solver calls this to learn exactly what the unknown
    voltages are* — so this method is the bridge into File 2.
  - **Check there's a ground.** It reports whether anything touches net "0". A circuit with
    no ground has no reference point, so the solver checks this before doing anything.
  - **Write the circuit out as standard text**, and **read it back in.** This is the same
    plain-text format the professional simulator uses — one line per component. Having it
    means we can save circuits to files, hand them to the real simulator for cross-checking
    in a later phase, and write test circuits by hand as plain text.

---

## File 2 — `solver/mna.py` — the circuit solver (the centerpiece)

**Where this fits in the pipeline:** Station 7, the math engine — the single most important
and most impressive part of the project. It's the same core algorithm that runs inside
SPICE, the industry-standard circuit simulator, written here from scratch.

**What feeds in / what it feeds:** It *takes in* a netlist (File 1) and *gives back* the
voltage at every node plus the currents. The generator (File 3) calls it to fill in the
answer key for each fake circuit; later, the live overlay (station 8) will call it every
time you change a value on the paper.

**The core idea first:** the unknowns in any circuit are the voltages at the junctions. Two
physical laws give us enough equations to pin them down. **Ohm's law:** the current through
a resistor equals the voltage across it divided by its resistance. **Kirchhoff's Current
Law:** at every junction, the current flowing in equals the current flowing out — nothing
accumulates. If I write that current-balance at every junction, and express each current in
terms of the unknown voltages, I get one equation per junction. Stack all those equations
into a grid of numbers (a matrix) and hand it to numpy, and out come the voltages. The
"modified" twist is that an ideal battery *forces* a voltage instead of obeying Ohm's law,
so each battery needs one extra unknown (its own current) and one extra equation.

**How it's built, move by move:**

- First I'm pulling in the **numpy math library**, because solving many equations at once is
  exactly the matrix arithmetic it's built for. I'm also importing the **ground constant and
  the netlist type from File 1**, so the solver speaks the same language the netlist was
  written in. *This import is the concrete link between the two files.*

- Now I'm defining a solver-specific **error** (same idea as before, for solver problems)
  and a **result record** that will hold the answer in three labeled tables: the voltage at
  every net, the current through every battery, and the current through every resistor. The
  result also knows how to **print itself** as a tidy human-readable summary — that's what
  produces the neat voltage list you see when running a demo.

- Now the **solve routine** begins, and the very first thing it does is **refuse circuits it
  can't handle**: one with no ground (no reference point, so "voltage" would be meaningless),
  or one containing a diode (those bend the rules of Ohm's law and need a more advanced
  method saved for a later phase). Failing fast with a clear message beats producing
  nonsense quietly.

- Now I'm **deciding what the unknowns are.** I ask the netlist for its list of nodes
  (*calling straight into File 1's "list the nodes" method*) and give each one a row number.
  Then I find the batteries and give *each of them* its own extra row number, placed after
  the node rows. The total count of rows is the total number of unknowns, which is also the
  number of equations — the grid will be exactly that size, square.

- Now I'm creating an **empty grid and an empty right-hand-side list**, both filled with
  zeros, sized to match. Everything from here is about filling them in. The technique is
  called "stamping": I walk the components one at a time and each one adds its own small,
  fixed contribution into the grid. Components add up independently, which is what makes
  this systematic enough for a computer.

- Now I'm **stamping the resistors and current sources** in one pass over the components.
  For a resistor I work out its conductance (just one divided by its resistance) and add it
  into four spots in the grid — twice on the diagonal for its two nodes, and twice on the
  crossing spots that link them. That four-spot pattern *is* the current-balance-plus-Ohm's-
  law for that resistor, written into numbers. Any spot that would land on ground is skipped,
  because ground has no row. For a current source — which simply forces a fixed current — the
  value goes onto the right-hand-side list instead: it pulls current out of one node and
  pushes it into the other. A capacitor blocks steady current, so at this DC stage it does
  nothing and I let it fall through untouched.

- Now I'm **stamping the batteries** — the "modified" part. A battery insists that the
  voltage difference between its two ends equals its value, which can't be written with
  conductances. So for each battery I place simple +1 and −1 markers where its two nodes
  meet its own extra row and column. Those markers do two jobs at once: they feed the
  battery's unknown current into the two nodes' balance equations, and they state the rule
  "this end minus that end equals the battery's voltage." That voltage goes onto the right-
  hand side at the battery's row.

- Now the grid and the right-hand side are complete, so I'm **asking numpy to solve the whole
  system in one step.** It returns a single list of numbers — all the unknown voltages and
  battery currents together. If numpy reports the system has no unique answer (which happens
  for a broken circuit like a floating, unconnected node), I catch that and re-raise it as a
  friendly error explaining the likely cause, instead of letting a cryptic math error escape.

- Now I'm **translating that raw list of numbers back into meaningful, named results.**
  Ground is set to exactly zero. Each node's row becomes that net's voltage; each battery's
  row becomes that battery's current. Then I compute every resistor's current from the
  voltages I just found, using Ohm's law again — these will drive the current-arrow overlay
  later. Finally I bundle all three tables into the result record and hand it back. *That
  returned record is exactly what File 3 reads to build its answer key.*

---

## File 3 — `data_collection/synthetic.py` — the fake-circuit generator

**Where this fits in the pipeline:** It's *not* a station on the belt. It's a workshop tool
that manufactures test material — clean, computer-drawn circuits — so we can exercise the
real stations without hand-drawing and hand-labeling hundreds of circuits.

**What feeds in / what it feeds:** It *uses* File 1 (to record each circuit as a netlist)
and File 2 (to solve each circuit so the answer is included). It *produces*, for every
sample, two matching files: a picture, and an "answer key" listing every component, exactly
where it sits in the picture, the circuit as text, and the solved voltages. Those answer
keys are what the vision stations (3 and 4) will later be measured against.

**The core idea first:** because our own code places every part on the page, it knows the
truth about each one for free. So as it draws, it simultaneously writes down the matching
netlist and the location of each part. That guarantees the picture and the answer key can
never disagree.

**How it's built, move by move:**

- First, a small but important housekeeping move: I'm **adding the project's main folder to
  the list of places this script looks for code**, because a script only searches its own
  folder by default, and the solver lives one level up. Without this, the script couldn't
  reach Files 1 and 2 at all. Then I import the drawing library, the math library, an
  image library, and — crucially — the **solve routine from File 2 and the netlist type
  from File 1.**

- Now I'm listing the **realistic values to choose from** — a pool of common resistor sizes
  and battery voltages — so the generated circuits look like things a real person would draw.

- Now I'm defining a small **record for one component's ground-truth label**: its name, its
  type, its written value, and its box — the rectangle, in pixels, marking where it sits in
  the image.

- Now I'm writing the **circuit templates**, which are the heart of this file. Each template
  is a recipe that draws one *style* of circuit and, in lockstep, records the matching
  netlist and the list of parts to box. The key discipline is that drawing and recording
  happen together, step for step, so they always match.
  - The **series divider** places a battery on the left going up, runs a wire across the
    top, then stacks a random number of resistors going down the right side back to ground.
    As it places each resistor it *immediately* records that resistor in the netlist,
    connecting it between the node above and the node below — so the electrical description
    is built at the same instant as the drawing. It drops a junction dot at each internal
    connection, matching the drawing rule that every junction gets a dot.
  - The **parallel bank** places a battery on the left, then extends a top wire and a bottom
    wire to the right *together, one step at a time*, and bridges them with a resistor at
    each step. Because every resistor bridges the same top wire and the same bottom wire,
    they all share the same two nodes — which is exactly what "in parallel" means. Drawing
    both rails in lockstep is what guarantees each resistor is genuinely connected at both
    ends, rather than dangling. *(An earlier version forgot the bottom rail, so the picture
    showed open circuits that disagreed with the netlist; building both rails together fixed
    it — a good example of the picture and the answer key having to be kept honest with each
    other.)*

- Now I'm collecting the templates into a **list to pick from at random**, so the dataset
  gets a mix of circuit styles.

- Now I'm writing the **box-to-pixels converter**, which solves a subtle but real problem.
  The drawing library knows where each part is in its own diagram units, but the answer key
  needs pixel locations in the final image. The naive conversion is thrown off by Retina
  screens, where the real image is secretly twice the size the drawing tool reports. So
  instead I convert each part's location into a **fraction of the whole figure** — a zero-to-
  one coordinate that doesn't care about screen resolution — and only then multiply by the
  image's *true* pixel size. I also flip the vertical direction, because diagrams measure up
  from the bottom while images count rows down from the top. *(Getting this right was the
  fiddliest part of the file; the first attempt put every box in the wrong place, and I only
  caught it by drawing the boxes onto a sample image and looking at them.)*

- Now I'm writing the routine that **makes one complete sample.** It picks a template at
  random and runs it, getting back the drawing, the netlist, and the list of parts to box.
  It renders the drawing, then reaches down to the real underlying figure and forces it to
  draw so the pixels actually exist. It reads the finished image straight out of the render
  buffer, and — to dodge the Retina trap — takes the **true width and height from the image
  itself** rather than trusting the reported size. Then for each recorded part it asks the
  drawing library for that part's location and runs it through the box-to-pixels converter.
  Next it **calls the solver from File 2 on the netlist** to get the voltages, so the answer
  key is complete. Finally it assembles everything — the circuit style, the image size, the
  list of boxed components, the circuit as text, and the solved voltages — into one tidy
  answer-key bundle, and returns that alongside the image.

- Now I'm writing the **main entry point**, the part that runs when you launch the script.
  It reads the options you typed — how many circuits to make, which folder to put them in,
  and a seed number that makes the random choices repeatable so the same command always
  produces the same dataset. Then it loops the requested number of times, and for each one
  it saves the picture and its matching answer key side by side under the same name, and
  prints a one-line note about what it made.

---

## The tests — `tests/test_netlist.py` and `tests/test_mna.py`

**Where this fits:** not a station, but the safety net under Files 1 and 2. The project's
rule is that every module is checked on its own, so when something breaks you know exactly
which part broke.

**What feeds in / what it feeds:** the tests build small circuits by hand using File 1, run
them through File 2, and check the answers against numbers worked out on paper.

**How they work, in plain English:** each test is a tiny self-contained check that feeds a
known input in and insists on a known answer. The value-reader tests confirm that "10k"
becomes ten thousand, that "4u7" becomes 4.7 millionths, that a trailing unit is ignored,
and that nonsense is rejected. The solver tests use circuits simple enough to solve by hand:
a voltage divider where the midpoint must land at exactly half; resistors in series and in
parallel with currents you can check by hand; a two-battery circuit; a current source; a
capacitor that must change nothing at this DC stage; and the deliberately broken cases — no
ground, a floating node, a diode — which must each raise a clear error rather than return a
wrong number. There's also one test that writes a circuit as text, reads it back, and solves
it, proving Files 1 and 2 work together end to end. A single command runs all of them at
once and reports pass or fail. A companion empty file at the project root simply tells the
test runner where the project begins, so the tests can find the `solver` code.

---

## File 4 — `demo.py` — the end-to-end proof

**Where this fits in the pipeline:** not a station — a demonstration that the whole back
half of the belt works as one connected machine.

**What feeds in / what it feeds:** it pulls together all three built files — it builds
netlists (File 1), solves them (File 2), and generates a fresh circuit (File 3) — and
prints the results for a human to read.

**How it works, move by move:**

- First I'm importing the netlist type, the solve routine, and the generator's
  make-one-circuit routine — so this one script can reach every piece we've built.

- Now I'm running the **hand-built demo**: I construct a simple 12-volt divider with a
  1k and a 3k resistor by adding components one at a time, solve it, and print the
  voltages. I also print the paper answer (the output should be nine volts) right next to
  the solver's answer, so anyone can confirm by eye that the engine is telling the truth.

- Now I'm running the **generated demo**, which proves the loop closes. I ask the
  generator for one fresh random circuit; it hands back an image and an answer key. I take
  the circuit-as-text out of that answer key, **read it back into a brand-new netlist**,
  and solve *that* independently. Because the re-solved voltages match the ones the
  generator already stored, this shows three things at once: the generator builds correct
  circuits, the text format survives a round trip, and the solver agrees with itself across
  two different paths into it.

- Finally it prints a one-line confirmation that the Phase-0 spine holds together.

## Where the build stands

Built and tested: the netlist (File 1), the solver (File 2), and the generator (File 3).
That's the entire back half of the conveyor belt — the hard math — proven against circuits
we can check by hand, plus a tool that manufactures unlimited labeled test circuits.

Still ahead, in order: tracing wires from a real image, training the component detector on a
real dataset, the live camera and overlay, cross-checking our solver against the
professional simulator, and finally the explain-and-critique and what-if features.

---

# Phase 1 — teaching the machine to *see* components

Phase 0 built the back half of the belt: given a circuit-as-text, we solve it. Phase 1
builds the **eyes** — the part that looks at a photo of a hand-drawn circuit and says
"resistor here, battery there." Nothing in Phase 1 runs from a clean drawing we made
ourselves; it learns from thousands of *real* hand-drawn photos.

## The idea in one breath

A photo is just a grid of coloured dots. To turn dots into "there's a resistor at this
spot," we train an **object detector** — a program that outputs labelled boxes. We use
**YOLO** (a fast, popular detector) in its smallest "nano" size, because our drawings are
simple and we want it to run on a plain laptop. Training means showing the detector
thousands of example photos where a human already drew the correct boxes, and letting it
adjust itself until its guesses match. The single file it produces, `best.pt`, *is* the
trained eyes.

## File 5 — `data_collection/cghd_prep.py` — turning a public dataset into our training set

**Where this fits:** the detector can't learn without labelled examples. A research group
published **CGHD** — thousands of photos of hand-drawn circuits with every component already
boxed by humans. But it's labelled for *their* problem (50-plus component types in a format
called VOC XML), and we only care about *our* 8. This file is the translator.

**What it does, move by move:**

- **Reads the VOC labels.** Each CGHD photo comes with an XML file listing every box and
  what it is. We read the image's width/height and every `(class, xmin, ymin, xmax, ymax)`
  straight from that XML — we never even open the image to do the label maths.

- **Remaps the 50-plus classes down to our 8.** A table at the top of the file says, e.g.,
  `capacitor.unpolarized` and `capacitor.polarized` both become just `capacitor`; both
  battery and DC-source symbols become `voltage_source`. Anything we don't support
  (inductors, transistors, logic gates, AC sources…) is simply **dropped**. Our 8 classes,
  in fixed order, are: capacitor, diode, ground, junction, resistor, switch, text,
  voltage_source — numbered 0–7.

- **Converts box format.** VOC stores boxes as corner pixels; YOLO wants centre-x, centre-y,
  width, height, each as a fraction of the image (0–1). Pure arithmetic, done per box.

- **Splits by *person*, not by photo — the important bit.** CGHD has several photos of the
  same drawing and several drawings by the same person. If we shuffled all photos and
  randomly held some back to grade ourselves on, we'd be grading the model on drawings it
  basically already saw — an inflated, dishonest score. Instead we hold back **whole
  people**: every photo from a given drafter goes entirely into train, or entirely into
  the test pile, never split. Now the test score answers the real question: *how well does
  it do on a stranger's handwriting?*

- **Writes a tidy YOLO dataset** — train/val/test folders of images and label files, plus a
  `data.yaml` that tells YOLO where everything is and names the 8 classes.

It's all standard-library Python and covered by hand-checkable tests (`tests/test_cghd_prep.py`):
the box maths, the class remap, and — most importantly — a test proving the train/val/test
people never overlap.

## Files 6 & 7 — `training/train_colab.py` and `training/evaluate.py`

**Why these run somewhere else:** training crunches thousands of images and really wants a
GPU. This Mac's GPU is awkward to use for this, so training happens on **Google Colab** —
a free website that lends you a computer with a good GPU for a couple of hours.
`training/COLAB_INSTRUCTIONS.md` is the click-by-click guide.

- **`train_colab.py`** points YOLO-nano at our `data.yaml` and lets it learn (~1–2 hours),
  producing `best.pt` — the trained detector.

- **`evaluate.py`** grades that `best.pt` on the held-out *test people* and reports a score
  called **mAP** (higher is better; it measures how well predicted boxes line up with the
  human-drawn truth). Because the test people were never trained on, this number is honest.

## Where the build stands now

Written and tested: the dataset translator and the train/evaluate scripts. What's left in
Phase 1 is the part only a human with a browser can kick off — download CGHD, run the
training on Colab, and read back the score. After that comes Phase 2: feeding *real* photos
through detection into the wire-tracing and netlist code.
