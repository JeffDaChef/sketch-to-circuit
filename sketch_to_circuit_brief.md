# Sketch-to-Circuit — Project Brief

> **How to use this file:** Paste this entire document as the FIRST message into a new
> Claude Code session, then say: *"This is my project context. Read it fully, then help
> me build [specific component]. Start with Phase 0."* You should never have to
> re-explain the project after this.

---

## 1. What this project is (one paragraph)

A computer-vision system that turns a **hand-drawn circuit schematic on paper** into a
**live, solved circuit simulation**. A downward-facing camera looks at the drawing; the
software detects the components, traces the wires, builds a netlist, solves the circuit
with a from-scratch nodal-analysis engine, and overlays the computed node voltages and
branch currents back onto the live camera image. Change a value on screen and every
voltage updates. The system also explains what the circuit does and flags mistakes using
an LLM. It is built and applied by a high-school student for college applications (EE
focus) — so the bar is: finished, demoable, defensibly rigorous, and clearly EE.

**One-sentence pitch (for the application):**
*"Built a computer-vision system that converts hand-drawn circuit schematics into live
simulations — fine-tuned an object detector on 3,000+ schematic images, implemented a
nodal-analysis solver validated against SPICE, and achieved X% end-to-end extraction
accuracy."* (Fill X in only when it's a real measured number.)

---

## 2. Hardware (deliberately minimal — no Arduino, no breadboard)

- **Camera:** phone-as-webcam (Continuity Camera on Mac / Iriun / DroidCam) is the
  preferred option — best image quality for $0. A Logitech C920/C920s (~$55, lockable
  manual focus) is the fallback if a dedicated camera is wanted. Avoid sub-$25 fixed-focus
  webcams — handwritten values blur.
- **Mount:** gooseneck phone clamp (~$12) or a stack of books. Camera looks straight down
  at the paper.
- **Lighting:** one desk lamp positioned to avoid hand-shadow on the page.
- **ArUco markers:** print 4, tape to the corners of a clipboard/board. OpenCV detects
  them natively → perfect perspective correction and overlay registration. This is the
  single most important $0 decision; it eliminates a whole class of alignment bugs.
- **Drawing tools:** thick BLACK marker (not pencil — too faint), white paper.

No microcontroller. Everything runs on the PC. The whole simulation→reality validation
story is handled in software (see §6).

---

## 3. The drawing convention (LOCK THIS BEFORE WRITING CODE)

Extracting *arbitrary* hand-drawn circuits is an unsolved research problem. We sidestep it
by constraining the input, exactly like a real product defines its input format:

- Thick black marker on white paper.
- Components drawn at 0° or 90° only.
- **Every junction gets a solid filled dot.**
- **No crossing wires in v1** — route around. (Crossover handling = documented stretch goal.)
- Component values written next to components, horizontally.
- One circuit per page, drawn inside the 4 ArUco markers.

Treat this as a spec committed to the repo. "I defined input constraints to make an
unsolved problem tractable" is itself a mature engineering decision worth describing.

**v1 component classes (8):** resistor, capacitor, voltage source, ground, diode/LED,
switch, junction dot, text label.
**Excluded from v1:** inductors (AC only), anything needing crossing wires.

---

## 4. Software architecture — 8 modules, each independently testable

Build and unit-test each module in isolation before integrating. When something breaks
you want to know *which* module broke.

1. **Capture & registration.** OpenCV grabs frames → detect ArUco markers → warp paper to
   a flat rectangle. Only run the full pipeline when the scene has been STILL for ~1s
   (frame-difference trigger), so it doesn't half-process images of a moving hand.
2. **Preprocessing.** Grayscale → adaptive threshold (handles uneven light) → small
   morphological cleanup → clean binary (black ink / white background).
3. **Component detection.** Fine-tuned YOLO-nano → bounding boxes for the 8 classes.
   (Only trained-ML module. See §5.)
4. **Wire extraction (HARDEST MODULE).** Erase detected component boxes from the binary
   image → skeletonize remaining wires to 1px paths → **prune short spurs/branches** →
   walk the skeleton to build a graph → match each wire endpoint to the nearest component
   terminal ("endpoint is 9px from R2's left pin → connects to R2.left"). The drawing
   convention is what makes terminal-matching tractable.
5. **Netlist construction.** Merge connected terminals into nets with union-find → assign
   net numbers → emit a SPICE-format netlist. Pure logic, fully unit-testable, no camera.
6. **Value reading (hybrid).** Crop text boxes → send to Claude API with a constrained
   prompt ("return JSON: component values only") → parse "10k", "4u7", "5V" into numbers.
   Always include a UI fallback: click a component, type its value (OCR will sometimes
   misread sloppy digits).
7. **Solver (BIGGEST IMPRESSIVENESS LEVER).** Implement **Modified Nodal Analysis (MNA)**
   from scratch: build the conductance matrix from the netlist, solve the linear system
   with numpy → node voltages. ~150 lines for resistors + sources. This is the actual
   algorithm inside SPICE. **Validate it against ngspice** on a test suite (this is the
   credibility centerpiece — see §6). v1 = linear DC only. Diodes/LEDs need iterative
   Newton-Raphson → stretch goal; for v1 approximate an LED as a fixed ~2V drop.
8. **Overlay UI.** Draw node voltages at junctions, current arrows on wires, values on
   components, registered onto the live feed (easy thanks to ArUco warp). Include a
   **debug view** rendering the extracted graph so errors are *visible*, not mysterious.

---

## 5. Training the detector

- **Dataset:** CGHD (Circuit Graph Hand-Drawn) — thousands of annotated smartphone photos
  of hand-drawn circuits, hundreds of thousands of bounding boxes incl. junctions and
  crossovers. On Kaggle (`johannesbayer/cghd1152`) and Hugging Face (`lowercaseonly/cghd`).
- **Prep:** script to remap CGHD's 45+ classes down to our 8, convert VOC → YOLO format,
  drop images dominated by unsupported components.
- **CRITICAL — split by DRAFTER, not randomly.** CGHD has multiple photos of the same
  drawing and multiple drawings of the same circuit by the same person. A random split
  leaks near-identical images across train/test and inflates accuracy into a lie. Hold out
  entire people for the test set. (Same disease as single-session overfitting — different
  dataset.)
- **Training:** Ultralytics YOLO v8-nano or 11-nano, 640px, on a **free Google Colab GPU**
  (~1–2 hrs). **Do NOT train on the local AMD GPU** — PyTorch-on-ROCm is a time sink with
  zero application payoff. CPU inference is fine at the few-fps rate the still-trigger
  design needs.
- **Domain adaptation (great talking point):** after the base model works, draw 30–40
  circuits in YOUR marker/handwriting, photograph with YOUR camera, annotate (Roboflow
  free tier / labelImg), fine-tune. Report before/after: "X% on strangers' drawings →
  Y% on mine after fine-tuning on 35 of my own." That's distribution shift demonstrated
  with your own data.

---

## 6. The four software-only elevation layers (the "make colleges care" tier)

No hardware. Each is independently real and defensible:

1. **From-scratch MNA solver validated against ngspice.** Your solver is the live engine;
   ngspice is an OFFLINE validation tool only (never a live dependency — install pain).
   Run both on a suite of ~40 test circuits, report agreement to numerical tolerance.
   This is the independent-confirmation credibility the (dropped) Arduino rig was for.
2. **Self-defined end-to-end extraction metric.** On N held-out drawings, fraction that
   produce a netlist electrically equivalent to ground truth, checked by graph isomorphism
   (networkx). Reporting a metric you *designed* for the task is research behavior and
   gives a hard headline number.
3. **LLM "explain & critique" mode.** Claude API call on the extracted netlist: explain
   what the circuit does, flag mistakes (LED w/ no series resistor, floating node, shorted
   source). This is the project's *purpose* framing — instant feedback on paper homework.
4. **Interactive what-if / fault-injection.** Perturb the circuit ("what if this 10k → 1k?",
   "what if this wire breaks?") and re-solve live. Proves the solver is a real programmable
   engine, and it's the strongest demo-video beat: change a value, watch every voltage
   cascade.

**Competitions / venues:** Congressional App Challenge (per-district, runs ~summer→late
fall — verify your district's dates; software-app focus fits perfectly and "winner/
finalist" is a named application line). Bay Area county→state science fairs (the defined
metric + ngspice validation is the rigor they score).

---

## 7. Build order (this sequence keeps you NEVER blocked)

At every stage you have something that demonstrably works.

- **Phase 0 — synthetic pipeline (no camera, no training).** Use `schemdraw` to render
  synthetic schematics with random jitter/rotation. Build & test modules 4–8 against
  perfect synthetic images on day one.
- **Phase 1 — detector.** Prep CGHD, train YOLO-nano on Colab (drafter split), evaluate.
- **Phase 2 — real photos.** Feed real hand-drawn photos through detection → wire
  extraction → netlist. Debug terminal-matching here.
- **Phase 3 — live camera.** ArUco capture + still-trigger + overlay on live feed.
- **Phase 4 — elevation layers.** ngspice validation suite, end-to-end metric,
  explain/critique mode, what-if perturbation.
- **Phase 5 — domain adaptation.** Draw/annotate your own 30–40, fine-tune, report deltas.
- **Phase 6 — polish.** Repo (with the §3 spec, wiring photo, metrics table), 60–90s
  one-take demo video (draw → simulate → change a value → watch it cascade).

---

## 8. Known roadblocks (pre-solved)

- **Wire extraction is the danger phase.** If terminal-matching is still flaky two weeks
  in, TIGHTEN the drawing convention — don't rewrite the algorithm.
- **Webcam autofocus hunting** ruins frames mid-demo → lock focus (C920) or use phone;
  fix camera height permanently once focused.
- **Skeleton spurs/artifacts at junctions** → prune branches below a length threshold.
- **Resistor vs. inductor squiggle confusion** → inductors excluded from v1 anyway.
- **ngspice install annoyance** → that's *why* your own MNA solver is primary and ngspice
  is offline-only validation.
- **General-extraction research is hard (published mAP can be low across all 45 classes)**
  → you are NOT solving the general problem; the drawing convention + 8 classes is what
  makes it tractable.
- **OCR misreads** → UI click-to-type fallback for every value.

---

## 9. What Claude Code does vs. what you do

**Claude Code (~90%):** all module code; CGHD prep/conversion scripts; YOLO training
script; the MNA solver; ngspice validation harness; union-find netlist builder; graph-
isomorphism metric; LLM prompt plumbing; overlay/UI; synthetic-data generator; debugging.

**You (the irreplaceable ~10%):** mount the camera, draw the circuits, photograph and
annotate your own 30–40 for fine-tuning, judge whether results look right, record the
demo. Claude Code can't see your paper or hold your marker.

---

### First thing to ask Claude Code
*"Start with Phase 0: set up the repo structure (data_collection/, training/, vision/,
solver/, ui/), then write the schemdraw synthetic-schematic generator and a stub netlist
so we can build the MNA solver against synthetic data before any camera or training."*


Notes: I do not know anything about Claude Code, so I will probably be very confused for a lot of things. However, since it is still a project, I want to actually know what is going on, so I am going to want a lot of explinations of what you did, and help on what to actually do for each part since I am new to coding, terminal, and Claude Code and am already making projects.