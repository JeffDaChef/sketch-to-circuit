# EXPLAINED.md — what every piece of this project does, in plain English

This file is the running "textbook" of the project. Every time code gets written, a
section gets added here explaining **what was built, what every function does, and what
the big chunks of code actually do** — no jargon without a definition. Read it top to
bottom and you should understand the whole repo.

---

## Part 1 — The tools (the stuff around the code)

### The terminal
A text-based way to control your computer. Instead of clicking icons, you type commands
like `ls` ("list the files here") or `cd solver` ("go into the solver folder"). Every
command in this project's history was one of these. A few you'll use constantly:
- `ls` — list files in the current folder (`ls -a` includes hidden ones)
- `cd <folder>` — move into a folder; `cd ..` moves up one
- `pwd` — print which folder you're currently in

### Homebrew (`brew`) — and why we DIDN'T end up using it
The "app store" for developer tools on a Mac. We planned to use it, but installing it
requires an **administrator** account password, and this Mac account isn't an admin (the
admin is probably a parent's account). Instead we used **`uv`** (below). The only thing
we'll genuinely need admin help for is installing `ngspice` in Phase 4, months from now.

### uv — what we used instead
A modern, very fast Python manager that installs entirely inside your own user folder
(`~/.local/bin`), so it never needs an admin password. One command
(`uv venv --python 3.12`) made it download a standalone Python 3.12.13 AND create our
project's virtual environment. `uv pip install <libs>` then installed all our libraries
in seconds.

### Python and why we didn't use the one already on your Mac
Your Mac came with Python 3.9.6 (released 2021). It's owned by Apple's system — installing
project libraries into it can break system stuff, and several of our libraries want a
newer Python anyway. So uv downloaded us a private Python 3.12 and Apple's copy stays
untouched.

### Virtual environment (`.venv` folder)
A private copy of Python + libraries that belongs to *this project only*. When we
"activate" it, `python` and `pip` point at the project's toolbox instead of the global
one. Why: two projects on your computer might need different versions of the same
library — venvs stop them from fighting. The `.venv/` folder is disposable; you can
delete and recreate it anytime, which is also why git ignores it.

### pip
Python's library installer. `pip install numpy` downloads the numpy library into the
active environment so `import numpy` works in our code. (We run it through uv as
`uv pip install`, which does the same thing but faster.) The project's library list
lives in `requirements.txt`, so anyone can recreate the environment with one command.

### git
A save-point system for code. A **commit** is a named snapshot of every file at a moment
in time — you can always look back at (or restore) any commit. We commit after every
working chunk, so the project history shows real incremental progress. Key commands:
- `git status` — what changed since the last save-point?
- `git add -A` — stage all changes ("put them in the box")
- `git commit -m "message"` — seal the box with a description
- `git log --oneline` — list all save-points

`.gitignore` is a list of files git should *never* track: the venv (recreatable),
`__pycache__` (Python's auto-generated junk), `.DS_Store` (invisible macOS metadata
files), and generated images.

### The libraries we're installing (and why)
| Library | What it's for here |
|---|---|
| `schemdraw` | Draws circuit schematics programmatically → our synthetic test images |
| `numpy` | Fast matrix math → the heart of the MNA circuit solver |
| `networkx` | Graph algorithms → later, checking extracted circuits match ground truth |
| `matplotlib` | The rendering engine schemdraw uses to make PNGs |
| `pytest` | Test runner: finds functions named `test_*` and runs them, reporting pass/fail |

---

## Part 2 — The repo layout

| Path | What it is |
|---|---|
| `README.md` | Front page: what the project is + build status checklist |
| `EXPLAINED.md` | This file |
| `sketch_to_circuit_brief.md` | The full project spec/plan |
| `docs/drawing_convention.md` | The input rules every drawing must follow (this constraint is what makes the vision problem solvable) |
| `data_collection/` | Synthetic schematic generator; later, dataset prep |
| `training/` | (empty for now) YOLO detector training — Phase 1 |
| `vision/` | (empty for now) camera, preprocessing, wire tracing — Phases 2–3 |
| `solver/` | Circuit math: netlist data structures + MNA solver |
| `ui/` | (empty for now) live overlay — Phase 3 |
| `tests/` | Unit tests — small programs that check our code gives known-correct answers |

---

## Part 3 — The code

### `solver/netlist.py` — describing a circuit in software

**The big idea:** before we can solve or even detect circuits, we need a way to *write
one down* as data. The standard way (used by every circuit simulator since the 1970s,
including SPICE) is a **netlist**: a parts list where each part says which **nets** it
touches. A *net* is one connected blob of wire — every point on it is electrically
identical, so it gets one name. Ground is always net `"0"` (SPICE tradition).

Example — a 10V battery feeding two 1kΩ resistors in series:
```
V1 in 0 10        ← voltage source between net "in" and ground
R1 in mid 1000    ← resistor from "in" to "mid"
R2 mid 0 1000     ← resistor from "mid" to ground
```
That text IS the circuit. Drawing position, wire routing — none of it matters
electrically; only the connections do.

**Now the code itself, translated top-to-bottom.** Open `solver/netlist.py` next to
this and read them together — each block below is the same block in the file, in the
same order, said in English.

**Block 1 — the imports (top of file).**
```python
from __future__ import annotations
import re
from dataclasses import dataclass, field
```
*In English:* "Bring in three tools before we start. `from __future__ import
annotations` lets us write modern type hints like `tuple[str, str]` even on slightly
older Python. `re` is Python's regular-expression toolkit — pattern-matching on text,
which `parse_value` needs. `dataclass` and `field` are shortcuts for making
record-style objects without writing boilerplate."

**Block 2 — the constants `KINDS` and `GROUND`.**
```python
KINDS = {"R": "resistor", "C": "capacitor", "V": "voltage source", ...}
GROUND = "0"
```
*In English:* "Define the list of component types we allow, as a dictionary mapping a
one-letter code to a human name. Anything not in here will be rejected later. Also fix
the name of the ground net as the string `"0"` once, so the rest of the file can refer
to `GROUND` instead of hardcoding `"0"` everywhere."

**Block 3 — our own error type.**
```python
class NetlistError(Exception): ...
```
*In English:* "Make a custom error labelled `NetlistError`. When something is wrong with
a circuit or a value, we'll *raise* this. Having our own named error means tests and
callers can catch specifically *our* circuit errors and not get them confused with
unrelated Python errors."

**Block 4 — the multiplier table and the three regex patterns.**
```python
_MULTIPLIERS = {"k": 1e3, "M": 1e6, "m": 1e-3, "u": 1e-6, ...}
_UNIT_SUFFIX = re.compile(r"(ohms?|Ω|[VvAaFf])$")
_PLAIN  = re.compile(r"^(\d+\.?\d*|\.\d+)(MEG|meg|[TGkKmMuµnp])?$")
_INFIX  = re.compile(r"^(\d+)(MEG|meg|[TGkKmMuµnp])(\d+)$")
```
*In English:* "Set up the lookup data for reading written values. `_MULTIPLIERS` says
what each engineering letter multiplies by (`k` = ×1000, `u` = ÷1,000,000, etc.). The
three `re.compile(...)` lines pre-build text patterns we'll reuse: `_UNIT_SUFFIX`
matches a trailing unit like `V`, `A`, `F`, or `Ω` at the end of a string; `_PLAIN`
matches 'a number, optionally followed by one multiplier letter' (the `10k` shape);
`_INFIX` matches 'digits, a multiplier letter, then more digits' (the `4u7` shape,
where the letter stands in for the decimal point). The leading underscore on these
names is a convention meaning 'internal — not meant to be used outside this file.'"

**Block 5 — `parse_value(text)`, the value reader.**
```python
def parse_value(text: str) -> float:
    token = text.strip().replace(" ", "")
    if not token: raise NetlistError("empty value")
    stripped = _UNIT_SUFFIX.sub("", token)
    if stripped and (...): token = stripped
    m = _INFIX.match(token)
    if m: ... return float(f"{whole}.{frac}") * _MULTIPLIERS[...]
    m = _PLAIN.match(token)
    if m: ... return float(number) * scale
    raise NetlistError(f"can't parse component value: {text!r}")
```
*In English, step by step (this is exactly the order the code runs):*
1. "Clean up the input: remove surrounding spaces and any spaces inside. Call the
   result `token`."
2. "If there's nothing left, that's an error — stop and raise `NetlistError`."
3. "Try to chop a trailing unit off the end (`5V` → `5`). Only actually use the chopped
   version if what's left still ends in a digit or a multiplier, so we don't accidentally
   ruin a plain number."
4. "Test the `4u7` infix shape first. If it matches, glue the two digit-groups back
   together around a decimal point (`4` and `7` → `4.7`) and multiply by the letter's
   value from `_MULTIPLIERS`. Return that number."
5. "Otherwise test the plain `10k` shape. If it matches, take the number part, multiply
   by the multiplier if there is one, and return it."
6. "If neither shape matched, we don't understand the input — raise `NetlistError` with
   the original text so the message is useful." *(This same function is the safety net
   behind the handwriting-OCR module later.)*

**Block 6 — the `Component` record.**
```python
@dataclass
class Component:
    kind: str
    name: str
    value: float
    nodes: tuple[str, str]
    def __post_init__(self): ...validate...
```
*In English:* "`@dataclass` tells Python: this class is just a record holding these four
named fields, so auto-write the boring setup code for me. A `Component` holds its `kind`
(one of the `KINDS` codes), its `name` like `'R1'`, its `value` as a plain number, and
`nodes`, the pair of net names its two ends touch. The `__post_init__` method runs
automatically right after a Component is created and checks the data is sane — kind must
be known, and there must be exactly two nodes — raising `NetlistError` if not. Node
order matters: for a source, the first node is the `+` terminal."

**Block 7 — the `Netlist` container and its methods.**
```python
@dataclass
class Netlist:
    components: list[Component] = field(default_factory=list)
```
*In English:* "A `Netlist` is the whole circuit: mainly a list of `Component`s. The
`field(default_factory=list)` part means 'each new Netlist starts with its own fresh
empty list.' Then come its methods — the things a netlist can do:"
- **`add(kind, name, value, node_a, node_b)`** — *"Add one component to the circuit.
  First refuse if the name is already taken. If the value came in as a string like
  `'10k'`, run it through `parse_value` to get a number. Build the `Component`, append
  it to the list, and hand it back."* This is the method the rest of the project calls
  to build circuits.
- **`node_names()`** — *"Collect every net name mentioned by any component, throw out
  ground, and return them sorted."* The solver calls this to learn what the unknown
  voltages are.
- **`has_ground()`** — *"Return true if any component touches net `'0'`."* A circuit
  with no ground has no voltage reference, so the solver checks this first.
- **`to_spice(title)`** — *"Write the circuit out as standard SPICE text: a title
  line, then one line per component (`name nodeA nodeB value`), then `.end`."* Lets us
  save circuits and feed them to the real ngspice in Phase 4.
- **`from_spice(text)`** — *"The reverse: read SPICE text back into a Netlist. Skip
  blank lines, comment lines starting with `*`, and `.end`. For every real line, split
  it into four pieces, take the kind from the first letter of the name, and `add` it."*
  Lets us write test circuits as plain text.

### `tests/` + `conftest.py` — how we know the code works

`tests/test_netlist.py` contains 11 small functions, each named `test_...`, each
asserting that a known input gives a known output (e.g. `parse_value("2k2") == 2200`).
The `pytest` tool finds and runs them all with one command. The principle (from the
brief): every module gets tested **in isolation** so when something breaks, we know
*which* module broke. `conftest.py` at the repo root is empty plumbing — its presence
tells pytest "imports start from here," letting tests say `from solver.netlist import …`.

Run them yourself anytime: `.venv/bin/python -m pytest tests/ -v`
