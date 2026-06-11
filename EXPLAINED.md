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

### Homebrew (`brew`)
The "app store" for developer tools on a Mac, run from the terminal. Apple doesn't ship
modern versions of programming tools, so almost every Mac developer installs Homebrew
first, then uses `brew install <thing>` to get everything else.

### Python and why we didn't use the one already on your Mac
Your Mac came with Python 3.9.6 (released 2021). It's owned by Apple's system — installing
project libraries into it can break system stuff, and several of our libraries want a
newer Python anyway. So we install Python 3.12 with Homebrew and leave Apple's copy alone.

### Virtual environment (`.venv` folder)
A private copy of Python + libraries that belongs to *this project only*. When we
"activate" it, `python` and `pip` point at the project's toolbox instead of the global
one. Why: two projects on your computer might need different versions of the same
library — venvs stop them from fighting. The `.venv/` folder is disposable; you can
delete and recreate it anytime, which is also why git ignores it.

### pip
Python's library installer. `pip install numpy` downloads the numpy library into the
active environment so `import numpy` works in our code.

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

*(Sections get added here as each module is built.)*
