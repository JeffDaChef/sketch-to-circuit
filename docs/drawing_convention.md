# Drawing Convention (Input Spec) — v1

Extracting *arbitrary* hand-drawn circuits is an unsolved research problem. This project
sidesteps it by constraining the input, exactly like a real product defines its input
format. **Every drawing fed to the system must follow these rules:**

1. Thick **black marker** on white paper (no pencil — too faint).
2. Components drawn at **0° or 90° only**.
3. **Every junction gets a solid filled dot.**
4. **No crossing wires** — route around. (Crossover handling is a documented stretch goal.)
5. Component values written next to components, **horizontally**.
6. One circuit per page, drawn **inside the 4 ArUco markers**.

## v1 component classes (8)

resistor · capacitor · voltage source · ground · diode/LED · switch · junction dot · text label

**Excluded from v1:** inductors (AC only), anything needing crossing wires.

> Defining input constraints to make an unsolved problem tractable is itself an
> engineering decision — this spec is part of the project, not a limitation of it.
