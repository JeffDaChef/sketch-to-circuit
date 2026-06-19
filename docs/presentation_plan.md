# Presentation & sharing plan (future)

A forward-looking note (not a to-do for today) on how to *present* this project for
college applications and competitions. Captured so it isn't lost.

## The key reframe: most target schools don't do interviews

That changes the strategy in one important way: **if no one will ask you questions,
your artifacts have to do all the talking.** The repo, a project page, the writeup,
and an honest demo *become* the interview. So the "can you defend it?" moat doesn't
vanish — it moves from a whiteboard into your **writing**. Net effect: presentation
and the writeup matter *more*, not less.

(Caveat: competitions — Congressional App Challenge, science fairs — often *do*
involve judging and sometimes live Q&A, and scholarships can too. So being able to
explain the work still pays off there. And you should understand your own project
regardless.)

## What's good presentation, ranked by payoff

1. **The mini-paper / writeup (the centerpiece).** 2–4 pages: problem, why the input
   was constrained, method, a results table (extraction accuracy + difficulty curve,
   noise-robustness, the 40%→100% ablation, solver-vs-analytic / ngspice), and an
   honest limitations section. For a no-interview application this single artifact
   carries depth, defensible understanding, and honesty all at once.
2. **GitHub repo.** Home base; shows you ship real, tested, reproducible code
   (`reproduce.sh`, 200+ tests, EXPLAINED.md). Low effort. Note: public = indexed
   forever, so do a once-over before pushing.
3. **Project website (showcase).** Let a stranger *experience* it: the story plus the
   figures you already generate — RC/RLC curves, Bode plots, the rectifier, the
   extractor before/after, the difficulty curve. GitHub Pages is free.
4. **Honest in-browser live demo.** The solver + wire-tracer are pure Python
   (numpy/scipy/skimage/networkx) and run in the browser via Pyodide. So a real demo
   is achievable: click → generate a circuit → watch the *actual code* trace, solve,
   and plot it. Real "wow," zero faking.
5. **Camera demo + recorded video — ONLY after the detector is trained.** A
   point-camera-and-it-works demo needs (a) the trained component detector (not built
   yet) and (b) real-photo robustness (the noise study predicts speckle/lighting will
   break the current pipeline). Build it as the *reward* for the detector step.

## The one non-negotiable: never fake or stage a demo

The whole project's edge is honesty (honest measurement, reported failure modes). A
staged "it works!" video is the opposite and the single biggest risk — especially
where a judge might say "show me on this circuit I just drew." If the system only
works on synthetic input, the demo shows synthetic input and says so plainly.

## The keystone

Training the detector (CGHD + Colab) unlocks BOTH the real camera demo AND the
validation of the core "reads hand-drawn circuits" claim. Every flashy presentation
idea routes through it — and so does the #1 moat (your own ~35 drawings + the
before/after domain-adaptation number). Presentation is the wrapper; that is the work.

## Sequencing suggestion

Writeup + GitHub + a static project site can all happen now with what exists (honest,
synthetic-validated). The live in-browser demo is a nice mid-term add. The camera demo
and its video come *after* the detector. Don't let polish substitute for the keystone.
