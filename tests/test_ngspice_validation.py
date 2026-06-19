"""Tests for solver/ngspice_validation.py.

The point of this harness is to compare our solver against ngspice — but ngspice
needs a one-time admin install and may be absent. So the tests come in two tiers:

  * Tier 1 (always runs): deck-building and output-parsing are pure text work, so
    we test them by feeding *canned* ngspice output — no ngspice required. The
    `compare()` logic is exercised the same way, with a fake runner injected.
  * Tier 2 (skipped unless ngspice is installed): actually shells out to ngspice
    and checks our solver agrees with it on the hand-checked suite.
"""

import pytest

from data_collection.synthetic import TEMPLATES
from solver.mna import solve
from solver.netlist import Netlist
from solver.ngspice_validation import (
    NgspiceError,
    build_deck,
    compare,
    default_suite,
    ngspice_available,
    parse_ngspice_output,
    validate_suite,
)


def divider() -> Netlist:
    """The equal-resistor divider: V(in)=10, V(mid)=5, I(V1)=-5mA. Hand-checked."""
    n = Netlist()
    n.add("V", "V1", "10", "in", "0")
    n.add("R", "R1", "1k", "in", "mid")
    n.add("R", "R2", "1k", "mid", "0")
    return n


CANNED_DIVIDER_OUTPUT = """\
Note: No compatibility mode selected!


Circuit: * sketch-to-circuit validation

Doing analysis at TEMP = 27.000000 and TNOM = 27.000000

No. of Data Rows : 1
v(in) = 1.000000e+01
v(mid) = 5.000000e+00
i(v1) = -5.000000e-03
"""



def test_deck_has_components_and_control_block():
    deck = build_deck(divider(), title="my divider")
    assert deck.splitlines()[0] == "* my divider"
    assert "V1 in 0 10" in deck
    assert "R1 in mid 1000" in deck
    assert ".control" in deck and "op" in deck and ".endc" in deck
    assert deck.strip().endswith(".end")


def test_deck_prints_each_nonground_node_and_source_current():
    deck = build_deck(divider())
    assert "print v(in)" in deck
    assert "print v(mid)" in deck
    assert "print v(0)" not in deck
    assert "print i(V1)" in deck


def test_deck_rejects_diodes():
    n = Netlist()
    n.add("V", "V1", "5", "a", "0")
    n.add("D", "D1", "0.7", "a", "0")
    with pytest.raises(NgspiceError, match="linear DC subset"):
        build_deck(n)


def test_deck_requires_ground():
    n = Netlist()
    n.add("V", "V1", "5", "a", "b")
    n.add("R", "R1", "1k", "a", "b")
    with pytest.raises(NgspiceError, match="ground"):
        build_deck(n)



def test_parse_reads_voltages_and_currents():
    v, i = parse_ngspice_output(CANNED_DIVIDER_OUTPUT, divider())
    assert v == {"in": pytest.approx(10.0), "mid": pytest.approx(5.0)}
    assert i == {"V1": pytest.approx(-5e-3)}


def test_parse_is_case_insensitive_to_ngspice_lowercasing():
    n = Netlist()
    n.add("V", "Vsrc", "5", "TOP", "0")
    n.add("R", "R1", "1k", "TOP", "0")
    text = "v(top) = 5.000000e+00\ni(vsrc) = -5.000000e-03\n"
    v, i = parse_ngspice_output(text, n)
    assert v["TOP"] == pytest.approx(5.0)
    assert i["Vsrc"] == pytest.approx(-5e-3)


def test_parse_accepts_branch_vector_form():
    text = "v(in) = 1.000000e+01\nv(mid) = 5.000000e+00\nv1#branch = -5.000000e-03\n"
    v, i = parse_ngspice_output(text, divider())
    assert i["V1"] == pytest.approx(-5e-3)


def test_parse_ignores_banner_and_noise_lines():
    noisy = "Total elapsed time: 0.001 seconds.\n" + CANNED_DIVIDER_OUTPUT + "\nbye\n"
    v, _ = parse_ngspice_output(noisy, divider())
    assert v["in"] == pytest.approx(10.0)



def test_compare_agrees_when_ngspice_matches_solver():
    report = compare(divider(), "divider", runner=lambda deck: CANNED_DIVIDER_OUTPUT)
    assert report.agrees
    assert report.max_voltage_error < 1e-6
    assert report.max_current_error < 1e-9


def test_compare_flags_disagreement():
    bad = CANNED_DIVIDER_OUTPUT.replace("v(mid) = 5.000000e+00", "v(mid) = 9.000000e+00")
    report = compare(divider(), "divider", runner=lambda deck: bad)
    assert not report.agrees
    assert report.voltage_errors["mid"] == pytest.approx(4.0)


def test_compare_raises_if_node_missing_from_output():
    missing = "v(in) = 1.000000e+01\ni(v1) = -5.000000e-03\n"
    with pytest.raises(NgspiceError, match="missing node voltage"):
        compare(divider(), "divider", runner=lambda deck: missing)


def test_validate_suite_runs_all_cases():
    reports = validate_suite([("divider", divider())], runner=lambda deck: CANNED_DIVIDER_OUTPUT)
    assert len(reports) == 1 and reports[0].agrees



def _fake_ngspice_from_solver(netlist):
    """A runner that prints OUR solver's answer in ngspice's format.

    Lets us exercise build_deck + parse + compare on real netlists without
    ngspice installed. It's not a correctness check (it can't disagree with
    itself) — it's a wiring check: every node/source the deck asks about must
    survive the round trip.
    """
    r = solve(netlist)
    def run(_deck):
        lines = [f"v({n}) = {r.node_voltages[n]:.6e}" for n in netlist.node_names()]
        lines += [f"i({name}) = {cur:.6e}" for name, cur in r.source_currents.items()]
        return "Note: banner line\n\n" + "\n".join(lines) + "\nbye\n"
    return run


@pytest.mark.parametrize("template", TEMPLATES, ids=lambda t: t.__name__.lstrip("_"))
def test_harness_handles_real_generator_netlists(template):
    import random
    _, netlist, _ = template(random.Random(7))
    deck = build_deck(netlist, template.__name__)
    for node in netlist.node_names():
        assert f"print v({node})" in deck
    report = compare(netlist, template.__name__, runner=_fake_ngspice_from_solver(netlist))
    assert report.agrees, f"plumbing mismatch on {template.__name__}:\n{report}"



@pytest.mark.skipif(not ngspice_available(), reason="ngspice not installed")
def test_live_solver_agrees_with_ngspice_on_full_suite():
    reports = validate_suite(default_suite())
    for r in reports:
        assert r.agrees, f"solver disagrees with ngspice on {r.title}:\n{r}"
