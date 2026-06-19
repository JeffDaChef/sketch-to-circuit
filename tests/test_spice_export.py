"""Tests for solver/spice_export.py.

The key checks: (1) engineering-notation values round-trip through the existing
parse_value() — so what we write, real tools (and we) read back identically; and
(2) diodes get a proper `.model` line, which is what makes the deck loadable by
ngspice/KiCad rather than just textually plausible.
"""

import pytest

from solver.netlist import Netlist, parse_value
from solver.nonlinear import LED, SILICON
from solver.spice_export import export_spice, format_value


@pytest.mark.parametrize("value", [10000, 220, 9.0, 0.001, 100e-6, 1e6, 4700, 2.2e3, 1e-9])
def test_format_value_round_trips_through_parser(value):
    assert parse_value(format_value(value)) == pytest.approx(value, rel=1e-9)


def test_format_value_uses_meg_not_m_for_mega():
    assert format_value(1e6) == "1MEG"
    assert format_value(1e-3) == "1m"


def test_resistor_and_source_lines():
    n = Netlist()
    n.add("V", "V1", "5", "in", "0")
    n.add("R", "R1", "10k", "in", "0")
    deck = export_spice(n)
    assert deck.splitlines()[0].startswith("*")
    assert "V1 in 0 5" in deck
    assert "R1 in 0 10k" in deck
    assert deck.strip().endswith(".end")


def test_diode_emits_model_reference_and_model_line():
    n = Netlist()
    n.add("V", "V1", "5", "in", "0")
    n.add("R", "R1", "220", "in", "a")
    n.add("D", "D1", 0.0, "a", "0")
    deck = export_spice(n)
    assert "D1 a 0 DMODEL1" in deck
    assert ".model DMODEL1 D(Is=1e-14 N=1)" in deck


def test_identical_diode_models_are_shared():
    n = Netlist()
    n.add("V", "V1", "5", "in", "0")
    n.add("D", "D1", 0.0, "in", "mid")
    n.add("D", "D2", 0.0, "mid", "0")
    deck = export_spice(n)
    assert deck.count(".model") == 1
    assert "D1 in mid DMODEL1" in deck and "D2 mid 0 DMODEL1" in deck


def test_distinct_diode_models_get_separate_lines():
    n = Netlist()
    n.add("V", "V1", "5", "in", "0")
    n.add("R", "R1", "220", "in", "a")
    n.add("D", "Dsi", 0.0, "a", "b")
    n.add("D", "Dled", 0.0, "b", "0")
    deck = export_spice(n, models={"Dled": LED})
    assert deck.count(".model") == 2
    assert "D(Is=1e-17 N=2)" in deck


def test_analysis_directive_is_included():
    n = Netlist()
    n.add("V", "V1", "5", "in", "0")
    n.add("R", "R1", "1k", "in", "0")
    assert ".tran 10u 20m" in export_spice(n, analysis=".tran 10u 20m")
    assert ".op" in export_spice(n, analysis="op")


def test_missing_ground_is_flagged_in_the_file():
    n = Netlist()
    n.add("V", "V1", "5", "a", "b")
    n.add("R", "R1", "1k", "a", "b")
    deck = export_spice(n)
    assert "WARNING" in deck and "ground" in deck


def test_rvi_export_still_parses_with_ngspice_harness():
    from solver.netlist import GROUND
    from solver.ngspice_validation import build_deck
    n = Netlist()
    n.add("V", "V1", "5", "in", "0")
    n.add("R", "R1", "10k", "in", "mid")
    n.add("R", "R2", "4700", "mid", "0")
    deck = export_spice(n)
    assert "R2 mid 0 4.7k" in deck
    assert "print v(mid)" in build_deck(n)
