"""Tests for solver/netlist.py — every expected value here is hand-checkable."""

import pytest

from solver.netlist import GROUND, Netlist, NetlistError, parse_value


class TestParseValue:
    def test_plain_numbers(self):
        assert parse_value("470") == 470
        assert parse_value("3.3") == 3.3
        assert parse_value(".5") == 0.5

    def test_engineering_suffixes(self):
        assert parse_value("10k") == 10_000
        assert parse_value("10K") == 10_000
        assert parse_value("1M") == 1_000_000
        assert parse_value("1MEG") == 1_000_000
        assert parse_value("5m") == pytest.approx(5e-3)
        assert parse_value("100n") == pytest.approx(100e-9)
        assert parse_value("4.7u") == pytest.approx(4.7e-6)
        assert parse_value("22p") == pytest.approx(22e-12)
        assert parse_value("1G") == 1e9

    def test_units_are_stripped(self):
        assert parse_value("5V") == 5
        assert parse_value("5v") == 5
        assert parse_value("2.2kΩ") == 2200
        assert parse_value("10kohm") == 10_000
        assert parse_value("100nF") == pytest.approx(100e-9)
        assert parse_value("2A") == 2

    def test_european_infix_notation(self):
        assert parse_value("4u7") == pytest.approx(4.7e-6)
        assert parse_value("2k2") == 2200
        assert parse_value("1M5") == 1_500_000

    def test_scientific_notation_and_signs(self):
        assert parse_value("1e6") == 1_000_000
        assert parse_value("1e+06") == 1_000_000
        assert parse_value("2.2e3") == 2200
        assert parse_value("1e-6") == pytest.approx(1e-6)
        assert parse_value("-5") == -5
        assert parse_value("-2.5e-3") == pytest.approx(-2.5e-3)

    def test_garbage_raises(self):
        for bad in ["", "abc", "10kk", "k10", "1.2.3"]:
            with pytest.raises(NetlistError):
                parse_value(bad)


class TestNetlist:
    def make_divider(self):
        """10V source driving two 1k resistors in series (a voltage divider)."""
        n = Netlist()
        n.add("V", "V1", "10", "in", GROUND)
        n.add("R", "R1", "1k", "in", "mid")
        n.add("R", "R2", "1k", "mid", GROUND)
        return n

    def test_add_and_node_names(self):
        n = self.make_divider()
        assert n.node_names() == ["in", "mid"]
        assert n.has_ground()

    def test_string_values_are_parsed(self):
        n = self.make_divider()
        r1 = next(c for c in n.components if c.name == "R1")
        assert r1.value == 1000

    def test_duplicate_names_rejected(self):
        n = self.make_divider()
        with pytest.raises(NetlistError):
            n.add("R", "R1", "5k", "a", "b")

    def test_unknown_kind_rejected(self):
        with pytest.raises(NetlistError):
            Netlist().add("X", "X1", 1, "a", "b")

    def test_nonpositive_passive_values_rejected(self):
        for kind in ("R", "C", "L"):
            with pytest.raises(NetlistError, match="positive"):
                Netlist().add(kind, f"{kind}1", 0, "a", "b")
            with pytest.raises(NetlistError, match="positive"):
                Netlist().add(kind, f"{kind}2", -1, "a", "b")

    def test_sources_may_be_zero_or_negative(self):
        Netlist().add("V", "V1", -5, "a", "0")
        Netlist().add("I", "I1", 0, "a", "0")

    def test_non_finite_values_rejected(self):
        with pytest.raises(NetlistError, match="finite"):
            Netlist().add("V", "V1", float("inf"), "a", "0")
        with pytest.raises(NetlistError, match="finite"):
            Netlist().add("R", "R1", float("nan"), "a", "b")

    def test_megohm_and_nanofarad_round_trip(self):
        n = Netlist()
        n.add("R", "R1", 1e6, "a", "b")
        n.add("C", "C1", 100e-9, "b", "0")
        again = Netlist.from_spice(n.to_spice())
        assert again.components[0].value == pytest.approx(1e6)
        assert again.components[1].value == pytest.approx(100e-9)

    def test_spice_round_trip(self):
        n = self.make_divider()
        text = n.to_spice()
        again = Netlist.from_spice(text)
        assert len(again.components) == 3
        assert again.node_names() == n.node_names()
        for orig, parsed in zip(n.components, again.components):
            assert (orig.kind, orig.name, orig.nodes) == (parsed.kind, parsed.name, parsed.nodes)
            assert orig.value == pytest.approx(parsed.value)

    def test_spice_text_looks_right(self):
        text = self.make_divider().to_spice()
        assert "V1 in 0 10" in text
        assert "R1 in mid 1000" in text
        assert text.rstrip().endswith(".end")
