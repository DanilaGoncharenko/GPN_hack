import math

from oilhack.vak_formulas import ALL_ANALYZERS, all_present


def test_avt_density_formula_matches_manual_computation():
    analyzer = ALL_ANALYZERS["AVT6:240-350:D15"]
    row = {"F30": 120.0, "F32": 80.0, "T66": 250.0, "T33": 140.0}
    expected = 791.22872 - 5.30294 * (120.0 / (80.0 + 120.0)) + 0.52755 * 250.0 - 0.15629 * 140.0
    assert math.isclose(analyzer.compute(row), expected, rel_tol=1e-9)


def test_godt_formula_uses_lims_input():
    analyzer = ALL_ANALYZERS["24-2000:GODT:D15"]
    row = {"F22": 100.0, "T11": 200.0}
    lims_latest = {"LIMS:24-2000.Pipeline.D15": 840.0}
    expected = 667.881 + 0.15417 * 840.0 + 0.00005 * 100.0 + 0.10774 * 200.0
    assert math.isclose(analyzer.compute(row, lims_latest), expected, rel_tol=1e-9)


def test_all_present_detects_missing_and_nan():
    assert all_present({"A": 1.0, "B": 2.0}, ("A", "B"))
    assert not all_present({"A": 1.0}, ("A", "B"))
    assert not all_present({"A": 1.0, "B": float("nan")}, ("A", "B"))


def test_division_by_zero_is_nan_not_crash():
    analyzer = ALL_ANALYZERS["AVT6:240-350:D15"]
    row = {"F30": 0.0, "F32": 0.0, "T66": 1.0, "T33": 1.0}
    result = analyzer.compute(row)
    assert math.isnan(result)
