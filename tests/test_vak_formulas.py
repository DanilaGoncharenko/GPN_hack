"""Формулы ВАК проверяются примерами расчёта из `формулы_ВАК.xlsx`.

Организаторы приложили к каждой формуле численный пример: входные значения
тегов и ожидаемый результат. Эти примеры — единственный внешний эталон,
который у нас есть, поэтому они зафиксированы здесь как регрессионные тесты.
Допуск 0.2 выбран потому, что в примерах промежуточные шаги округлены
(например, P8=0.117 записан как 0.12).
"""
import math

import pytest

from oilhack.vak_formulas import ALL_ANALYZERS, all_present

TOL = 0.2

AVT_CASES = [
    ("AVT6:240-350:D15",
     {"F65": 789.54, "F32": 79.00, "F30": 98.97, "T66": 255.04, "T33": 335.41}, 849.83),
    ("AVT6:240-350:T50",
     {"F7": 217.54, "F30": 98.97, "F34": 60.02, "F45": 16.45, "F59": 196.85, "F63": 88.16}, 271.82),
    ("AVT6:240-350:CFPP",
     {"T33": 335.41, "P67": 1.10, "P4": 3.85, "F65": 789.54, "F32": 79.00, "F30": 98.97}, -5.53),
    ("AVT6:240-350:EBP",
     {"F30": 98.97, "T33": 335.41, "F36": 139.29, "T37": 68.18, "T40": 180.86, "T58": 66.04}, 264.42),
    ("AVT6:350-500:ViscosityK",
     {"T6": 234.73, "T13": 89.20, "T18": 158.40, "T20": 113.80, "L43": 60.08, "T48": 354.43,
      "P50": 0.96, "F53": 80.13, "P51": 42.90, "F59": 196.85, "T61": 193.06}, 4.18),
    ("AVT6:350:CFPP",
     {"T48": 354.43, "T40": 180.86, "F31": 500.83, "F57": 23.84}, -2.10),
    ("AVT6:350:T50",
     {"T42": 275.79, "T48": 355.21, "F31": 498.33, "F57": 20.53, "T66": 256.08, "T33": 336.20}, 299.67),
    ("AVT6:350:I350",
     {"L43": 60.08, "T6": 234.73, "T18": 158.40, "F64": 104.56, "T15": 135.58, "T11": 56.14}, 88.82),
    ("AVT6:350:D15",
     {"T42": 275.99, "T48": 354.43, "F31": 500.83, "F57": 23.84}, 878.25),
]

GODT_CASES = [
    ("24-2000:GODT:T90",
     {"T12": 175.10, "F15": 2787.38, "W7": 0.143, "T23": 234.20, "F1": 2.613, "F26": 201.23},
     {}, 324.92),
    ("24-2000:GODT:T50",
     {"P13": 3.759, "F9": 171.09, "T6": 360.13}, {}, 263.86),
    ("24-2000:GODT:I250",
     {"T5": 365.14, "T11": 363.60, "F25": 13242.36, "F14": 5.824, "T23": 234.20, "T16": 30.011},
     {}, 20.70),
    ("24-2000:GODT:D15",
     {"F22": 10507.55, "T11": 363.60},
     {"LIMS:24-2000.Pipeline.D15": 840.0}, 837.08),
    ("24-2000:GODT:CloudPoint",
     {"F22": 10507.55, "W7": 0.143, "F25": 13242.36, "F1": 2.613, "T6": 360.13,
      "F9": 171.09, "T16": 30.011}, {}, -0.91),
    ("24-2000:GODT:CFPP",
     {"T23": 234.20, "P8": 0.117, "F9": 171.09, "W7": 0.143, "P24": 0.595}, {}, -17.33),
    ("24-2000:GODT:T95",
     {"F9": 171.09, "F2": 88476.80, "T6": 360.13},
     {"LIMS:24-2000.Pipeline.95%.T": 347.5}, 343.54),
    ("24-2000:GODT:IBP",
     {"F26": 201.23, "F22": 10507.55, "P13": 3.759, "P24": 0.595, "F14": 5.824,
      "W4": 1.818, "T23": 234.20, "T16": 30.011}, {}, 197.73),
]


@pytest.mark.parametrize("key,row,expected", AVT_CASES)
def test_avt_formula_matches_organizers_example(key, row, expected):
    got = ALL_ANALYZERS[key].compute(row)
    assert got == pytest.approx(expected, abs=TOL), f"{key}: получили {got}, ждали {expected}"


@pytest.mark.parametrize("key,row,lims,expected", GODT_CASES)
def test_godt_formula_matches_organizers_example(key, row, lims, expected):
    got = ALL_ANALYZERS[key].compute(row, lims)
    assert got == pytest.approx(expected, abs=TOL), f"{key}: получили {got}, ждали {expected}"


def test_every_analyzer_is_covered_by_an_example():
    covered = {c[0] for c in AVT_CASES} | {c[0] for c in GODT_CASES}
    assert covered == set(ALL_ANALYZERS), "не у всех формул есть эталонный пример"


def test_all_present_detects_missing_and_nan():
    assert all_present({"A": 1.0, "B": 2.0}, ("A", "B"))
    assert not all_present({"A": 1.0}, ("A", "B"))
    assert not all_present({"A": 1.0, "B": float("nan")}, ("A", "B"))


def test_division_by_zero_is_nan_not_crash():
    row = {"F65": 1.0, "F30": 0.0, "F32": 0.0, "T66": 1.0, "T33": 1.0}
    assert math.isnan(ALL_ANALYZERS["AVT6:240-350:D15"].compute(row))
