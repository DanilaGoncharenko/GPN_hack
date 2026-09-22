"""Virtual quality analyzers (ВАК) — transcribed verbatim from
`data/raw/tags_reference.xlsx` (sheet "ВАК"). These are the organizers'
own linear regressions tying KIP tags (and, for two 24-2000 formulas, the
latest known LIMS value) to lab quality metrics. Quality Agent uses them
directly instead of training a new model — they are given ground truth for
this prototype, not something we are free to re-fit.

Источник истины — `формулы_ВАК.xlsx` от организаторов (9 формул ЭЛОУ-АВТ-6 и
8 формул ЛЧ-24-2000), присланный отдельно от первого пакета. Первая версия
справочника содержала ошибки (F30 вместо F65, слипшиеся коэффициенты
AVT6:350:T50, F15 без деления на 2000 и др.); здесь всё сверено с примерами
расчёта из этого файла — они закреплены тестами в `tests/test_vak_formulas.py`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping


def all_present(row: Mapping[str, float], required_tags: tuple[str, ...]) -> bool:
    """True if every tag is in `row` and not NaN (sentinel-masked upstream)."""
    for t in required_tags:
        if t not in row:
            return False
        v = row[t]
        if isinstance(v, float) and math.isnan(v):
            return False
    return True


@dataclass(frozen=True)
class VirtualAnalyzer:
    key: str
    dataset: str  # "avt" | "242000"
    unit: str
    required_tags: tuple[str, ...]
    fn: Callable[[Mapping[str, float]], float]
    lims_inputs: tuple[str, ...] = ()  # keys into a lims-latest-values dict
    note: str = ""

    def compute(self, row: Mapping[str, float], lims_latest: Mapping[str, float] | None = None) -> float:
        values = dict(row)
        if self.lims_inputs:
            lims_latest = lims_latest or {}
            for key in self.lims_inputs:
                values[key] = lims_latest[key]
        return float(self.fn(values))


def _safe_div(a: float, b: float) -> float:
    return a / b if b else float("nan")


AVT_ANALYZERS: dict[str, VirtualAnalyzer] = {
    a.key: a
    for a in [
        VirtualAnalyzer(
            "AVT6:240-350:D15", "avt", "кг/м3", ("F65", "F32", "F30", "T66", "T33"),
            lambda r: 791.22872 - 5.30294 * _safe_div(r["F65"], r["F32"] + r["F30"])
            + 0.52755 * r["T66"] - 0.15629 * r["T33"],
        ),
        VirtualAnalyzer(
            "AVT6:240-350:T50", "avt", "°C", ("F7", "F30", "F34", "F45", "F59", "F63"),
            lambda r: 283.177 - 0.01685 * r["F7"] + 0.06248 * r["F30"] + 0.22048 * r["F34"]
            - 0.25816 * r["F45"] - 0.12159 * r["F59"] + 0.01221 * r["F63"],
        ),
        VirtualAnalyzer(
            "AVT6:240-350:EBP", "avt", "°C", ("F30", "T33", "F36", "T37", "T40", "T58"),
            lambda r: 813.883 + 2.66463 * r["F30"] - 0.20239 * r["T33"] - 3.65888 * r["F36"]
            - 14.08235 * r["T37"] - 1.32603 * r["T40"] + 14.60206 * r["T58"],
        ),
        VirtualAnalyzer(
            "AVT6:240-350:CFPP", "avt", "°C", ("T33", "P67", "P4", "F65", "F32", "F30"),
            lambda r: 31.40363 - 0.06784 * r["T33"] + 17.411 * r["P67"] - 8.11544 * r["P4"]
            - 0.47309 * _safe_div(r["F65"], r["F32"] + r["F30"]),
        ),
        VirtualAnalyzer(
            "AVT6:350:T50", "avt", "°C", ("T42", "T48", "F31", "F57", "T66", "T33"),
            lambda r: 493.6798 + 1.281193 * r["T42"] - 0.955342 * r["T48"]
            - 0.018454 * r["F31"] + 0.265904 * r["F57"] - 0.082047 * r["T66"]
            - 0.545083 * r["T33"],
        ),
        VirtualAnalyzer(
            "AVT6:350:I350", "avt", "% об.", ("L43", "T6", "T18", "F64", "T15", "T11"),
            lambda r: 39.562 - 1.62865 * r["L43"] + 0.76664 * r["T6"] - 0.22361 * r["T18"]
            + 0.00031 * r["F64"] * (r["T15"] - r["T11"]),
        ),
        VirtualAnalyzer(
            "AVT6:350:D15", "avt", "кг/м3", ("T42", "T48", "F31", "F57"),
            lambda r: 983.092 + 0.27467 * r["T42"] - 0.49014 * r["T48"]
            - 0.32983 * _safe_div(r["F31"], r["F57"]),
        ),
        VirtualAnalyzer(
            "AVT6:350:CFPP", "avt", "°C", ("T48", "T40", "F31", "F57"),
            lambda r: 19.27111 - 0.10582 * r["T48"] + 0.13836 * r["T40"]
            - 0.42304 * _safe_div(r["F31"], r["F57"]),
        ),
        VirtualAnalyzer(
            "AVT6:350-500:ViscosityK", "avt", "мм2/с",
            ("T6", "T13", "T18", "T20", "L43", "T48", "P50", "F53", "P51", "F59", "T61"),
            lambda r: 5.831 + 0.00976 * r["T6"] + 0.01188 * r["T13"] + 0.00224 * r["T18"]
            + 0.01905 * r["T20"] + 0.00794 * r["L43"] - 0.02496 * r["T48"] - 0.00008 * r["P50"]
            - 0.00882 * r["F53"] - 0.00394 * r["P51"] - 0.00255 * r["F59"] + 0.01153 * r["T61"],
        ),
    ]
}

U242000_ANALYZERS: dict[str, VirtualAnalyzer] = {
    a.key: a
    for a in [
        VirtualAnalyzer(
            "24-2000:GODT:T90", "242000", "°C", ("T12", "F15", "W7", "T23", "F1", "F26"),
            lambda r: 162.998 + 0.12945 * r["T12"] + 59.57 * (r["F15"] / 2000.0) + 0.00036 * r["W7"]
            + 0.26366 * r["T23"] - 424.72638 * _safe_div(r["F1"], r["F26"]),
        ),
        VirtualAnalyzer(
            "24-2000:GODT:T50", "242000", "°C", ("P13", "F9", "T6"),
            lambda r: 44.625 + 10.0224 * r["P13"] + 0.06981 * r["F9"] + 0.471 * r["T6"],
        ),
        VirtualAnalyzer(
            "24-2000:GODT:I250", "242000", "% об.", ("T5", "T11", "F25", "F14", "T23", "T16"),
            lambda r: 84.585 - 0.21172 * r["T5"] + 0.12137 * r["T11"] - 0.00014 * r["F25"]
            + 0.56248 * r["F14"] - 0.16317 * r["T23"] + 0.20272 * r["T16"],
        ),
        VirtualAnalyzer(
            "24-2000:GODT:D15", "242000", "кг/м3", ("F22", "T11"),
            lambda r: 667.881 + 0.15417 * r["LIMS:24-2000.Pipeline.D15"] + 0.00005 * r["F22"]
            + 0.10774 * r["T11"],
            lims_inputs=("LIMS:24-2000.Pipeline.D15",),
            note="Требует последнего известного значения ЛИМС по продукту "
                 "'Гидроочистка, точка отбора 2' (допущение маппинга).",
        ),
        VirtualAnalyzer(
            "24-2000:GODT:CloudPoint", "242000", "°C", ("W7", "F25", "F1", "T6", "F9", "T16", "F22"),
            lambda r: 0.0002 * r["F22"] + 0.0021 * r["W7"] + 0.00008 * r["F25"] - 0.30656 * r["F1"]
            + 0.12018 * r["T6"] + 0.01916 * r["F9"] - 48.254 - 0.05249 * r["T16"] + 0.00011,
        ),
        VirtualAnalyzer(
            "24-2000:GODT:T95", "242000", "°C", ("F9", "F2", "T6"),
            lambda r: 0.03814 * r["F9"] - 9.201 - 0.00002 * r["F2"] + 0.50 * r["T6"]
            + 0.48321 * r["LIMS:24-2000.Pipeline.95%.T"],
            lims_inputs=("LIMS:24-2000.Pipeline.95%.T",),
            note="Требует последнего известного значения ЛИМС 95%.T по тому же "
                 "продукту (допущение маппинга).",
        ),
        VirtualAnalyzer(
            "24-2000:GODT:CFPP", "242000", "°C", ("T23", "P8", "F9", "W7", "P24"),
            lambda r: 0.22088 * r["T23"] - 102.375 - 47.75834 * r["P8"] + 0.03862 * r["F9"]
            + 43.60207 * r["W7"] + 43.81849 * r["P24"],
        ),
        VirtualAnalyzer(
            "24-2000:GODT:IBP", "242000", "°C",
            ("F26", "F22", "P13", "P24", "F14", "W4", "T23", "T16"),
            lambda r: 137.762 - 0.0653 * r["F26"] + 0.00011 * r["F22"] + 5.78137 * r["P13"]
            - 34.58028 * r["P24"] - 0.00993 * r["F14"] - 0.99962 * r["W4"] + 0.32232 * r["T23"]
            - 0.09406 * r["T16"],
        ),
    ]
}

ALL_ANALYZERS: dict[str, VirtualAnalyzer] = {**AVT_ANALYZERS, **U242000_ANALYZERS}

# The one hard-constrained analyzer this prototype tracks end-to-end for the
# sulfur limit (ТЗ: "содержание серы... не более 10 мг/кг"). Sulfur itself is
# measured directly by ПАК (`W7` / `24-2000:Mg.Sulfur`), not via a ВАК
# regression — Quality Agent reads it straight from the PAK feed when fresh.
SULFUR_PAK_TAG = "24-2000:Mg.Sulfur"
DENSITY_PAK_TAG = "24-2000:D15"
