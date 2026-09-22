"""Агент качества: прогноз показателей ДТ через ВАК-формулы, риск по сере,
уверенность в прогнозе (ТЗ шаг 3).

Источник серы выбирается по приоритету ЛИМС -> ПАК (в данных пакета нет
ВАК-формулы для серы, поэтому третьей ступени приоритета здесь просто нет —
явно задокументировано, а не подразумевается).
"""
from __future__ import annotations

import pandas as pd

from ..config import (
    LIMS_STALE_AFTER_HOURS,
    PAK_STALE_AFTER_HOURS,
    SULFUR_LIMIT_MG_KG,
    SULFUR_RISK_MARGIN_FRACTION,
)
from ..data_io import asof_lookup
from ..schemas import DataQualityReport, QualityForecast
from ..vak_formulas import AVT_ANALYZERS, U242000_ANALYZERS, all_present

SULFUR_LIMS_GROUP_FRAGMENTS = ("Гидроочистка", "'2'")
SULFUR_LIMS_PARAM = "Mg.Sulfur"  # мг/кг, comparable to PAK ppm; "Mass.Sulfur" is a
# separate (likely % масс.) measurement in the same sheet — not used for the
# 10 мг/кг hard limit, explicit assumption.
SULFUR_PAK_TAG = "24-2000:Mg.Sulfur"


def _sulfur_reading(lims_long: pd.DataFrame, pak_long: pd.DataFrame, asof: pd.Timestamp):
    sub = lims_long[
        lims_long["group"].astype(str).str.contains(SULFUR_LIMS_GROUP_FRAGMENTS[0], na=False)
        & lims_long["group"].astype(str).str.contains(SULFUR_LIMS_GROUP_FRAGMENTS[1], na=False)
        & (lims_long["param"] == SULFUR_LIMS_PARAM)
    ]
    lims_res = asof_lookup(sub, asof, filters={})
    if lims_res.is_available and lims_res.age_hours <= LIMS_STALE_AFTER_HOURS:
        return lims_res.value, "ЛИМС", lims_res.age_hours

    pak_res = asof_lookup(pak_long, asof, filters={"tag": SULFUR_PAK_TAG})
    if pak_res.is_available and pak_res.age_hours <= PAK_STALE_AFTER_HOURS:
        return pak_res.value, "ПАК", pak_res.age_hours

    # both stale or missing: report the freshest available number but flag it
    best = min(
        [r for r in (lims_res, pak_res) if r.is_available],
        key=lambda r: r.age_hours,
        default=None,
    )
    if best is None:
        return None, "unavailable", None
    source = "ЛИМС" if best is lims_res else "ПАК"
    return best.value, f"{source} (устарело)", best.age_hours


def assess(
    timestamp: pd.Timestamp,
    avt_row: dict[str, float],
    u242000_row: dict[str, float],
    lims_long: pd.DataFrame,
    pak_long: pd.DataFrame,
    data_report: DataQualityReport,
) -> QualityForecast:
    predictions: dict[str, float] = {}
    for key, analyzer in AVT_ANALYZERS.items():
        if all_present(avt_row, analyzer.required_tags):
            try:
                predictions[key] = analyzer.compute(avt_row)
            except (KeyError, ZeroDivisionError):
                pass

    lims_latest: dict[str, float] = {}
    hydrotreat = lims_long[
        lims_long["group"].astype(str).str.contains("Гидроочистка", na=False)
        & lims_long["group"].astype(str).str.contains("'2'", na=False)
    ]
    for key, analyzer in U242000_ANALYZERS.items():
        ok = all_present(u242000_row, analyzer.required_tags)
        for lims_key in analyzer.lims_inputs:
            param = lims_key.rsplit(".", 1)[-1]
            res = asof_lookup(hydrotreat[hydrotreat["param"] == param], timestamp, filters={})
            if res.is_available:
                lims_latest[lims_key] = res.value
            else:
                ok = False
        if ok:
            try:
                predictions[key] = analyzer.compute(u242000_row, lims_latest)
            except (KeyError, ZeroDivisionError):
                pass

    sulfur_val, sulfur_source, sulfur_age = _sulfur_reading(lims_long, pak_long, timestamp)

    spec_risk = (
        sulfur_val is None
        or sulfur_val >= SULFUR_LIMIT_MG_KG * SULFUR_RISK_MARGIN_FRACTION
    )

    reasons = []
    confidence = 1.0
    if data_report.tag_coverage < 1.0:
        confidence -= 0.3 * (1.0 - data_report.tag_coverage)
        reasons.append(f"Полнота КИП {data_report.tag_coverage:.0%}")
    if sulfur_val is None:
        confidence -= 0.5
        reasons.append("Нет ни одного свежего измерения серы (ЛИМС/ПАК)")
    elif "устарело" in sulfur_source:
        confidence -= 0.3
        reasons.append(f"Последнее измерение серы устарело ({sulfur_age:.1f} ч)")
    confidence = max(0.0, min(1.0, confidence))

    return QualityForecast(
        timestamp=timestamp,
        predictions=predictions,
        lims_inputs_used=lims_latest,
        sulfur_ppm=sulfur_val,
        sulfur_source=sulfur_source,
        sulfur_age_hours=sulfur_age,
        spec_risk=spec_risk,
        confidence=confidence,
        confidence_reasons=reasons,
    )
