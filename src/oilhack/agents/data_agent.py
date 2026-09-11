"""Agent данных: полнота/актуальность/согласованность входа (ТЗ шаг 1-2).

Не решает, можно ли выдавать рекомендацию — только сообщает факты.
Приоритет источников (ЛИМС -> ПАК -> ВАК) применяется здесь один раз через
`SOURCE_PRIORITY`, чтобы Quality Agent не решал это неявно.
"""
from __future__ import annotations

import math

import pandas as pd

from ..config import MIN_TAG_COVERAGE, SENTINEL_BAD_VALUE
from ..data_io import asof_lookup
from ..history_stats import TagStats
from ..schemas import DataQualityReport

ANOMALY_Z_THRESHOLD = 4.0

SOURCE_PRIORITY = ("ЛИМС", "ПАК", "ВАК")

# LIMS groups this prototype actively tracks freshness for (used by GODT ВАК
# formulas and by the sulfur spec check as a LIMS cross-check).
TRACKED_LIMS_GROUPS = {
    "hydrotreat_pipeline": ("Гидроочистка", "'2'"),
}


def _row_to_report_fields(row: dict[str, float], stats: dict[str, TagStats]) -> tuple[float, list[str], list[str]]:
    total = len(stats)
    missing, anomalous = [], []
    present = 0
    for tag, tag_stats in stats.items():
        val = row.get(tag)
        if val is None or (isinstance(val, float) and math.isnan(val)) or val == SENTINEL_BAD_VALUE:
            missing.append(tag)
            continue
        present += 1
        if tag_stats.std > 0:
            z = abs(val - tag_stats.mean) / tag_stats.std
            if z > ANOMALY_Z_THRESHOLD:
                anomalous.append(tag)
    coverage = present / total if total else 0.0
    return coverage, missing, anomalous


def assess(
    timestamp: pd.Timestamp,
    avt_row: dict[str, float],
    u242000_row: dict[str, float],
    avt_stats: dict[str, TagStats],
    u242000_stats: dict[str, TagStats],
    lims_long: pd.DataFrame,
    pak_long: pd.DataFrame,
) -> DataQualityReport:
    cov_a, miss_a, anom_a = _row_to_report_fields(avt_row, avt_stats)
    cov_h, miss_h, anom_h = _row_to_report_fields(u242000_row, u242000_stats)
    n_a, n_h = len(avt_stats), len(u242000_stats)
    coverage = (cov_a * n_a + cov_h * n_h) / (n_a + n_h) if (n_a + n_h) else 0.0

    dead_tags = [t for t, s in {**avt_stats, **u242000_stats}.items() if s.dead]

    lims_age: dict[str, float | None] = {}
    for key, (frag_a, frag_b) in TRACKED_LIMS_GROUPS.items():
        sub = lims_long[lims_long["group"].astype(str).str.contains(frag_a, na=False)
                         & lims_long["group"].astype(str).str.contains(frag_b, na=False)]
        res = asof_lookup(sub, timestamp, filters={})
        lims_age[key] = res.age_hours

    pak_age: dict[str, float | None] = {}
    for tag in pak_long["tag"].unique():
        res = asof_lookup(pak_long, timestamp, filters={"tag": tag})
        pak_age[tag] = res.age_hours

    reasons = []
    if coverage < MIN_TAG_COVERAGE:
        reasons.append(f"Полнота КИП {coverage:.0%} ниже порога {MIN_TAG_COVERAGE:.0%}")
    is_usable = coverage >= MIN_TAG_COVERAGE

    return DataQualityReport(
        timestamp=timestamp,
        tag_coverage=coverage,
        missing_tags=miss_a + miss_h,
        anomalous_tags=anom_a + anom_h,
        dead_tags=dead_tags,
        lims_age_hours=lims_age,
        pak_age_hours=pak_age,
        is_usable=is_usable,
        reasons=reasons,
    )
