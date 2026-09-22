"""Агент надёжности: прокси-индекс тяжести режима (ТЗ шаг 4).

Нет прямой разметки отказов/деградации катализатора в пакете данных, поэтому
индекс — явная прокси-метрика из двух компонент:
  1) отклонение текущих значений КИП от исторической рабочей области (z-score);
  2) скорость изменения ключевых тегов между соседними тиками (резкое
     переключение режима само по себе — риск для оборудования).
Оба допущения зафиксированы здесь, а не спрятаны в коде агента оптимизации.
"""
from __future__ import annotations

import math

import pandas as pd

from ..config import RELIABILITY_CRITICAL_THRESHOLD, RELIABILITY_WARN_THRESHOLD
from ..history_stats import TagStats
from ..schemas import ReliabilityAssessment

Z_CLIP = 6.0
TOP_N_FACTORS = 3
DEVIATION_WEIGHT = 0.6
RATE_WEIGHT = 0.4


def _clean(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return val


def _tag_scores(row: dict[str, float], prev_row: dict[str, float] | None,
                 stats: dict[str, TagStats]) -> list[tuple[str, float, float]]:
    """Return (tag, deviation_0_1, rate_0_1) for tags with usable stats."""
    out = []
    for tag, s in stats.items():
        if s.std <= 0 or s.dead:
            continue
        val = _clean(row.get(tag))
        if val is None:
            continue
        deviation = min(abs(val - s.mean) / s.std, Z_CLIP) / Z_CLIP
        rate = 0.0
        if prev_row is not None:
            prev = _clean(prev_row.get(tag))
            if prev is not None:
                rate = min(abs(val - prev) / s.std, Z_CLIP) / Z_CLIP
        out.append((tag, deviation, rate))
    return out


def assess(
    timestamp: pd.Timestamp,
    avt_row: dict[str, float],
    u242000_row: dict[str, float],
    avt_prev_row: dict[str, float] | None,
    u242000_prev_row: dict[str, float] | None,
    avt_stats: dict[str, TagStats],
    u242000_stats: dict[str, TagStats],
) -> ReliabilityAssessment:
    scores = _tag_scores(avt_row, avt_prev_row, avt_stats) + _tag_scores(
        u242000_row, u242000_prev_row, u242000_stats
    )
    if not scores:
        return ReliabilityAssessment(timestamp, 0.0, "normal", [], 1.0)

    scores.sort(key=lambda t: t[1] + t[2], reverse=True)
    top = scores[:TOP_N_FACTORS]
    deviation_avg = sum(s[1] for s in top) / len(top)
    rate_avg = sum(s[2] for s in top) / len(top)
    severity = DEVIATION_WEIGHT * deviation_avg + RATE_WEIGHT * rate_avg
    severity = max(0.0, min(1.0, severity))

    if severity >= RELIABILITY_CRITICAL_THRESHOLD:
        severity_class = "critical"
        allowed_delta_scale = 0.15
    elif severity >= RELIABILITY_WARN_THRESHOLD:
        severity_class = "warn"
        allowed_delta_scale = 0.5
    else:
        severity_class = "normal"
        allowed_delta_scale = 1.0

    risk_factors = [f"{tag} (откл. {dev:.2f}, скорость {rate:.2f})" for tag, dev, rate in top]

    return ReliabilityAssessment(
        timestamp=timestamp,
        severity_index=severity,
        severity_class=severity_class,
        risk_factors=risk_factors,
        allowed_delta_scale=allowed_delta_scale,
    )
