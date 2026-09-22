"""Explicit assumptions and configuration for the МАС prototype.

Every numeric assumption that is *not* a documented technological limit from
the organizers' materials lives here, so it can be reviewed and challenged in
one place (per ТЗ: "все допущения... должны быть явно описаны").
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data-quality sentinel
# ---------------------------------------------------------------------------
# Cross-tag analysis of avt_tags.csv / 242000_tags.csv shows the exact value
# 307 recurring across unrelated, physically unrelated tags (up to 99.99% of
# one column, D10). That is a historian "bad/no data" code, not a real
# measurement. Data Agent treats it as missing.
SENTINEL_BAD_VALUE = 307.0

# A tag is considered "dead" for a session if more than this share of its
# history is the sentinel value (assumption, not given by organizers).
SENTINEL_DEAD_TAG_SHARE = 0.5


@dataclass(frozen=True)
class ControllableParam:
    tag: str
    dataset: str  # "avt" | "242000"
    description: str
    unit: str
    lo: float  # conservative modelled lower bound (assumption)
    hi: float  # conservative modelled upper bound (assumption)

    @property
    def range_(self) -> float:
        return self.hi - self.lo


# Список пересобран по `теги АВТ_24-2000.xlsx`: P8 оказался перепадом давления
# на Р-202 (симптом закоксовывания, а не уставка), Q21 — поточным анализатором
# серы, а квенч идёт по F14, а не по F15. Управляемым по 24-2000 оставлены
# только те теги, которыми оператор действительно задаёт режим.
#
# Bounds below are the 5th/95th percentile of sentinel-cleaned history —
# explicitly a *modelled experiment range*, not a validated industrial limit
# (ТЗ: "не выдавайте исторический min/max за промышленный предел").
CONTROLLABLE_PARAMS: dict[str, ControllableParam] = {
    p.tag: p
    for p in [
        ControllableParam("F3", "avt", "Расход бензина на орошение К1", "т/ч", 56.2, 112.3),
        ControllableParam("F5", "avt", "Расход пара в К1", "т/ч", -26.6, 108.3),
        ControllableParam("F19", "avt", "Расход остр. орошения К2", "т/ч", 44.7, 100.6),
        ControllableParam("F26", "avt", "Расход пара в К6", "т/ч", -25.7, -3.0),
        ControllableParam("F27", "avt", "Расход пара в К7", "т/ч", 99.7, 239.8),
        ControllableParam("F28", "avt", "Расход пара в К9", "т/ч", 150.2, 374.7),
        ControllableParam("F29", "avt", "Расход пара в К2", "т/ч", 6.2, 9.8),
        ControllableParam("F30", "avt", "Расход фр.290-350°С с установки", "т/ч", 95.1, 153.1),
        ControllableParam("F32", "avt", "Расход фр.240-290°С с установки", "т/ч", 53.0, 95.9),
        ControllableParam("F34", "avt", "Расход фр.150-250°С с установки", "т/ч", 56.0, 113.0),
        ControllableParam("T55", "avt", "Температура на выходе из печи П3", "°C", 375.8, 385.0),
        ControllableParam("F45", "avt", "Расход холодного гудрона в К10", "т/ч", 13.9, 20.1),
        ControllableParam("W4", "242000", "К-206: массовый расход бензина в колонну", "т/ч", 0.70, 4.76),
        ControllableParam("T6", "242000", "Р-202: температура ГСС на входе (жёсткость гидроочистки)", "°C", 285.2, 380.2),
        ControllableParam("F9", "242000", "Расход сырья на установку (массовый)", "т/ч", 137.2, 252.7),
        ControllableParam("F14", "242000", "Расход квенча в реактор Р-202", "т/ч", 2.97, 10.33),
        ControllableParam("F22", "242000", "К-201: расход газа поддува (объёмный)", "нм³/ч", 3882.5, 16097.3),
    ]
}

# Fraction of a param's modelled range used as one optimizer grid step.
OPTIMIZER_STEP_FRACTION = 0.08
# How many steps to try in each direction (+/-) per parameter, one lever at a time.
OPTIMIZER_STEPS_PER_SIDE = 2

# Materiality gate (assumption, not a given tolerance): a candidate is only
# considered an "action" at all if it improves quality or reliability by more
# than these margins. Throughput/energy proxies are tie-breakers *among*
# already quality/reliability-justified candidates, never a reason on their
# own to touch a setpoint (avoids continuously trimming flows for a marginal,
# unmodelled energy saving — real operators don't do that, and ТЗ explicitly
# wants no unnecessary action in a stable period).
QUALITY_MATERIAL_EPS = 2.0  # °C-scale improvement in the tracked cold-flow objective
# Калибровка: на спокойном тике максимум, который даёт один рычаг своим
# наибольшим допустимым шагом, — 1.83 °C (T6), затем 0.39 (F22) и 0.10 (F30).
# Порог поставлен выше этого потолка, чтобы обычное «подкручивание» одной
# уставки не считалось значимым улучшением: действие предлагается только при
# заметно большем выигрыше или по линии надёжности. Пересчитывать при любом
# изменении формул ВАК или списка управляемых параметров.
RELIABILITY_MATERIAL_EPS = 0.03  # severity-index improvement

# Secondary Pareto tolerance once a candidate has passed the materiality gate
# above — order: (quality_score, throughput_frac, -energy_frac, -reliability_delta).
PARETO_EPS = (0.1, 0.02, 0.02, 0.01)

# ---------------------------------------------------------------------------
# Hard constraints
# ---------------------------------------------------------------------------
SULFUR_LIMIT_MG_KG = 10.0
# Predictive risk flag fires before the hard limit is actually breached
# (ТЗ: "оценивает... риск выхода за спецификацию", not only the breach itself).
SULFUR_RISK_MARGIN_FRACTION = 0.9

# Freshness thresholds (assumption): beyond this age a source is "stale" and
# must lower confidence / block a recommendation if it is the only evidence.
LIMS_STALE_AFTER_HOURS = 48.0
PAK_STALE_AFTER_HOURS = 6.0

# Reliability agent: severity index thresholds (assumption, proxy metric —
# no direct equipment-failure labels exist in the provided data).
RELIABILITY_WARN_THRESHOLD = 0.6
RELIABILITY_CRITICAL_THRESHOLD = 0.85

# Data completeness: block a recommendation if fewer than this fraction of
# required KIP tags are present/non-sentinel in the current tick.
MIN_TAG_COVERAGE = 0.8

DASHBOARD_TICK_MINUTES = 10
