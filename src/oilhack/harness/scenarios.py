"""Реплей исторических данных как 4 сценария демонстрации (ТЗ п.6:
"На демонстрации покажите минимум..."). Окна дат подобраны по реальным
рядам (см. `docs/concept.md`), а не выдуманы:

- stable: 2025-01-08 — устойчиво низкая сера (~4.5-5 мг/кг), спокойный режим,
  все тики дают "no_action" (проверено прогоном харнесса).
- quality_risk: 2026-07-16 — реальный эпизод, где именно контрольный факт
  ЛИМС (не только ПАК) фиксирует серу 11.0 мг/кг, то есть выше жёсткого
  предела 10 мг/кг (проверено прогоном: "no_feasible_solution").
- degraded_data: синтетическая порча (явно помечено) — обрезаем ЛИМС/ПАК по
  времени и обнуляем часть КИП-тегов, чтобы показать отказ Data Agent
  ("insufficient_data" на всех тиках, проверено прогоном).
- full_cycle: 2024-01-15 — реальный день со смесью "no_action" и настоящих,
  надёжностно-обоснованных "action" (проверено прогоном: ~10 из 40 тиков),
  чтобы продемонстрировать весь цикл вплоть до Impact Agent и мониторинга
  отклика, а не искусственно навязанное действие на каждом тике.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from .. import data_io
from ..history_stats import TagStats, compute_tag_stats


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    start: str
    end: str
    inject_degraded: bool = False


SCENARIOS: dict[str, Scenario] = {
    s.key: s
    for s in [
        Scenario("stable", "Стабильный период (без лишних действий)",
                 "2025-01-08 00:00", "2025-01-08 12:00"),
        Scenario("quality_risk", "Риск качества (сера у/за пределом спецификации)",
                 "2026-07-16 10:00", "2026-07-16 22:00"),
        Scenario("degraded_data", "Неполные / устаревшие / аномальные данные",
                 "2023-06-01 00:00", "2023-06-01 06:00", inject_degraded=True),
        Scenario("full_cycle", "Полный цикл: рекомендация -> HITL -> воздействие -> мониторинг",
                 "2024-01-15 00:00", "2024-01-15 12:00"),
    ]
}


@dataclass
class Environment:
    avt: pd.DataFrame
    u242000: pd.DataFrame
    avt_stats: dict[str, TagStats]
    u242000_stats: dict[str, TagStats]
    lims: pd.DataFrame
    pak: pd.DataFrame


def load_environment() -> Environment:
    """Load all raw sources from `data/raw/` (default paths in `data_io`)."""
    avt_raw = data_io.load_avt_telemetry()
    u242000_raw = data_io.load_242000_telemetry()

    avt_stats = compute_tag_stats(avt_raw)
    u242000_stats = compute_tag_stats(u242000_raw)

    avt = data_io.mask_sentinel(avt_raw)
    u242000 = data_io.mask_sentinel(u242000_raw)

    lims = data_io.load_lims()
    pak = data_io.load_pak()

    return Environment(avt, u242000, avt_stats, u242000_stats, lims, pak)


def iter_ticks(env: Environment, scenario: Scenario):
    """Yield (timestamp, avt_row, u242000_row, avt_prev_row, u242000_prev_row,
    lims_long, pak_long) for each 10-min tick in the scenario window, applying
    the scenario's synthetic data-quality degradation if any.
    """
    idx = env.avt.index.intersection(env.u242000.index)
    idx = idx[(idx >= pd.Timestamp(scenario.start)) & (idx <= pd.Timestamp(scenario.end))]

    lims, pak = env.lims, env.pak
    if scenario.inject_degraded:
        cutoff = pd.Timestamp(scenario.start) - pd.Timedelta(days=5)
        lims = lims[lims["timestamp"] <= cutoff]
        pak = pak[pak["timestamp"] <= cutoff]

    rng = random.Random(42)
    avt_prev, u242000_prev = None, None
    for ts in idx:
        avt_row = env.avt.loc[ts].to_dict()
        u242000_row = env.u242000.loc[ts].to_dict()
        if scenario.inject_degraded:
            drop_n = max(1, int(len(avt_row) * 0.3))
            for k in rng.sample(list(avt_row.keys()), drop_n):
                avt_row[k] = float("nan")
        yield ts, avt_row, u242000_row, avt_prev, u242000_prev, lims, pak
        avt_prev, u242000_prev = avt_row, u242000_row
