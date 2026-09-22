"""Агент оптимизации: кандидатные сценарии, фильтр жёстких ограничений,
Парето-сравнение (ТЗ шаг 5-7).

Явное ограничение метода (задокументировано, а не скрыто): в выданном
пакете ВАК-формул нет регрессии для серы — только измеренные ЛИМС/ПАК.
Поэтому прогноз серы для гипотетических сценариев не строится; жёсткий
лимит проверяется по последнему измеренному значению и одинаков для всех
кандидатов на этом тике. Если текущая сера уже на грани/за пределом —
весь набор кандидатов признаётся недопустимым (см. `feasible`).

Направление "лучше/хуже" по CFPP и CloudPoint (ниже — лучше) — допущение
для зимнего сценария эксплуатации, явно помечено.
"""
from __future__ import annotations

import pandas as pd

from ..config import (
    CONTROLLABLE_PARAMS,
    OPTIMIZER_STEP_FRACTION,
    OPTIMIZER_STEPS_PER_SIDE,
    PARETO_EPS,
    QUALITY_MATERIAL_EPS,
    RELIABILITY_MATERIAL_EPS,
    SULFUR_LIMIT_MG_KG,
)
from ..history_stats import TagStats
from ..schemas import (
    QualityForecast,
    ReliabilityAssessment,
    ScenarioCandidate,
    OptimizationResult,
)
from ..vak_formulas import AVT_ANALYZERS, U242000_ANALYZERS, all_present
from . import reliability_agent

COLD_FLOW_METRICS = ("AVT6:240-350:CFPP", "AVT6:350:CFPP", "24-2000:GODT:CFPP",
                      "24-2000:GODT:CloudPoint")
ENERGY_TAGS = {"F5", "F26", "F27", "F28", "F29", "F45", "T55", "T6", "F14", "F22"}
THROUGHPUT_TAGS = {"F9", "F30", "F32", "F34", "W4"}

_ANALYZERS_BY_DATASET = {"avt": AVT_ANALYZERS, "242000": U242000_ANALYZERS}


def _recompute_affected(dataset: str, changed_tag: str, row: dict[str, float],
                         baseline_predictions: dict[str, float],
                         lims_inputs_used: dict[str, float]) -> dict[str, float]:
    preds = dict(baseline_predictions)
    for key, analyzer in _ANALYZERS_BY_DATASET[dataset].items():
        if changed_tag in analyzer.required_tags and all_present(row, analyzer.required_tags):
            try:
                # LIMS-dependent GODT formulas keep their last resolved LIMS
                # input (unaffected by an AVT/242000 setpoint change).
                lims_inputs = {k: lims_inputs_used.get(k) for k in analyzer.lims_inputs}
                if any(v is None for v in lims_inputs.values()):
                    continue
                preds[key] = analyzer.compute(row, lims_inputs)
            except (KeyError, ZeroDivisionError):
                pass
    return preds


def _score(baseline_predictions: dict[str, float], new_predictions: dict[str, float],
           changed_tag: str, delta_value: float, param_range: float) -> tuple[float, float, float]:
    quality_score = 0.0
    for key in COLD_FLOW_METRICS:
        if key in new_predictions and key in baseline_predictions:
            quality_score += baseline_predictions[key] - new_predictions[key]  # lower is better

    delta_frac = delta_value / param_range if param_range else 0.0
    throughput_proxy = delta_frac if changed_tag in THROUGHPUT_TAGS else 0.0
    energy_proxy = delta_frac if changed_tag in ENERGY_TAGS else 0.0
    return quality_score, throughput_proxy, energy_proxy


def _dominates(a: ScenarioCandidate, b: ScenarioCandidate, scores: dict[str, tuple],
               eps: tuple[float, ...] = PARETO_EPS) -> bool:
    sa, sb = scores[a.label], scores[b.label]
    at_least_as_good = all(x >= y - e for x, y, e in zip(sa, sb, eps))
    strictly_better = any(x > y + e for x, y, e in zip(sa, sb, eps))
    return at_least_as_good and strictly_better


def _pareto_front(candidates: list[ScenarioCandidate], scores: dict[str, tuple]) -> list[ScenarioCandidate]:
    front = []
    for c in candidates:
        if not any(_dominates(other, c, scores) for other in candidates if other.label != c.label):
            front.append(c)
    return front


def generate(
    timestamp: pd.Timestamp,
    avt_row: dict[str, float],
    u242000_row: dict[str, float],
    avt_stats: dict[str, TagStats],
    u242000_stats: dict[str, TagStats],
    quality: QualityForecast,
    reliability: ReliabilityAssessment,
) -> OptimizationResult:
    baseline = ScenarioCandidate(
        label="Текущий режим (без изменений)",
        deltas={},
        predicted_sulfur_ppm=quality.sulfur_ppm,
        predicted_quality=dict(quality.predictions),
        reliability_severity=reliability.severity_index,
        throughput_proxy=0.0,
        energy_proxy=0.0,
        feasible=True,
    )

    sulfur_known = quality.sulfur_ppm is not None
    sulfur_safe = sulfur_known and quality.sulfur_ppm < SULFUR_LIMIT_MG_KG
    sulfur_gate_msg = None
    if not sulfur_known:
        sulfur_gate_msg = "Нет свежего измерения серы — прогноз для новых режимов не строится"
    elif not sulfur_safe:
        sulfur_gate_msg = (
            f"Сера {quality.sulfur_ppm:.2f} мг/кг у/на пределе {SULFUR_LIMIT_MG_KG} мг/кг — "
            "смена режима не рассматривается, пока сера не подтверждена в норме"
        )

    candidates: list[ScenarioCandidate] = []
    infeasible_count = 0

    if sulfur_gate_msg is not None:
        return OptimizationResult(timestamp=timestamp, candidates=[], infeasible_count=1, baseline=baseline)

    scores: dict[str, tuple] = {baseline.label: (0.0, 0.0, 0.0, 0.0)}

    for param in CONTROLLABLE_PARAMS.values():
        row = avt_row if param.dataset == "avt" else u242000_row
        current_val = row.get(param.tag)
        if current_val is None or (isinstance(current_val, float) and current_val != current_val):
            continue
        step = param.range_ * OPTIMIZER_STEP_FRACTION * reliability.allowed_delta_scale
        for n in range(1, OPTIMIZER_STEPS_PER_SIDE + 1):
            for sign in (-1, 1):
                delta_value = sign * n * step
                new_val = max(param.lo, min(param.hi, current_val + delta_value))
                if new_val == current_val:
                    continue
                new_row = dict(row)
                new_row[param.tag] = new_val

                new_predictions = _recompute_affected(
                    param.dataset, param.tag, new_row, quality.predictions, quality.lims_inputs_used
                )

                if param.dataset == "avt":
                    new_reliability = reliability_agent.assess(
                        timestamp, new_row, u242000_row, row, u242000_row, avt_stats, u242000_stats
                    )
                else:
                    new_reliability = reliability_agent.assess(
                        timestamp, avt_row, new_row, avt_row, row, avt_stats, u242000_stats
                    )

                violations = []
                if new_reliability.severity_class == "critical":
                    violations.append("Индекс тяжести режима становится критическим")

                quality_score, throughput_frac, energy_frac = _score(
                    quality.predictions, new_predictions, param.tag, delta_value, param.range_
                )
                reliability_delta = new_reliability.severity_index - reliability.severity_index

                label = f"{param.tag} {'+' if delta_value > 0 else ''}{delta_value:.2f} ({param.description})"
                cand = ScenarioCandidate(
                    label=label,
                    deltas={param.tag: new_val},
                    predicted_sulfur_ppm=quality.sulfur_ppm,
                    predicted_quality=new_predictions,
                    reliability_severity=new_reliability.severity_index,
                    throughput_proxy=delta_value if param.tag in THROUGHPUT_TAGS else 0.0,
                    energy_proxy=delta_value if param.tag in ENERGY_TAGS else 0.0,
                    feasible=not violations,
                    violations=violations,
                )
                if not cand.feasible:
                    infeasible_count += 1
                    continue
                # Materiality gate: throughput/energy alone never justify a
                # setpoint change — only quality or reliability do.
                if quality_score <= QUALITY_MATERIAL_EPS and -reliability_delta <= RELIABILITY_MATERIAL_EPS:
                    continue
                candidates.append(cand)
                scores[label] = (quality_score, throughput_frac, -energy_frac, -reliability_delta)

    all_for_pareto = [baseline] + candidates
    front = _pareto_front(all_for_pareto, scores)
    front.sort(key=lambda c: sum(scores[c.label]), reverse=True)

    return OptimizationResult(
        timestamp=timestamp,
        candidates=[c for c in front if c.label != baseline.label] or front,
        infeasible_count=infeasible_count,
        baseline=baseline,
    )
