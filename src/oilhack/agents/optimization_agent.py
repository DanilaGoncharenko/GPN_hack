"""Агент оптимизации: сценарии, контрфактический прогноз, Safety Gate и Pareto.

Порядок строго такой:

    candidate -> VAK -> Reliability -> SulfurForecast -> ConstraintAgent -> Pareto

Следовательно, экономические/proxy-цели никогда не могут «перевесить» hard
constraint по сере или критический режим.
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
    SULFUR_MATERIAL_EPS,
    SULFUR_SCORE_WEIGHT,
)
from ..history_stats import TagStats
from ..schemas import OptimizationResult, QualityForecast, ReliabilityAssessment, ScenarioCandidate
from ..vak_formulas import AVT_ANALYZERS, U242000_ANALYZERS, all_present
from . import constraint_agent, reliability_agent
from .sulfur_forecast_agent import SulfurForecastAgent, SulfurPrediction

COLD_FLOW_METRICS = (
    "AVT6:240-350:CFPP", "AVT6:350:CFPP", "24-2000:GODT:CFPP",
    "24-2000:GODT:CloudPoint",
)
ENERGY_TAGS = {"F5", "F26", "F27", "F28", "F29", "F45", "T55", "T6", "F14", "F22"}
THROUGHPUT_TAGS = {"F9", "F30", "F32", "F34", "W4"}
_ANALYZERS_BY_DATASET = {"avt": AVT_ANALYZERS, "242000": U242000_ANALYZERS}


def _recompute_affected(
    dataset: str,
    changed_tag: str,
    row: dict[str, float],
    baseline_predictions: dict[str, float],
    lims_inputs_used: dict[str, float],
) -> dict[str, float]:
    preds = dict(baseline_predictions)
    for key, analyzer in _ANALYZERS_BY_DATASET[dataset].items():
        if changed_tag in analyzer.required_tags and all_present(row, analyzer.required_tags):
            try:
                lims_inputs = {k: lims_inputs_used.get(k) for k in analyzer.lims_inputs}
                if any(v is None for v in lims_inputs.values()):
                    continue
                preds[key] = analyzer.compute(row, lims_inputs)
            except (KeyError, ZeroDivisionError):
                pass
    return preds


def _score(
    baseline_predictions: dict[str, float],
    new_predictions: dict[str, float],
    changed_tag: str,
    delta_value: float,
    param_range: float,
) -> tuple[float, float, float]:
    quality_score = 0.0
    for key in COLD_FLOW_METRICS:
        if key in new_predictions and key in baseline_predictions:
            quality_score += baseline_predictions[key] - new_predictions[key]

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
    return [
        c for c in candidates
        if not any(_dominates(other, c, scores) for other in candidates if other.label != c.label)
    ]


def generate(
    timestamp: pd.Timestamp,
    avt_row: dict[str, float],
    u242000_row: dict[str, float],
    avt_stats: dict[str, TagStats],
    u242000_stats: dict[str, TagStats],
    quality: QualityForecast,
    reliability: ReliabilityAssessment,
    sulfur_forecaster: SulfurForecastAgent | None = None,
    history_avt: pd.DataFrame | None = None,
    history_hydro: pd.DataFrame | None = None,
    data_usable: bool = True,
) -> OptimizationResult:
    baseline_sulfur_forecast = None
    if sulfur_forecaster is not None and history_avt is not None and history_hydro is not None:
        baseline_sulfur_forecast = sulfur_forecaster.predict(history_avt, history_hydro, anchor_sulfur=quality.sulfur_ppm)

    baseline = ScenarioCandidate(
        label="Текущий режим (без изменений)",
        deltas={},
        predicted_sulfur_ppm=(baseline_sulfur_forecast.p50 if baseline_sulfur_forecast else quality.sulfur_ppm),
        predicted_quality=dict(quality.predictions),
        reliability_severity=reliability.severity_index,
        throughput_proxy=0.0,
        energy_proxy=0.0,
        feasible=True,
        predicted_sulfur_upper_ppm=(baseline_sulfur_forecast.upper if baseline_sulfur_forecast else None),
    )

    # Сначала формируем численные контрфакты и Reliability/VAK. Прогноз серы
    # считаем пакетно после этого цикла, чтобы не вызывать sklearn 50-70 раз
    # на каждом 10-минутном тике.
    raw: list[tuple[object, float, float, dict[str, float], ReliabilityAssessment, dict[str, float], float, float, float, str]] = []
    sulfur_change_keys: list[dict[str, float]] = []

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
                quality_score, throughput_frac, energy_frac = _score(
                    quality.predictions, new_predictions, param.tag, delta_value, param.range_
                )
                reliability_delta = new_reliability.severity_index - reliability.severity_index
                label = f"{param.tag} {'+' if delta_value > 0 else ''}{delta_value:.2f} ({param.description})"
                raw.append((
                    param, delta_value, new_val, new_predictions, new_reliability,
                    {param.tag: new_val}, quality_score, throughput_frac, energy_frac, label,
                ))
                sulfur_change_keys.append({f"{param.dataset}__{param.tag}": new_val})

    sulfur_predictions: list[SulfurPrediction | None]
    if (
        sulfur_forecaster is not None
        and history_avt is not None
        and history_hydro is not None
        and sulfur_change_keys
    ):
        sulfur_predictions = list(sulfur_forecaster.predict_many(history_avt, history_hydro, sulfur_change_keys, anchor_sulfur=quality.sulfur_ppm))
    else:
        sulfur_predictions = [None] * len(raw)

    candidates: list[ScenarioCandidate] = []
    infeasible_count = 0
    scores: dict[str, tuple] = {baseline.label: (0.0, 0.0, 0.0, 0.0)}

    for item, sulfur_prediction in zip(raw, sulfur_predictions):
        param, delta_value, new_val, new_predictions, new_reliability, deltas, quality_score, throughput_frac, energy_frac, label = item
        constraint = constraint_agent.check(
            deltas,
            sulfur_prediction.upper if sulfur_prediction else None,
            new_reliability.severity_index,
            data_usable=data_usable,
        )

        sulfur_gain = 0.0
        if quality.sulfur_ppm is not None and sulfur_prediction is not None and quality.spec_risk:
            sulfur_gain = quality.sulfur_ppm - sulfur_prediction.p50
            quality_score += SULFUR_SCORE_WEIGHT * sulfur_gain
        reliability_delta = new_reliability.severity_index - reliability.severity_index

        cand = ScenarioCandidate(
            label=label,
            deltas=deltas,
            predicted_sulfur_ppm=(sulfur_prediction.p50 if sulfur_prediction else None),
            predicted_sulfur_upper_ppm=(sulfur_prediction.upper if sulfur_prediction else None),
            predicted_quality=new_predictions,
            reliability_severity=new_reliability.severity_index,
            throughput_proxy=delta_value if param.tag in THROUGHPUT_TAGS else 0.0,
            energy_proxy=delta_value if param.tag in ENERGY_TAGS else 0.0,
            feasible=constraint.feasible,
            violations=list(constraint.violations),
        )
        if not cand.feasible:
            infeasible_count += 1
            continue

        sulfur_material = sulfur_gain >= SULFUR_MATERIAL_EPS
        if (
            quality_score <= QUALITY_MATERIAL_EPS
            and -reliability_delta <= RELIABILITY_MATERIAL_EPS
            and not sulfur_material
        ):
            continue
        candidates.append(cand)
        scores[label] = (quality_score, throughput_frac, -energy_frac, -reliability_delta)

    front = _pareto_front([baseline] + candidates, scores)
    front.sort(key=lambda c: sum(scores[c.label]), reverse=True)

    return OptimizationResult(
        timestamp=timestamp,
        candidates=[c for c in front if c.label != baseline.label] or front,
        infeasible_count=infeasible_count,
        baseline=baseline,
    )

