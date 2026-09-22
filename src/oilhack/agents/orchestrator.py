"""Оркестратор полного цикла принятия решения (ТЗ шаги 1-8).

Численная логика детерминирована. Новые роли:
- Sulfur Forecast Agent — контрфактический прогноз серы;
- Constraint Agent — hard safety gate;
- Blending Agent — проверка доступности блендинга без выдуманных тегов.

LLM, если позднее подключается для объяснения, не получает права менять
уставки или обходить ограничения.
"""
from __future__ import annotations

import pandas as pd

from ..config import CONTROLLABLE_PARAMS, SULFUR_LIMIT_MG_KG
from ..history_stats import TagStats
from ..schemas import CycleResult, Recommendation, ScenarioCandidate
from . import blending_agent, constraint_agent, data_agent, optimization_agent, quality_agent, reliability_agent
from .sulfur_forecast_agent import SulfurForecastAgent, SulfurPrediction


def _confidence_label(value: float) -> str:
    if value >= 0.75:
        return f"высокая ({value:.0%})"
    if value >= 0.4:
        return f"средняя ({value:.0%})"
    return f"низкая ({value:.0%})"


def _fmt_candidate(cand: ScenarioCandidate) -> str:
    if not cand.deltas:
        return "без изменений"
    parts = []
    for tag, val in cand.deltas.items():
        param = CONTROLLABLE_PARAMS.get(tag)
        desc = param.description if param else tag
        parts.append(f"{desc} ({tag}) -> {val:.2f}")
    return "; ".join(parts)


def run_cycle(
    timestamp: pd.Timestamp,
    avt_row: dict[str, float],
    u242000_row: dict[str, float],
    avt_prev_row: dict[str, float] | None,
    u242000_prev_row: dict[str, float] | None,
    avt_stats: dict[str, TagStats],
    u242000_stats: dict[str, TagStats],
    lims_long: pd.DataFrame,
    pak_long: pd.DataFrame,
    sulfur_forecaster: SulfurForecastAgent | None = None,
    history_avt: pd.DataFrame | None = None,
    history_hydro: pd.DataFrame | None = None,
) -> CycleResult:
    trace: list[str] = []
    blending = blending_agent.assess()

    trace.append(f"[Thought] Тик {timestamp}: получить состояние процесса и оценить пригодность данных.")
    data_report = data_agent.assess(
        timestamp, avt_row, u242000_row, avt_stats, u242000_stats, lims_long, pak_long
    )
    trace.append(
        f"[Action] Data Agent -> полнота КИП {data_report.tag_coverage:.0%}, "
        f"аномалий: {len(data_report.anomalous_tags)}, пропусков: {len(data_report.missing_tags)}."
    )

    if not data_report.is_usable:
        trace.append("[Observation] Данных недостаточно для дальнейшего цикла.")
        rec = Recommendation(
            timestamp=timestamp,
            status="insufficient_data",
            state_summary=f"Полнота КИП {data_report.tag_coverage:.0%}",
            problem="Недостаточно данных для формирования рекомендации",
            action="Без изменений",
            expected_effect="—",
            constraints_checked="Проверка полноты данных не пройдена",
            confidence="низкая",
            explanation=(
                "Надёжной рекомендации нет: полнота входных технологических данных "
                f"{data_report.tag_coverage:.0%} ниже требуемого порога. Причины: "
                + "; ".join(data_report.reasons or ["не указаны"])
            ),
            react_trace=trace,
        )
        return CycleResult(timestamp, data_report, None, None, None, rec, None, None, blending)

    trace.append("[Thought] Оценить текущее/прогнозное качество и риск выхода за спецификацию.")
    quality = quality_agent.assess(timestamp, avt_row, u242000_row, lims_long, pak_long, data_report)
    trace.append(
        f"[Action] Quality Agent -> сера {quality.sulfur_ppm if quality.sulfur_ppm is not None else 'н/д'} "
        f"мг/кг (источник: {quality.sulfur_source}), риск спецификации: {quality.spec_risk}, "
        f"уверенность {quality.confidence:.0%}."
    )

    sulfur_forecast: SulfurPrediction | None = None
    if sulfur_forecaster is not None and history_avt is not None and history_hydro is not None:
        try:
            sulfur_forecast = sulfur_forecaster.predict(history_avt, history_hydro, anchor_sulfur=quality.sulfur_ppm)
            trace.append(
                f"[Action] Sulfur Forecast Agent -> горизонт {sulfur_forecast.horizon_minutes} мин, "
                f"p50={sulfur_forecast.p50:.2f}, upper={sulfur_forecast.upper:.2f} мг/кг."
            )
        except (ValueError, RuntimeError):
            trace.append("[Observation] Sulfur Forecast Agent не смог построить прогноз для текущего тика.")
    else:
        trace.append("[Observation] Контрфактический прогноз серы не подключён.")

    trace.append("[Thought] Оценить тяжесть режима и ограничения для оптимизации.")
    reliability = reliability_agent.assess(
        timestamp, avt_row, u242000_row, avt_prev_row, u242000_prev_row, avt_stats, u242000_stats
    )
    trace.append(
        f"[Action] Reliability Agent -> индекс тяжести {reliability.severity_index:.2f} "
        f"({reliability.severity_class})."
    )

    trace.append("[Thought] Сгенерировать варианты и прогнать каждый через Safety Gate.")
    opt = optimization_agent.generate(
        timestamp,
        avt_row,
        u242000_row,
        avt_stats,
        u242000_stats,
        quality,
        reliability,
        sulfur_forecaster=sulfur_forecaster,
        history_avt=history_avt,
        history_hydro=history_hydro,
        data_usable=data_report.is_usable,
    )
    trace.append(
        f"[Action] Optimization Agent -> допустимых вариантов: {len(opt.candidates)}, "
        f"отброшено Safety Gate: {opt.infeasible_count}."
    )
    trace.append(
        f"[Action] Blending Agent -> {'доступен' if blending.available else 'не используется'}: {blending.reason}"
    )

    state_sulfur = "н/д" if quality.sulfur_ppm is None else f"{quality.sulfur_ppm:.2f} мг/кг"
    forecast_text = (
        f", прогноз {sulfur_forecast.p50:.2f} / upper {sulfur_forecast.upper:.2f} мг/кг"
        if sulfur_forecast else ", контрфактический прогноз серы недоступен"
    )
    state_summary = (
        f"{timestamp}: сера {state_sulfur}{forecast_text}, "
        f"тяжесть режима {reliability.severity_class}, полнота данных {data_report.tag_coverage:.0%}"
    )

    baseline_upper = opt.baseline.predicted_sulfur_upper_ppm
    constraints_checked = (
        f"Сера <= {SULFUR_LIMIT_MG_KG:.1f} мг/кг: "
        f"верхняя граница прогноза {'н/д' if baseline_upper is None else f'{baseline_upper:.2f}'}; "
        f"индекс тяжести: {reliability.severity_class}; "
        "модельные диапазоны управляющих параметров соблюдены; "
        "блендинг не оптимизируется без подтверждённых компонентных потоков."
    )

    if not opt.candidates:
        trace.append("[Observation] Допустимого варианта нет после Safety Gate.")
        if baseline_upper is None:
            reason = "контрфактический прогноз серы недоступен"
        elif baseline_upper > SULFUR_LIMIT_MG_KG:
            reason = (
                f"верхняя граница базового прогноза серы {baseline_upper:.2f} мг/кг "
                f"выше лимита {SULFUR_LIMIT_MG_KG:.1f} мг/кг"
            )
        else:
            reason = "ни один кандидат не прошёл hard constraints и порог значимого улучшения"
        rec = Recommendation(
            timestamp=timestamp,
            status="no_feasible_solution",
            state_summary=state_summary,
            problem=f"Нет безопасного действия: {reason}",
            action="Без изменений — новые уставки не рекомендуются",
            expected_effect="—",
            constraints_checked=constraints_checked,
            confidence=_confidence_label(quality.confidence),
            explanation=(
                f"После проверки {opt.infeasible_count} отклонённых сценариев безопасной рекомендации нет: {reason}. "
                "Экономический или производственный эффект не используется для обхода Safety Gate."
            ),
            react_trace=trace,
            sulfur_ppm=quality.sulfur_ppm,
            reliability_severity=reliability.severity_index,
            predicted_sulfur_upper_ppm=baseline_upper,
        )
        return CycleResult(
            timestamp, data_report, quality, reliability, opt, rec,
            sulfur_forecast, None, blending,
        )

    chosen = opt.candidates[0]
    alternatives = opt.candidates[1:]

    constraint_report = constraint_agent.check(
        chosen.deltas,
        chosen.predicted_sulfur_upper_ppm,
        chosen.reliability_severity,
        data_usable=data_report.is_usable,
    )
    trace.append(
        f"[Action] Constraint Agent -> {'PASS' if constraint_report.feasible else 'REJECT'}"
        + (f": {'; '.join(constraint_report.violations)}" if constraint_report.violations else "")
    )

    if not constraint_report.feasible:
        trace.append("[Observation] Выбранный Pareto-кандидат дополнительно отклонён Safety Gate.")
        rec = Recommendation(
            timestamp=timestamp,
            status="no_feasible_solution",
            state_summary=state_summary,
            problem="Все найденные сценарии были отклонены финальной проверкой ограничений",
            action="Без изменений",
            expected_effect="—",
            constraints_checked="; ".join(constraint_report.violations),
            confidence=_confidence_label(quality.confidence),
            explanation="Оптимизация не имеет права переопределять hard constraints, поэтому режим не предлагается.",
            react_trace=trace,
            sulfur_ppm=quality.sulfur_ppm,
            reliability_severity=reliability.severity_index,
            predicted_sulfur_upper_ppm=baseline_upper,
        )
        return CycleResult(
            timestamp, data_report, quality, reliability, opt, rec,
            sulfur_forecast, constraint_report, blending,
        )

    if not chosen.deltas:
        trace.append("[Observation] Лучший вариант — не менять режим.")
        rec = Recommendation(
            timestamp=timestamp,
            status="no_action",
            state_summary=state_summary,
            problem="Существенных отклонений не обнаружено",
            action="Без изменений — текущий режим уже является предпочтительным",
            expected_effect="Показатели качества/надёжности остаются на текущем уровне",
            constraints_checked=constraints_checked,
            confidence=_confidence_label(quality.confidence),
            explanation=(
                "Ни один рассмотренный вариант не даёт значимого преимущества при соблюдении Safety Gate; "
                "лишних управляющих действий не требуется."
            ),
            react_trace=trace,
            chosen=chosen,
            alternatives=alternatives,
            sulfur_ppm=quality.sulfur_ppm,
            reliability_severity=reliability.severity_index,
            predicted_sulfur_upper_ppm=baseline_upper,
        )
        return CycleResult(
            timestamp, data_report, quality, reliability, opt, rec,
            sulfur_forecast, constraint_report, blending,
        )

    trace.append(f"[Observation] Выбран вариант: {chosen.label}.")
    chosen_sulfur = (
        f"сера {chosen.predicted_sulfur_ppm:.2f} мг/кг, upper {chosen.predicted_sulfur_upper_ppm:.2f} мг/кг"
        if chosen.predicted_sulfur_ppm is not None and chosen.predicted_sulfur_upper_ppm is not None
        else "контрфактический прогноз серы недоступен"
    )
    rec = Recommendation(
        timestamp=timestamp,
        status="action",
        state_summary=state_summary,
        problem="Найден допустимый сценарий изменения режима",
        action=_fmt_candidate(chosen),
        expected_effect=(
            f"{chosen_sulfur}; тяжесть режима: {reliability.severity_index:.2f} -> "
            f"{chosen.reliability_severity:.2f}; ВАК и proxy-цели учтены в Pareto."
        ),
        constraints_checked=(
            f"Сера upper <= {SULFUR_LIMIT_MG_KG:.1f}: {'да' if chosen.predicted_sulfur_upper_ppm is not None and chosen.predicted_sulfur_upper_ppm <= SULFUR_LIMIT_MG_KG else 'нет'}; "
            f"тяжесть: {chosen.reliability_severity:.2f}; модельные диапазоны: да."
        ),
        confidence=_confidence_label(quality.confidence),
        explanation=(
            f"Вариант «{chosen.label}» прошёл hard constraints и находится на Pareto-фронте. "
            f"Прогноз серы выполняется на горизонте {sulfur_forecast.horizon_minutes if sulfur_forecast else 'н/д'} мин. "
            "Экономические proxy используются только после проверки безопасности."
        ),
        react_trace=trace,
        chosen=chosen,
        alternatives=alternatives,
        sulfur_ppm=quality.sulfur_ppm,
        reliability_severity=reliability.severity_index,
        predicted_sulfur_upper_ppm=baseline_upper,
    )
    return CycleResult(
        timestamp, data_report, quality, reliability, opt, rec,
        sulfur_forecast, constraint_report, blending,
    )
