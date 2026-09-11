"""Оркестратор: детерминированный ReAct-цикл (ТЗ шаг 1-8).

Важно: сама LLM (если подключена) используется только чтобы красиво
переизложить уже посчитанный `Recommendation.explanation` пользователю —
она не выбирает уставки и не проверяет ограничения. Здесь эта функция
даже не вызывается: цикл полностью детерминирован и воспроизводим без
какого-либо внешнего API, что и требуется по ТЗ ("запуск должен быть
воспроизводимым").
"""
from __future__ import annotations

import pandas as pd

from ..config import CONTROLLABLE_PARAMS
from ..history_stats import TagStats
from ..schemas import CycleResult, Recommendation, ScenarioCandidate
from . import data_agent, optimization_agent, quality_agent, reliability_agent


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
) -> CycleResult:
    trace: list[str] = []

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
        return CycleResult(timestamp, data_report, None, None, None, rec)

    trace.append("[Thought] Оценить текущее/прогнозное качество и риск выхода за спецификацию.")
    quality = quality_agent.assess(timestamp, avt_row, u242000_row, lims_long, pak_long, data_report)
    trace.append(
        f"[Action] Quality Agent -> сера {quality.sulfur_ppm if quality.sulfur_ppm is not None else 'н/д'} "
        f"мг/кг (источник: {quality.sulfur_source}), риск спецификации: {quality.spec_risk}, "
        f"уверенность {quality.confidence:.0%}."
    )

    trace.append("[Thought] Оценить тяжесть режима и ограничения для оптимизации.")
    reliability = reliability_agent.assess(
        timestamp, avt_row, u242000_row, avt_prev_row, u242000_prev_row, avt_stats, u242000_stats
    )
    trace.append(
        f"[Action] Reliability Agent -> индекс тяжести {reliability.severity_index:.2f} "
        f"({reliability.severity_class})."
    )

    trace.append("[Thought] Сгенерировать и отфильтровать варианты изменения режима.")
    opt = optimization_agent.generate(
        timestamp, avt_row, u242000_row, avt_stats, u242000_stats, quality, reliability
    )
    trace.append(
        f"[Action] Optimization Agent -> допустимых вариантов: {len(opt.candidates)}, "
        f"отброшено: {opt.infeasible_count}."
    )

    state_summary = (
        f"{timestamp}: сера {quality.sulfur_ppm:.2f} мг/кг" if quality.sulfur_ppm is not None
        else f"{timestamp}: сера неизвестна"
    ) + f", тяжесть режима {reliability.severity_class}, полнота данных {data_report.tag_coverage:.0%}"

    constraints_checked = (
        f"Сера <= {10.0} мг/кг: {'нет данных' if quality.sulfur_ppm is None else ('нарушено' if quality.spec_risk and quality.sulfur_ppm >= 10.0 else 'в норме')}; "
        f"индекс тяжести режима: {reliability.severity_class}; "
        "модельные диапазоны управляемых параметров: соблюдены по построению (сценарии генерируются внутри допущенного диапазона)."
    )

    if not opt.candidates:
        trace.append("[Observation] Допустимого варианта нет.")
        reason = (
            "последнее измерение серы устарело или отсутствует"
            if quality.sulfur_ppm is None else
            f"текущая сера {quality.sulfur_ppm:.2f} мг/кг на пределе/за пределом спецификации"
        )
        rec = Recommendation(
            timestamp=timestamp,
            status="no_feasible_solution",
            state_summary=state_summary,
            problem=f"Риск по сере: {reason}",
            action="Без изменений — новые уставки не рекомендуются",
            expected_effect="—",
            constraints_checked=constraints_checked,
            confidence=_confidence_label(quality.confidence),
            explanation=(
                f"Надёжной рекомендации нет: {reason}, поэтому доступные варианты либо "
                "нарушают ограничение по качеству, либо не могут быть проверены. "
                "Умение корректно отказаться — часть данного решения."
            ),
            react_trace=trace,
            sulfur_ppm=quality.sulfur_ppm,
            reliability_severity=reliability.severity_index,
        )
        return CycleResult(timestamp, data_report, quality, reliability, opt, rec)

    chosen = opt.candidates[0]
    alternatives = opt.candidates[1:]

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
            explanation="Ни один из рассмотренных вариантов изменения режима не улучшает "
                        "сравниваемые критерии сильнее, чем текущий режим — лишних действий не требуется.",
            react_trace=trace,
            chosen=chosen,
            alternatives=alternatives,
            sulfur_ppm=quality.sulfur_ppm,
            reliability_severity=reliability.severity_index,
        )
        return CycleResult(timestamp, data_report, quality, reliability, opt, rec)

    trace.append(f"[Observation] Выбран вариант: {chosen.label}.")
    rec = Recommendation(
        timestamp=timestamp,
        status="action",
        state_summary=state_summary,
        problem="Найден вариант, улучшающий сравниваемые критерии без нарушения ограничений",
        action=_fmt_candidate(chosen),
        expected_effect=(
            f"Индекс тяжести режима: {reliability.severity_index:.2f} -> {chosen.reliability_severity:.2f}; "
            f"выпуск/энергопрокси учтены в ранжировании (Парето-фронт)."
        ),
        constraints_checked=constraints_checked,
        confidence=_confidence_label(quality.confidence),
        explanation=(
            f"Вариант «{chosen.label}» находится на Парето-фронте из {len(opt.candidates)} допустимых "
            f"сценариев (ещё {len(alternatives)} — альтернативы) и не нарушает жёсткие ограничения."
        ),
        react_trace=trace,
        chosen=chosen,
        alternatives=alternatives,
        sulfur_ppm=quality.sulfur_ppm,
        reliability_severity=reliability.severity_index,
    )
    return CycleResult(timestamp, data_report, quality, reliability, opt, rec)
