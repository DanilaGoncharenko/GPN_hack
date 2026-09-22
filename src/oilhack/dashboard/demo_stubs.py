"""Демо-заглушки: три готовых примера цикла принятия решения.

Нужны, чтобы дашборд открывался мгновенно и был понятен без загрузки
250 МБ телеметрии — для показа логики системы проверяющему. Числа взяты
из реальных прогонов харнесса на исторических данных (см. docs/concept.md),
но зашиты статически: это витрина, а не расчёт.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..schemas import (
    CycleResult,
    DataQualityReport,
    OptimizationResult,
    QualityForecast,
    Recommendation,
    ReliabilityAssessment,
    ScenarioCandidate,
)


@dataclass
class DemoExample:
    key: str
    title: str
    one_liner: str
    result: CycleResult
    series: pd.DataFrame  # timestamp, sulfur, severity — для графика


def _series(start: str, sulfur: list[float], severity: list[float]) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(start), periods=len(sulfur), freq="10min")
    return pd.DataFrame({"timestamp": idx, "sulfur": sulfur, "severity": severity})


def _calm() -> DemoExample:
    ts = pd.Timestamp("2025-01-08 00:00:00")
    data = DataQualityReport(
        timestamp=ts, tag_coverage=0.98, missing_tags=["D10", "F26"],
        anomalous_tags=[], dead_tags=["D10"],
        lims_age_hours={"hydrotreat_pipeline": 14.0},
        pak_age_hours={"24-2000:Mg.Sulfur": 0.0, "24-2000:D15": None},
        is_usable=True, reasons=[],
    )
    quality = QualityForecast(
        timestamp=ts,
        predictions={"24-2000:GODT:CFPP": -19.59, "24-2000:GODT:T90": 342.10,
                      "AVT6:240-350:D15": 846.64, "24-2000:GODT:CloudPoint": -3.26},
        lims_inputs_used={"LIMS:24-2000.Pipeline.D15": 838.4},
        sulfur_ppm=5.10, sulfur_source="ЛИМС", sulfur_age_hours=14.0,
        spec_risk=False, confidence=0.94,
        confidence_reasons=["Полнота КИП 98%"],
    )
    reliability = ReliabilityAssessment(
        timestamp=ts, severity_index=0.22, severity_class="normal",
        risk_factors=["F63 (откл. 0.44, скорость 0.00)", "F69 (откл. 0.36, скорость 0.00)"],
        allowed_delta_scale=1.0,
    )
    baseline = ScenarioCandidate(
        label="Текущий режим (без изменений)", deltas={}, predicted_sulfur_ppm=5.10,
        predicted_quality=dict(quality.predictions), reliability_severity=0.22,
        throughput_proxy=0.0, energy_proxy=0.0, feasible=True,
    )
    rec = Recommendation(
        timestamp=ts, status="no_action",
        state_summary="2025-01-08 00:00: сера 5.10 мг/кг, тяжесть режима normal, полнота данных 98%",
        problem="Существенных отклонений не обнаружено",
        action="Без изменений — текущий режим уже является предпочтительным",
        expected_effect="Показатели качества и надёжности остаются на текущем уровне",
        constraints_checked="Сера ≤ 10 мг/кг: в норме (5.10); тяжесть режима: normal; "
                            "уставки в пределах модельного диапазона",
        confidence="высокая (94%)",
        explanation="Ни один из 68 рассмотренных вариантов не улучшает сравниваемые критерии "
                    "заметнее порога значимости, поэтому трогать режим не нужно.",
        react_trace=[
            "[Мысль] Тик 2025-01-08 00:00: получить состояние процесса и оценить пригодность данных.",
            "[Действие] Агент данных → полнота КИП 98%, аномалий: 0, пропусков: 2.",
            "[Мысль] Оценить качество продукта и риск выхода за спецификацию.",
            "[Действие] Агент качества → сера 5.10 мг/кг (ЛИМС), риска нет, уверенность 94%.",
            "[Мысль] Оценить тяжесть режима для оборудования.",
            "[Действие] Агент надёжности → индекс 0.22 (норма).",
            "[Мысль] Сгенерировать и отфильтровать варианты изменения режима.",
            "[Действие] Агент оптимизации → 68 вариантов проверено, значимых улучшений нет.",
            "[Вывод] Лучший вариант — не менять режим.",
        ],
        chosen=baseline, alternatives=[], sulfur_ppm=5.10, reliability_severity=0.22,
    )
    opt = OptimizationResult(timestamp=ts, candidates=[baseline], infeasible_count=0, baseline=baseline)
    return DemoExample(
        key="calm",
        title="Спокойный режим — система молчит",
        one_liner="Всё в допуске, значимых улучшений нет, поэтому система не советует ничего менять.",
        result=CycleResult(ts, data, quality, reliability, opt, rec),
        series=_series("2025-01-08 00:00",
                        [5.1, 5.1, 5.0, 5.1, 5.2, 5.1, 5.0, 4.9, 5.0, 5.1, 5.1, 5.0],
                        [0.22, 0.23, 0.22, 0.21, 0.24, 0.22, 0.20, 0.21, 0.22, 0.23, 0.22, 0.21]),
    )


def _sulfur_risk() -> DemoExample:
    ts = pd.Timestamp("2026-07-16 10:00:00")
    data = DataQualityReport(
        timestamp=ts, tag_coverage=0.98, missing_tags=["D10", "F26"], anomalous_tags=[],
        dead_tags=["D10"], lims_age_hours={"hydrotreat_pipeline": 0.0},
        pak_age_hours={"24-2000:Mg.Sulfur": 0.0, "24-2000:D15": None},
        is_usable=True, reasons=[],
    )
    quality = QualityForecast(
        timestamp=ts,
        predictions={"24-2000:GODT:CFPP": -18.90, "24-2000:GODT:T90": 344.80,
                      "AVT6:240-350:D15": 848.00, "24-2000:GODT:CloudPoint": -2.90},
        lims_inputs_used={"LIMS:24-2000.Pipeline.D15": 841.0},
        sulfur_ppm=11.0, sulfur_source="ЛИМС", sulfur_age_hours=0.0,
        spec_risk=True, confidence=0.94, confidence_reasons=["Полнота КИП 98%"],
    )
    reliability = ReliabilityAssessment(
        timestamp=ts, severity_index=0.22, severity_class="normal",
        risk_factors=["F63 (откл. 0.41, скорость 0.02)"], allowed_delta_scale=1.0,
    )
    baseline = ScenarioCandidate(
        label="Текущий режим (без изменений)", deltas={}, predicted_sulfur_ppm=11.0,
        predicted_quality=dict(quality.predictions), reliability_severity=0.22,
        throughput_proxy=0.0, energy_proxy=0.0, feasible=True,
    )
    rec = Recommendation(
        timestamp=ts, status="no_feasible_solution",
        state_summary="2026-07-16 10:00: сера 11.00 мг/кг, тяжесть режима normal, полнота данных 98%",
        problem="Риск по сере: лабораторный факт 11.00 мг/кг выше предела 10 мг/кг",
        action="Без изменений — новые уставки не рекомендуются",
        expected_effect="—",
        constraints_checked="Сера ≤ 10 мг/кг: НАРУШЕНО (11.00); тяжесть режима: normal",
        confidence="высокая (94%)",
        explanation="Надёжной рекомендации нет: продукт уже вне спецификации по сере, а в выданных "
                    "материалах нет модели, связывающей уставки с содержанием серы. Система "
                    "отказывается предлагать режим, последствия которого не может проверить.",
        react_trace=[
            "[Мысль] Тик 2026-07-16 10:00: получить состояние процесса.",
            "[Действие] Агент данных → полнота КИП 98%, данные свежие.",
            "[Мысль] Проверить качество продукта.",
            "[Действие] Агент качества → сера 11.00 мг/кг по ЛИМС, выше предела 10 → риск.",
            "[Действие] Агент надёжности → индекс 0.22 (норма).",
            "[Мысль] Можно ли предложить безопасное изменение режима?",
            "[Действие] Агент оптимизации → все варианты отклонены: прогноз серы недоступен.",
            "[Вывод] Допустимого решения нет — сообщаем оператору и объясняем причину.",
        ],
        chosen=None, alternatives=[], sulfur_ppm=11.0, reliability_severity=0.22,
    )
    opt = OptimizationResult(timestamp=ts, candidates=[], infeasible_count=1, baseline=baseline)
    return DemoExample(
        key="sulfur",
        title="Риск по сере — система честно отказывается",
        one_liner="Продукт вне спецификации, а последствия изменений проверить нечем — система отказывается советовать.",
        result=CycleResult(ts, data, quality, reliability, opt, rec),
        series=_series("2026-07-16 10:00",
                        [11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0],
                        [0.22, 0.31, 0.25, 0.42, 0.28, 0.35, 0.26, 0.48, 0.30, 0.27, 0.33, 0.29]),
    )


def _bad_data() -> DemoExample:
    ts = pd.Timestamp("2023-06-01 00:00:00")
    data = DataQualityReport(
        timestamp=ts, tag_coverage=0.71,
        missing_tags=["T1", "P2", "F5", "T6", "F12", "T15", "F19", "T20", "P22", "T33",
                       "F41", "T47", "P50", "F53", "T58", "F62", "T66", "F68", "W70", "T71", "F30"],
        anomalous_tags=["F27", "T37"], dead_tags=["D10"],
        lims_age_hours={"hydrotreat_pipeline": 132.0},
        pak_age_hours={"24-2000:Mg.Sulfur": 120.0, "24-2000:D15": None},
        is_usable=False,
        reasons=["Полнота КИП 71% ниже порога 80%"],
    )
    rec = Recommendation(
        timestamp=ts, status="insufficient_data",
        state_summary="2023-06-01 00:00: полнота КИП 71%, ЛИМС устарел на 132 ч",
        problem="Недостаточно данных для формирования рекомендации",
        action="Без изменений",
        expected_effect="—",
        constraints_checked="Проверка полноты данных не пройдена",
        confidence="низкая",
        explanation="Надёжной рекомендации нет: 21 датчик не передаёт значения, последний "
                    "лабораторный анализ старше пяти суток. Система не строит прогноз на таких "
                    "входных данных и сообщает об этом вместо того, чтобы угадывать.",
        react_trace=[
            "[Мысль] Тик 2023-06-01 00:00: получить состояние процесса и оценить пригодность данных.",
            "[Действие] Агент данных → полнота КИП 71%, пропущен 21 тег, 2 аномалии, ЛИМС 132 ч.",
            "[Вывод] Данных недостаточно — остальные агенты не запускаются.",
        ],
        chosen=None, alternatives=[], sulfur_ppm=None, reliability_severity=None,
    )
    return DemoExample(
        key="baddata",
        title="Плохие данные — система останавливает цикл",
        one_liner="Датчики молчат, лаборатория устарела — система прекращает расчёт и объясняет почему.",
        result=CycleResult(ts, data, None, None, None, rec),
        series=_series("2023-06-01 00:00",
                        [float("nan")] * 12,
                        [float("nan")] * 12),
    )


def examples() -> dict[str, DemoExample]:
    return {e.key: e for e in (_calm(), _sulfur_risk(), _bad_data())}
