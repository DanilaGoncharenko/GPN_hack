"""Агент воздействия: применяет подтверждённую HITL рекомендацию и ведёт
мониторинг фактического отклика на следующих тиках (замыкает цикл ТЗ шаг 8
и заявленный "полный цикл взаимодействия агентов" для демонстрации).
"""
from __future__ import annotations

from ..schemas import ImpactRecord, Recommendation


def apply(recommendation: Recommendation, decision: str) -> ImpactRecord:
    chosen = recommendation.chosen
    return ImpactRecord(
        applied_at=recommendation.timestamp,
        decision=decision,
        deltas=dict(chosen.deltas) if chosen else {},
        predicted_reliability_severity=chosen.reliability_severity if chosen else None,
        predicted_sulfur_ppm=chosen.predicted_sulfur_ppm if chosen else None,
    )


def observe_response(record: ImpactRecord, later: Recommendation) -> str:
    """Compare what was predicted at decision time against a later tick's
    actual readings (ТЗ: замкнуть цикл мониторингом отклика)."""
    notes = []
    if record.predicted_reliability_severity is not None and later.reliability_severity is not None:
        diff = later.reliability_severity - record.predicted_reliability_severity
        notes.append(
            f"Тяжесть режима: прогноз {record.predicted_reliability_severity:.2f}, "
            f"факт {later.reliability_severity:.2f} (расхождение {diff:+.2f})"
        )
    if record.predicted_sulfur_ppm is not None and later.sulfur_ppm is not None:
        diff = later.sulfur_ppm - record.predicted_sulfur_ppm
        notes.append(
            f"Сера: на момент решения {record.predicted_sulfur_ppm:.2f} мг/кг, "
            f"сейчас {later.sulfur_ppm:.2f} мг/кг (расхождение {diff:+.2f})"
        )
    note = "; ".join(notes) if notes else "Нет данных для сравнения отклика"
    record.monitoring_notes.append(note)
    return note
