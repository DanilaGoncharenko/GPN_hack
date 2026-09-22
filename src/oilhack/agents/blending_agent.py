"""Явная роль блендинга.

В приложенном пакете нет подтверждённого набора компонентных потоков/
долей коммерческого блендинга. Поэтому агент не выдумывает управление, а
явно сообщает, что этот контур недоступен в рамках выданных данных.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlendingAssessment:
    available: bool
    feasible: bool
    reason: str
    fractions: dict[str, float]


def assess() -> BlendingAssessment:
    return BlendingAssessment(
        available=False,
        feasible=False,
        reason=(
            "В выданном пакете нет подтверждённых тегов/составов компонентов "
            "товарного блендинга; модуль не выдаёт фиктивные доли."
        ),
        fractions={},
    )
