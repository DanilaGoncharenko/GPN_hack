"""Жёсткий Safety Gate МАС.

Никакой Pareto/экономический score не может компенсировать нарушение
жёсткого ограничения. Входом агент получает уже рассчитанный контрфактический
верхний прогноз серы и прогноз тяжести режима.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import CONTROLLABLE_PARAMS, RELIABILITY_CRITICAL_THRESHOLD, SULFUR_LIMIT_MG_KG


@dataclass(frozen=True)
class ConstraintReport:
    feasible: bool
    violations: list[str]


def check(
    deltas: dict[str, float],
    sulfur_upper: float | None,
    reliability_severity: float,
    data_usable: bool = True,
) -> ConstraintReport:
    violations: list[str] = []

    if not data_usable:
        violations.append("Входные данные признаны непригодными")

    if sulfur_upper is None:
        violations.append("Нет консервативного контрфактического прогноза серы")
    elif sulfur_upper > SULFUR_LIMIT_MG_KG:
        violations.append(
            f"Верхняя граница прогноза серы {sulfur_upper:.2f} > "
            f"{SULFUR_LIMIT_MG_KG:.1f} мг/кг"
        )

    if reliability_severity >= RELIABILITY_CRITICAL_THRESHOLD:
        violations.append("Режим критический по индексу надёжности")

    for tag, value in deltas.items():
        param = CONTROLLABLE_PARAMS.get(tag)
        if param is None:
            violations.append(f"Неизвестный управляющий параметр: {tag}")
            continue
        if not (param.lo <= value <= param.hi):
            violations.append(
                f"{tag}={value:.3f} вне модельного диапазона "
                f"[{param.lo:.3f}, {param.hi:.3f}]"
            )

    return ConstraintReport(feasible=not violations, violations=violations)
