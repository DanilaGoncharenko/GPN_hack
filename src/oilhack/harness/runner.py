"""Reusable scenario-replay loop shared by CLI and dashboard."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..agents import orchestrator
from ..agents.sulfur_forecast_agent import SulfurForecastAgent
from ..schemas import CycleResult, Recommendation
from .scenarios import Environment, Scenario, iter_ticks


@dataclass
class Tick:
    timestamp: object
    avt_row: dict
    u242000_row: dict
    result: CycleResult

    @property
    def recommendation(self) -> Recommendation:
        return self.result.recommendation


def _fit_forecaster(env: Environment, scenario: Scenario) -> SulfurForecastAgent | None:
    """Train without looking beyond the scenario start (no future leakage)."""
    cutoff = pd.Timestamp(scenario.start) - pd.Timedelta(minutes=10)
    try:
        return SulfurForecastAgent().fit(env.avt, env.u242000, env.pak, cutoff)
    except (ValueError, RuntimeError):
        return None


def run_scenario(env: Environment, scenario: Scenario):
    """Yield one Tick per 10-min timestamp in the scenario window."""
    # В сценарии намеренно испорченных данных Data Agent всё равно должен
    # остановить цикл; обучать surrogate там бессмысленно и дорого.
    forecaster = None if scenario.inject_degraded else _fit_forecaster(env, scenario)
    max_history = max(forecaster.lags) + 1 if forecaster is not None else 13

    for ts, avt_row, u242000_row, avt_prev, u242000_prev, lims, pak in iter_ticks(env, scenario):
        history_avt = env.avt.loc[:ts].tail(max_history)
        history_hydro = env.u242000.loc[:ts].tail(max_history)
        result = orchestrator.run_cycle(
            ts,
            avt_row,
            u242000_row,
            avt_prev,
            u242000_prev,
            env.avt_stats,
            env.u242000_stats,
            lims,
            pak,
            sulfur_forecaster=forecaster,
            history_avt=history_avt,
            history_hydro=history_hydro,
        )
        yield Tick(ts, avt_row, u242000_row, result)


def main() -> None:
    import argparse

    from .scenarios import SCENARIOS, load_environment

    parser = argparse.ArgumentParser(description="Прогон харнесса без дашборда (CLI-демо).")
    parser.add_argument("scenario", choices=sorted(SCENARIOS), help="ключ сценария")
    parser.add_argument("--limit", type=int, default=12, help="сколько тиков показать")
    args = parser.parse_args()

    print("Загрузка данных...")
    env = load_environment()
    scenario = SCENARIOS[args.scenario]
    print(f"Сценарий: {scenario.title} [{scenario.start} .. {scenario.end}]\n")

    for i, tick in enumerate(run_scenario(env, scenario)):
        if i >= args.limit:
            break
        rec = tick.recommendation
        print(f"--- {tick.timestamp} [{rec.status}] ---")
        print(f"Состояние: {rec.state_summary}")
        print(f"Проблема: {rec.problem}")
        print(f"Действие: {rec.action}")
        print(f"Ожидаемый эффект: {rec.expected_effect}")
        print(f"Ограничения: {rec.constraints_checked}")
        print(f"Уверенность: {rec.confidence}")
        print(f"Объяснение: {rec.explanation}")
        for line in rec.react_trace:
            print(f"  {line}")
        print()


if __name__ == "__main__":
    main()
