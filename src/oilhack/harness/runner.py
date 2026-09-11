"""Reusable scenario-replay loop shared by the CLI and the dashboard."""
from __future__ import annotations

from dataclasses import dataclass

from ..agents import orchestrator
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


def run_scenario(env: Environment, scenario: Scenario):
    """Yield a `Tick` per timestamp in the scenario window."""
    for ts, avt_row, u242000_row, avt_prev, u242000_prev, lims, pak in iter_ticks(env, scenario):
        result = orchestrator.run_cycle(
            ts, avt_row, u242000_row, avt_prev, u242000_prev,
            env.avt_stats, env.u242000_stats, lims, pak,
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
