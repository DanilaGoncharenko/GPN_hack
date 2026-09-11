"""End-to-end regression test for the 4 required demo scenarios (ТЗ п.6).
Skipped automatically if the competition data isn't present locally, since
it isn't committed to git (see data/raw/.gitkeep and README).
"""
from collections import Counter
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
pytestmark = pytest.mark.skipif(
    not (DATA_DIR / "avt_tags.csv").exists(),
    reason="raw competition data not present in data/raw/ (see README)",
)


def test_four_scenarios_match_their_intended_status():
    from oilhack.harness import scenarios, runner

    env = scenarios.load_environment()

    for key, expected_status in [
        ("stable", "no_action"),
        ("quality_risk", "no_feasible_solution"),
        ("degraded_data", "insufficient_data"),
    ]:
        statuses = Counter(
            t.recommendation.status for t in runner.run_scenario(env, scenarios.SCENARIOS[key])
        )
        assert statuses[expected_status] == sum(statuses.values()), (
            f"scenario {key!r} expected only {expected_status!r}, got {dict(statuses)}"
        )

    full_cycle = Counter(
        t.recommendation.status
        for t in runner.run_scenario(env, scenarios.SCENARIOS["full_cycle"])
    )
    assert full_cycle["action"] > 0, "full_cycle scenario should contain at least one real action"
    assert full_cycle["no_action"] > 0, "full_cycle scenario should also contain calm ticks"
