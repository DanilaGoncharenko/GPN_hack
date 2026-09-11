from oilhack.agents.optimization_agent import _dominates, _pareto_front, _score
from oilhack.schemas import ScenarioCandidate


def _cand(label: str) -> ScenarioCandidate:
    return ScenarioCandidate(
        label=label, deltas={}, predicted_sulfur_ppm=5.0, predicted_quality={},
        reliability_severity=0.2, throughput_proxy=0.0, energy_proxy=0.0, feasible=True,
    )


def test_score_rewards_lower_cfpp_and_cloudpoint():
    baseline = {"AVT6:240-350:CFPP": 5.0}
    better = {"AVT6:240-350:CFPP": 3.0}  # lower CFPP is the modelled "better" direction
    quality_score, _, _ = _score(baseline, better, "F30", delta_value=1.0, param_range=10.0)
    assert quality_score == 2.0


def test_score_only_flags_throughput_or_energy_for_the_relevant_tag_set():
    _, throughput, energy = _score({}, {}, "F9", delta_value=2.0, param_range=10.0)
    assert throughput == 0.2  # F9 is a throughput tag
    assert energy == 0.0

    _, throughput2, energy2 = _score({}, {}, "F15", delta_value=2.0, param_range=10.0)
    assert throughput2 == 0.0
    assert energy2 == 0.2  # F15 is an energy tag


def test_dominates_requires_beating_the_epsilon_margin():
    a, b = _cand("a"), _cand("b")
    scores = {"a": (0.05, 0.0, 0.0, 0.0), "b": (0.0, 0.0, 0.0, 0.0)}
    # 0.05 improvement is inside a 0.1 tolerance on that dimension -> not a real win
    assert not _dominates(a, b, scores, eps=(0.1, 0.02, 0.02, 0.01))
    # 0.2 improvement clears the same tolerance
    scores["a"] = (0.2, 0.0, 0.0, 0.0)
    assert _dominates(a, b, scores, eps=(0.1, 0.02, 0.02, 0.01))


def test_pareto_front_lets_a_real_win_crowd_out_a_marginal_candidate():
    baseline, marginal, real_win = _cand("baseline"), _cand("marginal"), _cand("real_win")
    scores = {
        "baseline": (0.0, 0.0, 0.0, 0.0),
        "marginal": (0.02, 0.0, 0.0, 0.0),   # within eps of baseline: ties with it, not dominated
        "real_win": (0.5, 0.0, 0.0, 0.0),    # clears eps over both baseline and marginal
    }
    # Without a real alternative, a marginal candidate ties with baseline (eps-tolerant) --
    # the materiality gate in generate(), not _pareto_front, is what excludes such noise.
    front_without_real_win = _pareto_front([baseline, marginal], scores)
    assert {c.label for c in front_without_real_win} == {"baseline", "marginal"}

    front = _pareto_front([baseline, marginal, real_win], scores)
    labels = {c.label for c in front}
    assert "real_win" in labels
    assert "marginal" not in labels
