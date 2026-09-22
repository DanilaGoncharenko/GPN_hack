from oilhack.agents.constraint_agent import check


def test_sulfur_upper_bound_is_hard_constraint():
    result = check({"T6": 360.0}, sulfur_upper=10.1, reliability_severity=0.2)
    assert not result.feasible
    assert any("серы" in item for item in result.violations)


def test_safe_candidate_can_pass():
    result = check({"T6": 360.0}, sulfur_upper=8.5, reliability_severity=0.2)
    assert result.feasible


def test_unknown_control_is_rejected():
    result = check({"UNKNOWN": 1.0}, sulfur_upper=8.5, reliability_severity=0.2)
    assert not result.feasible


def test_critical_reliability_is_rejected():
    result = check({"T6": 360.0}, sulfur_upper=8.5, reliability_severity=0.9)
    assert not result.feasible
