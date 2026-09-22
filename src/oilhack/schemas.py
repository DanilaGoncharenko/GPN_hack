"""Typed messages passed between agents — the explicit inter-agent protocol
(ТЗ: "взаимодействие между ролями должно быть явно показано в коде").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DataQualityReport:
    timestamp: pd.Timestamp
    tag_coverage: float
    missing_tags: list[str]
    anomalous_tags: list[str]
    dead_tags: list[str]
    lims_age_hours: dict[str, float | None]
    pak_age_hours: dict[str, float | None]
    is_usable: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class QualityForecast:
    timestamp: pd.Timestamp
    predictions: dict[str, float]
    lims_inputs_used: dict[str, float]
    sulfur_ppm: float | None
    sulfur_source: str  # "PAK" | "LIMS" | "unavailable"
    sulfur_age_hours: float | None
    spec_risk: bool
    confidence: float  # 0..1
    confidence_reasons: list[str] = field(default_factory=list)


@dataclass
class ReliabilityAssessment:
    timestamp: pd.Timestamp
    severity_index: float  # 0..1, proxy metric
    severity_class: str  # "normal" | "warn" | "critical"
    risk_factors: list[str]
    allowed_delta_scale: float  # 1.0 normal, shrinks as severity rises


@dataclass
class ScenarioCandidate:
    label: str
    deltas: dict[str, float]  # tag -> absolute new value
    predicted_sulfur_ppm: float | None
    predicted_quality: dict[str, float]
    reliability_severity: float
    throughput_proxy: float
    energy_proxy: float
    feasible: bool
    violations: list[str] = field(default_factory=list)


@dataclass
class OptimizationResult:
    timestamp: pd.Timestamp
    candidates: list[ScenarioCandidate]  # feasible, pareto-sorted, best first
    infeasible_count: int
    baseline: ScenarioCandidate


@dataclass
class ImpactRecord:
    applied_at: pd.Timestamp
    decision: str  # "confirmed" | "rejected" | "alternative_requested"
    deltas: dict[str, float]
    predicted_reliability_severity: float | None
    predicted_sulfur_ppm: float | None
    monitoring_notes: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    timestamp: pd.Timestamp
    status: str  # "no_action" | "action" | "no_feasible_solution" | "insufficient_data"
    state_summary: str
    problem: str
    action: str
    expected_effect: str
    constraints_checked: str
    confidence: str
    explanation: str
    react_trace: list[str] = field(default_factory=list)
    chosen: ScenarioCandidate | None = None
    alternatives: list[ScenarioCandidate] = field(default_factory=list)
    sulfur_ppm: float | None = None
    reliability_severity: float | None = None


@dataclass
class CycleResult:
    """Everything one orchestrator cycle produced — bundled so the dashboard
    can inspect each agent's own output, not just the final recommendation.
    """
    timestamp: pd.Timestamp
    data_report: DataQualityReport
    quality: QualityForecast | None
    reliability: ReliabilityAssessment | None
    optimization: OptimizationResult | None
    recommendation: Recommendation
