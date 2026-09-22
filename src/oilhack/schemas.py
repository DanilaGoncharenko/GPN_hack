"""Typed messages passed between agents — explicit inter-agent protocol."""
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
    sulfur_source: str
    sulfur_age_hours: float | None
    spec_risk: bool
    confidence: float
    confidence_reasons: list[str] = field(default_factory=list)


@dataclass
class ReliabilityAssessment:
    timestamp: pd.Timestamp
    severity_index: float
    severity_class: str
    risk_factors: list[str]
    allowed_delta_scale: float


@dataclass
class ScenarioCandidate:
    label: str
    deltas: dict[str, float]
    predicted_sulfur_ppm: float | None
    predicted_quality: dict[str, float]
    reliability_severity: float
    throughput_proxy: float
    energy_proxy: float
    feasible: bool
    violations: list[str] = field(default_factory=list)
    # Консервативная верхняя граница прогноза используется только как hard gate.
    predicted_sulfur_upper_ppm: float | None = None


@dataclass
class OptimizationResult:
    timestamp: pd.Timestamp
    candidates: list[ScenarioCandidate]
    infeasible_count: int
    baseline: ScenarioCandidate


@dataclass
class ImpactRecord:
    applied_at: pd.Timestamp
    decision: str
    deltas: dict[str, float]
    predicted_reliability_severity: float | None
    predicted_sulfur_ppm: float | None
    monitoring_notes: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    timestamp: pd.Timestamp
    status: str
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
    predicted_sulfur_upper_ppm: float | None = None


@dataclass
class CycleResult:
    """Все выходы одного цикла: dashboard видит каждый агент отдельно."""
    timestamp: pd.Timestamp
    data_report: DataQualityReport
    quality: QualityForecast | None
    reliability: ReliabilityAssessment | None
    optimization: OptimizationResult | None
    recommendation: Recommendation
    sulfur_forecast: object | None = None
    constraint_report: object | None = None
    blending: object | None = None
