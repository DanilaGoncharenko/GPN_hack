"""One-off historical baselines (mean/std/percentiles per tag, sentinel-
masked) shared by Data Agent (anomaly detection) and Reliability Agent
(deviation-from-envelope proxy). Computed once per telemetry table, not per
tick — avoids re-scanning 189k rows on every harness step.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import SENTINEL_BAD_VALUE, SENTINEL_DEAD_TAG_SHARE


@dataclass(frozen=True)
class TagStats:
    mean: float
    std: float
    p05: float
    p95: float
    dead: bool  # more than SENTINEL_DEAD_TAG_SHARE of history is the sentinel


def compute_tag_stats(raw_df: pd.DataFrame) -> dict[str, TagStats]:
    n = len(raw_df)
    stats: dict[str, TagStats] = {}
    for col in raw_df.columns:
        series = raw_df[col]
        dead_share = float((series == SENTINEL_BAD_VALUE).mean()) if n else 0.0
        clean = series.mask(series == SENTINEL_BAD_VALUE)
        stats[col] = TagStats(
            mean=float(clean.mean()),
            std=float(clean.std() or 0.0),
            p05=float(clean.quantile(0.05)),
            p95=float(clean.quantile(0.95)),
            dead=dead_share > SENTINEL_DEAD_TAG_SHARE,
        )
    return stats
