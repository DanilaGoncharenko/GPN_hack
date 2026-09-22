"""Loaders for the raw competition data, producing a small set of typed,
tidy structures that the agents consume. Kept deliberately separate from
agent logic (ТЗ: source parsing/synchronization rules are a distinct
responsibility from quality/reliability/optimization judgement).

Key rule enforced here (ТЗ п.2): synchronization is time-based only. LIMS
and PAK are never joined to telemetry by row position.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import openpyxl
import pandas as pd

from .config import SENTINEL_BAD_VALUE

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_avt_telemetry(path: Path | str = DATA_DIR / "avt_tags.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    drop_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=drop_cols)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def load_242000_telemetry(path: Path | str = DATA_DIR / "242000_tags.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    drop_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=drop_cols)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def mask_sentinel(df: pd.DataFrame, sentinel: float = SENTINEL_BAD_VALUE) -> pd.DataFrame:
    """Return a copy with the historian bad-data code replaced by NaN."""
    return df.mask(df == sentinel)


def _ffill_row(row: list) -> list:
    out = list(row)
    for i in range(1, len(out)):
        if out[i] is None:
            out[i] = out[i - 1]
    return out


def _read_sheet_rows(path: Path | str, sheet: str | None = None) -> list[list]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet] if sheet else wb.worksheets[0]
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    wb.close()
    return rows


def load_lims(path: Path | str = DATA_DIR / "lims.xlsx") -> pd.DataFrame:
    """Tidy long-format LIMS table: group, param, unit, timestamp, value.

    Source layout: row0=install/point/product (merged across a group's
    column block), row1=param name, row2=unit, row3=summary counts (skipped),
    data from row index 4. Each (date_col, value_col) pair is one measured
    series; a pair's presence is marked by a non-null row1 cell at date_col.
    """
    rows = _read_sheet_rows(path)
    group_row = _ffill_row(rows[0])
    param_row = rows[1]
    unit_row = rows[2]
    data_rows = rows[4:]

    records = []
    ncols = len(param_row)
    for idx in range(ncols):
        if param_row[idx] is None or idx + 1 >= ncols:
            continue
        group = group_row[idx]
        param = param_row[idx]
        unit = unit_row[idx]
        for r in data_rows:
            ts = r[idx]
            if ts is None:
                continue
            val = r[idx + 1]
            if val is None or isinstance(val, str):
                continue  # e.g. "Pt Created" placeholder rows
            records.append((group, param, unit, ts, float(val)))

    out = pd.DataFrame(records, columns=["group", "param", "unit", "timestamp", "value"])
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out.sort_values("timestamp").reset_index(drop=True)


def load_pak(path: Path | str = DATA_DIR / "pak.xlsx") -> pd.DataFrame:
    """Tidy long-format PAK table: tag, unit, timestamp, value.

    Source layout: row0=tag id (only at each pair's date column, spacer
    columns are None), row1=unit, data from row index 2.
    """
    rows = _read_sheet_rows(path)
    tag_row = rows[0]
    unit_row = rows[1]
    data_rows = rows[2:]

    records = []
    ncols = len(tag_row)
    for idx in range(ncols):
        if tag_row[idx] is None or idx + 1 >= ncols:
            continue
        tag = tag_row[idx]
        unit = unit_row[idx]
        for r in data_rows:
            ts = r[idx]
            if ts is None:
                continue
            val = r[idx + 1]
            if val is None or isinstance(val, str):
                continue
            records.append((tag, unit, ts, float(val)))

    out = pd.DataFrame(records, columns=["tag", "unit", "timestamp", "value"])
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    return out.sort_values("timestamp").reset_index(drop=True)


@dataclass(frozen=True)
class AsOfResult:
    value: float | None
    timestamp: pd.Timestamp | None
    age_hours: float | None

    @property
    def is_available(self) -> bool:
        return self.value is not None


def asof_lookup(long_df: pd.DataFrame, asof: pd.Timestamp, filters: dict[str, str],
                 contains: dict[str, str] | None = None) -> AsOfResult:
    """Latest row of `long_df` at/before `asof` matching equality `filters`
    and optional substring `contains` filters (e.g. group name fragments).
    Never looks forward in time — this is the only join rule allowed between
    LIMS/PAK and telemetry per ТЗ.
    """
    sub = long_df
    for col, val in filters.items():
        sub = sub[sub[col] == val]
    if contains:
        for col, frag in contains.items():
            sub = sub[sub[col].astype(str).str.contains(frag, na=False)]
    sub = sub[sub["timestamp"] <= asof]
    if sub.empty:
        return AsOfResult(None, None, None)
    last = sub.loc[sub["timestamp"].idxmax()]
    age_hours = (asof - last["timestamp"]).total_seconds() / 3600.0
    return AsOfResult(float(last["value"]), last["timestamp"], age_hours)
