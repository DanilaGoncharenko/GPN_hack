"""Контрфактический прогноз серы для МАС «Нефтекод».

Модель не является каузальной физической моделью процесса. Это временной
surrogate, который обучается только на исторических данных до заданного
cutoff и используется в двух местах:

1. прогноз серы на горизонте 30/60/120 минут;
2. контрфактическая оценка "что будет, если изменить уставку сейчас".

Для безопасности в Constraint Agent используется консервативная верхняя
граница: p50 + 90-й перцентиль **положительной ошибки** (actual - predicted)
на временной validation. Это односторонний запас именно для верхнего ограничения.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..config import (
    FORECAST_DOWNSAMPLE_STEPS,
    FORECAST_HORIZONS_STEPS,
    FORECAST_LAGS,
    FORECAST_MIN_SAMPLES,
    FORECAST_MAX_ABS_DELTA_PPM,
    FORECAST_TRAINING_DAYS,
    SULFUR_LIMIT_MG_KG,
)
from ..config import CONTROLLABLE_PARAMS

TARGET_COLUMN = "hydro__Q21"
DELTA_TARGET_COLUMN = "sulfur_delta"

# Дополнительные технологические признаки сверх управляемых параметров.
AVT_STATE_TAGS = {
    "F7", "F8", "F9", "D10", "T11", "T13", "F14", "T15", "F30", "F31",
    "F32", "T33", "F34", "F35", "F36", "T37", "T38", "T39", "T47", "T48",
    "T49", "P50", "P51", "P52", "F56", "F60", "T66", "F65",
}
HYDRO_STATE_TAGS = {
    "Q20", "Q21", "P8", "P13", "P24", "T5", "T6", "T11", "T12", "F15",
    "T16", "F17", "F19", "F22", "T23", "F25", "F26", "W7", "T18",
}


@dataclass(frozen=True)
class SulfurPrediction:
    p50: float
    upper: float
    horizon_minutes: int
    calibration_mae: float
    calibration_rmse: float
    calibration_q90: float
    calibration_bias: float
    training_cutoff: pd.Timestamp
    training_samples: int

    @property
    def within_limit(self) -> bool:
        return self.upper <= SULFUR_LIMIT_MG_KG


class SulfurForecastAgent:
    """Хронологический surrogate для Q21 — серы гидроочищенного ДТ."""

    def __init__(
        self,
        training_days: int = FORECAST_TRAINING_DAYS,
        lags: tuple[int, ...] = FORECAST_LAGS,
        horizons: tuple[int, ...] = FORECAST_HORIZONS_STEPS,
    ) -> None:
        self.training_days = training_days
        self.lags = tuple(lags)
        self.horizons = tuple(horizons)
        self.model: HistGradientBoostingRegressor | None = None
        self.feature_names: list[str] = []
        self.horizon_steps = max(self.horizons)
        self.calibration_mae = float("nan")
        self.calibration_rmse = float("nan")
        self.calibration_q90 = float("nan")
        self.calibration_bias = float("nan")
        self.cutoff: pd.Timestamp | None = None
        self.training_samples = 0

    @staticmethod
    def _namespace(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        return df.rename(columns={c: f"{prefix}__{c}" for c in df.columns})

    @classmethod
    def _merge_sources(
        cls,
        avt: pd.DataFrame,
        hydro: pd.DataFrame,
        pak_long: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Собрать единый временной ряд без row-number join."""
        a = cls._namespace(avt, "avt")
        h = cls._namespace(hydro, "hydro")
        base = a.join(h, how="inner").sort_index().copy()

        # Q21 — непрерывный оперативный сигнал. Если в нём есть пропуски,
        # можно дополнить их прошлым ПАК-измерением, но не брать будущее.
        target = base.get(TARGET_COLUMN)
        if target is None:
            raise ValueError("В 242000 telemetry отсутствует Q21 — поточный анализатор серы")
        target = pd.to_numeric(target, errors="coerce")
        target = target.mask(target < 0)

        if pak_long is not None and not pak_long.empty and "tag" in pak_long.columns:
            sub = pak_long[pak_long["tag"].astype(str) == "24-2000:Mg.Sulfur"][
                ["timestamp", "value"]
            ].copy()
            if not sub.empty:
                sub["timestamp"] = pd.to_datetime(sub["timestamp"])
                sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
                sub = sub[sub["value"].notna() & (sub["value"] >= 0)].sort_values("timestamp")
                left = pd.DataFrame(index=base.index).reset_index(names="date").sort_values("date")
                right = sub.rename(columns={"value": "pak_sulfur"})
                aligned = pd.merge_asof(
                    left,
                    right,
                    left_on="date",
                    right_on="timestamp",
                    direction="backward",
                    tolerance=pd.Timedelta(hours=6),
                ).set_index("date")
                target = target.copy()
                target = target.fillna(aligned["pak_sulfur"])

        base[TARGET_COLUMN] = target
        return base

    @staticmethod
    def _feature_columns() -> list[str]:
        cols: list[str] = []
        seen: set[str] = set()
        for tag, param in CONTROLLABLE_PARAMS.items():
            col = f"{param.dataset}__{tag}"
            if col not in seen:
                cols.append(col)
                seen.add(col)
        for tag in sorted(AVT_STATE_TAGS):
            col = f"avt__{tag}"
            if col not in seen:
                cols.append(col)
                seen.add(col)
        for tag in sorted(HYDRO_STATE_TAGS):
            col = f"hydro__{tag}"
            if col not in seen:
                cols.append(col)
                seen.add(col)
        return cols

    def _make_features(
        self,
        df: pd.DataFrame,
        horizon_steps: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        base_cols = [c for c in self._feature_columns() if c in df.columns]
        features = {
            f"{col}__lag{lag}": df[col].shift(lag)
            for col in base_cols
            for lag in self.lags
        }
        x = pd.DataFrame(features, index=df.index)
        # Учим не абсолютный уровень, а изменение серы относительно текущего
        # Q21. Это стабилизирует surrogate при долгих дрейфах базового уровня
        # и делает baseline forecast близким к фактическому текущему состоянию.
        y = df[TARGET_COLUMN].shift(-horizon_steps) - df[TARGET_COLUMN]
        valid = y.notna() & df[TARGET_COLUMN].notna()
        return x.loc[valid], y.loc[valid]

    def fit(
        self,
        avt: pd.DataFrame,
        hydro: pd.DataFrame,
        pak_long: pd.DataFrame | None,
        cutoff: pd.Timestamp,
    ) -> "SulfurForecastAgent":
        cutoff = pd.Timestamp(cutoff)
        df = self._merge_sources(avt, hydro, pak_long)
        start = cutoff - pd.Timedelta(days=self.training_days)
        df = df.loc[(df.index >= start) & (df.index <= cutoff)].copy()
        # 10-min telemetry -> 30-min training cadence, reducing cost and avoiding
        # an artificial dominance of very-near-duplicate rows.
        df = df.iloc[::FORECAST_DOWNSAMPLE_STEPS].copy()

        trained_best = None
        for horizon in self.horizons:
            x, y = self._make_features(df, horizon)
            if len(x) < FORECAST_MIN_SAMPLES:
                continue
            i1, i2 = int(len(x) * 0.70), int(len(x) * 0.85)
            x_train, y_train = x.iloc[:i1], y.iloc[:i1]
            x_val, y_val = x.iloc[i1:i2], y.iloc[i1:i2]
            model = HistGradientBoostingRegressor(
                max_iter=40,
                learning_rate=0.06,
                max_leaf_nodes=7,
                l2_regularization=1.0,
                random_state=42,
            )
            model.fit(x_train, y_train)
            pred = model.predict(x_val)
            mae = float(mean_absolute_error(y_val, pred))
            rmse = float(mean_squared_error(y_val, pred) ** 0.5)
            residual = y_val.to_numpy() - pred
            bias = float(np.mean(residual))
            calibrated_residual = residual - bias
            q90 = float(max(0.0, np.quantile(calibrated_residual, 0.90)))
            candidate = (mae, rmse, horizon, model, list(x.columns), q90, bias, len(x))
            if trained_best is None or candidate[0] < trained_best[0]:
                trained_best = candidate

        if trained_best is None:
            raise RuntimeError(
                "Не удалось обучить прогноз серы: недостаточно валидных исторических Q21/ПАК данных"
            )

        (
            self.calibration_mae,
            self.calibration_rmse,
            self.horizon_steps,
            self.model,
            self.feature_names,
            self.calibration_q90,
            self.calibration_bias,
            self.training_samples,
        ) = trained_best
        self.cutoff = cutoff
        return self

    def _feature_row(
        self,
        history_avt: pd.DataFrame,
        history_hydro: pd.DataFrame,
        changes: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("SulfurForecastAgent не обучен")
        a = self._namespace(history_avt, "avt")
        h = self._namespace(history_hydro, "hydro")
        merged = a.join(h, how="inner").sort_index()
        merged = merged.tail(max(self.lags) + 1).copy()
        if merged.empty:
            raise ValueError("Нет истории для контрфактического прогноза серы")

        if changes:
            for key, value in changes.items():
                if key in merged.columns:
                    merged.loc[merged.index[-1], key] = float(value)

        values = {}
        for feature in self.feature_names:
            col, lag_s = feature.rsplit("__lag", 1)
            lag = int(lag_s)
            if col not in merged.columns or len(merged) <= lag:
                values[feature] = np.nan
            else:
                values[feature] = pd.to_numeric(merged[col].iloc[-1 - lag], errors="coerce")
        return pd.DataFrame([values], columns=self.feature_names)

    def _base_feature_row(self, history_avt: pd.DataFrame, history_hydro: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("SulfurForecastAgent не обучен")
        a = self._namespace(history_avt, "avt")
        h = self._namespace(history_hydro, "hydro")
        merged = a.join(h, how="inner").sort_index().tail(max(self.lags) + 1)
        if merged.empty:
            raise ValueError("Нет истории для контрфактического прогноза серы")
        values = {}
        for feature in self.feature_names:
            col, lag_s = feature.rsplit("__lag", 1)
            lag = int(lag_s)
            if col not in merged.columns or len(merged) <= lag:
                values[feature] = np.nan
            else:
                values[feature] = pd.to_numeric(merged[col].iloc[-1 - lag], errors="coerce")
        return pd.Series(values, index=self.feature_names, dtype=float)

    def predict_many(
        self,
        history_avt: pd.DataFrame,
        history_hydro: pd.DataFrame,
        changes_list: list[dict[str, float]],
        anchor_sulfur: float | None = None,
    ) -> list[SulfurPrediction]:
        """Векторизованный прогноз нескольких контрфактов за один model.predict."""
        if self.model is None or self.cutoff is None:
            raise RuntimeError("SulfurForecastAgent не обучен")
        base = self._base_feature_row(history_avt, history_hydro)
        rows = []
        for changes in changes_list:
            row = base.copy()
            for key, value in changes.items():
                feature = f"{key}__lag0"
                if feature in row.index:
                    row[feature] = float(value)
            rows.append(row)
        x = pd.DataFrame(rows, columns=self.feature_names)
        delta_pred = self.model.predict(x)
        current_q21 = pd.to_numeric(history_hydro.iloc[-1].get("Q21"), errors="coerce")
        anchor = float(anchor_sulfur) if anchor_sulfur is not None else current_q21
        if pd.isna(anchor):
            raise ValueError("Нет текущей контрольной серы или Q21 для delta-прогноза")
        calibrated_delta = np.clip(delta_pred + self.calibration_bias, -FORECAST_MAX_ABS_DELTA_PPM, FORECAST_MAX_ABS_DELTA_PPM)
        pred = np.maximum(0.0, float(anchor) + calibrated_delta)
        upper = np.maximum(pred, pred + self.calibration_q90)
        return [
            SulfurPrediction(
                p50=float(p50),
                upper=float(up),
                horizon_minutes=self.horizon_steps * 10,
                calibration_mae=self.calibration_mae,
                calibration_rmse=self.calibration_rmse,
                calibration_q90=self.calibration_q90,
                calibration_bias=self.calibration_bias,
                training_cutoff=self.cutoff,
                training_samples=self.training_samples,
            )
            for p50, up in zip(pred, upper)
        ]

    def predict(
        self,
        history_avt: pd.DataFrame,
        history_hydro: pd.DataFrame,
        changes: dict[str, float] | None = None,
        anchor_sulfur: float | None = None,
    ) -> SulfurPrediction:
        if self.cutoff is None:
            raise RuntimeError("SulfurForecastAgent не обучен")
        x = self._feature_row(history_avt, history_hydro, changes)
        delta_pred = float(self.model.predict(x)[0])
        current_q21 = pd.to_numeric(history_hydro.iloc[-1].get("Q21"), errors="coerce")
        anchor = float(anchor_sulfur) if anchor_sulfur is not None else current_q21
        if pd.isna(anchor):
            raise ValueError("Нет текущей контрольной серы или Q21 для delta-прогноза")
        calibrated_delta = float(np.clip(delta_pred + self.calibration_bias, -FORECAST_MAX_ABS_DELTA_PPM, FORECAST_MAX_ABS_DELTA_PPM))
        p50 = max(0.0, float(anchor) + calibrated_delta)
        upper = max(p50, p50 + self.calibration_q90)
        return SulfurPrediction(
            p50=p50,
            upper=upper,
            horizon_minutes=self.horizon_steps * 10,
            calibration_mae=self.calibration_mae,
            calibration_rmse=self.calibration_rmse,
            calibration_q90=self.calibration_q90,
            calibration_bias=self.calibration_bias,
            training_cutoff=self.cutoff,
            training_samples=self.training_samples,
        )
