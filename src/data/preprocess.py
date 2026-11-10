"""Feature transformation: V1-V28 + log1p(Amount), standardised with training-only statistics.

The transformer stores its parameters as plain JSON (no pickles), so it loads identically on
any scikit-learn / Python version, including Google Colab. ``inverse_transform`` maps model-space
vectors (e.g. VAE samples) back to the original V1-V28 + Amount scale for synthetic-data export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load import AMOUNT_COL, PCA_FEATURES, RAW_INPUT_COLUMNS
from src.utils.io import load_json, save_json

MODEL_FEATURES = [*PCA_FEATURES, "log1p_Amount"]


def _engineer(df: pd.DataFrame) -> np.ndarray:
    """Raw columns -> unscaled model features (n, 29). Time and Class are never touched."""
    missing = [c for c in RAW_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing input columns: {missing}")
    if (df[AMOUNT_COL] < 0).any():
        raise ValueError("Amount must be non-negative for log1p")
    out = np.empty((len(df), len(MODEL_FEATURES)), dtype=np.float64)
    out[:, :28] = df[PCA_FEATURES].to_numpy(dtype=np.float64)
    out[:, 28] = np.log1p(df[AMOUNT_COL].to_numpy(dtype=np.float64))
    return out


@dataclass
class FeatureTransformer:
    feature_names: list[str] = field(default_factory=lambda: list(MODEL_FEATURES))
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    n_fit_rows: int = 0
    fitted_on: str = ""
    amount_transform: str = "log1p"   # "log1p" (default) or "none" (raw Amount; ablation only)

    def __post_init__(self) -> None:
        if self.amount_transform not in ("log1p", "none"):
            raise ValueError("amount_transform must be 'log1p' or 'none'")
        if self.amount_transform == "none" and self.feature_names == MODEL_FEATURES:
            self.feature_names = [*PCA_FEATURES, AMOUNT_COL]

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        feats = _engineer(df)
        if self.amount_transform == "none":
            feats[:, 28] = df[AMOUNT_COL].to_numpy(dtype=np.float64)
        return feats

    def fit(self, df: pd.DataFrame, fitted_on: str) -> "FeatureTransformer":
        feats = self._features(df)
        self.mean_ = feats.mean(axis=0)
        std = feats.std(axis=0)  # population std (ddof=0), same as sklearn StandardScaler
        if np.any(std == 0):
            raise ValueError("Zero-variance feature in fit data")
        self.scale_ = std
        self.n_fit_rows = len(df)
        self.fitted_on = fitted_on
        return self

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FeatureTransformer is not fitted")

    def transform(self, df: pd.DataFrame, dtype=np.float64) -> np.ndarray:
        self._check_fitted()
        return ((self._features(df) - self.mean_) / self.scale_).astype(dtype)

    def inverse_transform(self, Z: np.ndarray, round_amount: bool = True) -> pd.DataFrame:
        """Model space -> DataFrame with V1-V28 and Amount (Amount clipped at 0, 2 decimals)."""
        self._check_fitted()
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim != 2 or Z.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected shape (n, {len(self.feature_names)}), got {Z.shape}")
        feats = Z * self.scale_ + self.mean_
        out = pd.DataFrame(feats[:, :28], columns=PCA_FEATURES)
        amount = np.clip(np.expm1(feats[:, 28]) if self.amount_transform == "log1p" else feats[:, 28], 0.0, None)
        out[AMOUNT_COL] = np.round(amount, 2) if round_amount else amount
        return out

    def to_dict(self) -> dict:
        self._check_fitted()
        return {
            "type": "FeatureTransformer",
            "feature_names": self.feature_names,
            "amount_transform": self.amount_transform,
            "scaler": "standard (population std, ddof=0)",
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "n_fit_rows": self.n_fit_rows,
            "fitted_on": self.fitted_on,
        }

    def save(self, path: Path) -> None:
        save_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: Path) -> "FeatureTransformer":
        d = load_json(path)
        return cls(feature_names=d["feature_names"], mean_=np.asarray(d["mean"]), scale_=np.asarray(d["scale"]),
                   n_fit_rows=d.get("n_fit_rows", 0), fitted_on=d.get("fitted_on", ""),
                   amount_transform=d.get("amount_transform", "log1p"))
