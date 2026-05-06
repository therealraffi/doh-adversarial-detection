"""
Benign flow-target sampling for the adversarial pipeline.

1) InterpolatedBenignSampler — same core idea as synthetic_realistic_doh_generator.ipynb:
   pick a real benign row, interpolate toward a nearest neighbor in feature space,
   add small Gaussian noise scaled by feature std, clip to quantile bounds.
   This preserves feature relationships better than independent per-feature noise.

2) CorrelatedBenignSampler — explicit multivariate Gaussian in standardized space with a
   Ledoit–Wolf shrunk covariance (stable correlations), then inverse-transform and clip.

3) traffic_params_from_cira_features — map CIRA-style row dict to Scapy session knobs.

4) tabular_correlation_validation_report — compare Pearson correlation matrices of real
   benign vs many synthetic draws (tabular realism check).

PCAP generation cannot reproduce CIRA rows exactly; targets are best-effort drivers.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Columns we try to match (overlap with L2 parquet / realism profile)
BENIGN_TARGET_COLS = [
    "FlowBytesSent",
    "FlowBytesReceived",
    "FlowSentRate",
    "FlowReceivedRate",
    "PacketLengthVariance",
    "PacketLengthStandardDeviation",
    "PacketLengthMean",
    "PacketLengthMedian",
    "PacketLengthMode",
    "PacketLengthCoefficientofVariation",
    "PacketTimeVariance",
    "PacketTimeStandardDeviation",
    "PacketTimeMean",
    "PacketTimeMedian",
    "PacketTimeCoefficientofVariation",
    "ResponseTimeTimeVariance",
    "ResponseTimeTimeStandardDeviation",
    "ResponseTimeTimeMean",
    "ResponseTimeTimeMedian",
    "ResponseTimeTimeCoefficientofVariation",
]


def _load_table(path: str) -> pd.DataFrame:
    path = os.path.abspath(os.path.expanduser(path))
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def load_benign_feature_frame(dataset_path: str, min_rows: int = 200) -> tuple[pd.DataFrame, List[str]]:
    """Return numeric benign-only frame and column list (same order as BENIGN_TARGET_COLS)."""
    real = _load_table(dataset_path)
    real.columns = [c.lstrip("\ufeff") for c in real.columns]
    real.replace([np.inf, -np.inf], np.nan, inplace=True)
    benign = real[real["Label"].astype(str).str.contains("Benign", case=False, na=False)]
    feature_cols = [c for c in BENIGN_TARGET_COLS if c in benign.columns]
    if len(feature_cols) < 5:
        raise ValueError(f"Too few CIRA columns in reference: {feature_cols}")
    X = benign[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if len(X) < min_rows:
        raise ValueError(f"Need at least {min_rows} benign rows; got {len(X)}")
    return X.reset_index(drop=True), feature_cols


def _cov_matrix_sqrt(cov: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    """Symmetric matrix square root L with L @ L.T ~ cov (eigen-clip for PSD)."""
    d = cov.shape[0]
    w, v = np.linalg.eigh(cov + jitter * np.eye(d))
    w = np.maximum(w, jitter)
    return v @ np.diag(np.sqrt(w))


def tabular_correlation_validation_report(
    real_X: pd.DataFrame,
    syn_X: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, Any]:
    """
    Compare Pearson correlation structure: real benign vs synthetic tabular samples.
    """
    Cr = real_X[feature_cols].corr()
    Cs = syn_X[feature_cols].corr()
    diff = Cr.values - Cs.values
    d = diff.shape[0]
    mask_off = ~np.eye(d, dtype=bool)
    off = np.abs(diff[mask_off])
    return {
        "n_features": len(feature_cols),
        "n_real": int(len(real_X)),
        "n_synthetic": int(len(syn_X)),
        "corr_frobenius_real_minus_synth": float(np.linalg.norm(diff, ord="fro")),
        "corr_max_abs_entry": float(np.abs(diff).max()),
        "corr_mean_abs_off_diagonal": float(off.mean()) if off.size else 0.0,
    }


def run_tabular_correlation_validation(
    dataset_path: str,
    sampler: Any,
    n_synthetic: int = 3000,
    report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Draw many tabular samples from sampler and compare corr matrix to real benign."""
    real_X, cols = load_benign_feature_frame(dataset_path, min_rows=200)
    rows: List[Dict[str, float]] = []
    for _ in range(n_synthetic):
        rows.append(sampler.sample())
    syn_X = pd.DataFrame(rows)
    rep = tabular_correlation_validation_report(real_X, syn_X, cols)
    rep["feature_columns"] = cols
    rep["sampler"] = type(sampler).__name__
    if hasattr(sampler, "_lw_shrinkage") and not np.isnan(sampler._lw_shrinkage):
        rep["ledoit_wolf_shrinkage"] = sampler._lw_shrinkage
    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
    return rep


def traffic_params_from_cira_features(feat: Dict[str, float]) -> Dict[str, float]:
    """Map CIRA L2 / detector feature dict to Scapy session parameters."""
    plm = float(feat.get("PacketLengthMean", 130.0))
    pls = float(feat.get("PacketLengthStandardDeviation", 60.0))
    ptm = float(feat.get("PacketTimeMean", 40.0))
    pts = float(feat.get("PacketTimeStandardDeviation", 12.0))
    fbs = float(feat.get("FlowBytesSent", 10_000.0))
    fbr = float(feat.get("FlowBytesReceived", max(fbs * 0.9, 500.0)))

    iat_mean = float(np.clip(ptm, 2.0, 8000.0))
    iat_std = float(np.clip(pts, 0.5, min(3000.0, iat_mean * 0.95)))

    req_wire_mean = float(np.clip(plm, 60.0, 1500.0))
    req_wire_std = float(np.clip(max(pls * 0.75, 3.0), 3.0, 400.0))
    resp_wire_mean = float(np.clip(plm * 1.25 + 20.0, 60.0, 1500.0))
    resp_wire_std = float(np.clip(max(pls * 0.85, 5.0), 5.0, 450.0))

    pair_bytes = max(120.0, req_wire_mean + resp_wire_mean)
    n_pairs = int(np.clip(fbs / pair_bytes, 10, 120))
    cover_scale = float(np.clip(fbr / max(fbs, 1.0), 0.6, 2.5))
    n_cover = int(np.clip(15 * cover_scale, 12, 90))

    return {
        "iat_ms_mean": iat_mean,
        "iat_ms_std": iat_std,
        "req_wire_mean": req_wire_mean,
        "req_wire_std": req_wire_std,
        "resp_wire_mean": resp_wire_mean,
        "resp_wire_std": resp_wire_std,
        "n_pairs_c2": n_pairs,
        "n_pairs_cover": n_cover,
    }


class InterpolatedBenignSampler:
    """
    Interpolate between nearby *benign* feature vectors (notebook method).
    """

    def __init__(
        self,
        dataset_path: str,
        k_neighbors: int = 10,
        noise_scale: float = 0.03,
        lower_q: float = 0.001,
        upper_q: float = 0.999,
        seed: Optional[int] = None,
        min_rows: int = 200,
    ):
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        X, self.feature_cols = load_benign_feature_frame(dataset_path, min_rows=min_rows)

        self.n_training = len(X)
        self._rng = np.random.default_rng(seed)
        self._noise_scale = noise_scale
        self._k = k_neighbors

        self.lower = X.quantile(lower_q)
        self.upper = X.quantile(upper_q)
        self.stds = X.std().replace(0, 1e-9)

        self._X = X.reset_index(drop=True)
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(self._X.values)
        n_n = min(k_neighbors, max(1, len(self._X) - 1))
        self._nn = NearestNeighbors(n_neighbors=min(n_n + 1, len(self._X)))
        self._nn.fit(Xs)
        self._Xs = Xs

    def sample(self) -> Dict[str, float]:
        n = len(self._X)
        anchor_idx = int(self._rng.integers(0, n))
        anchor = self._X.loc[anchor_idx, self.feature_cols].astype(float)

        if n < 2:
            neighbor_idx = anchor_idx
        else:
            dist, ind = self._nn.kneighbors(self._Xs[anchor_idx : anchor_idx + 1])
            options = [int(i) for i in ind[0] if int(i) != anchor_idx]
            if not options:
                neighbor_idx = anchor_idx
            else:
                neighbor_idx = int(self._rng.choice(options))

        neighbor = self._X.loc[neighbor_idx, self.feature_cols].astype(float)
        alpha = float(self._rng.uniform(0.15, 0.85))
        syn = anchor + alpha * (neighbor - anchor)

        noise = self._rng.normal(0.0, self._noise_scale, size=len(self.feature_cols))
        noise = noise * self.stds.to_numpy()
        syn = syn + noise

        syn = syn.clip(lower=self.lower, upper=self.upper)
        for c in self.feature_cols:
            if self._X[c].min() >= 0:
                syn[c] = max(0.0, float(syn[c]))
        return {c: float(syn[c]) for c in self.feature_cols}


class CorrelatedBenignSampler:
    """
    Sample from N(mu, Sigma) in standardized benign space with Sigma = Ledoit-Wolf
    shrunk covariance, then inverse-transform and clip to empirical quantiles.
    """

    def __init__(
        self,
        dataset_path: str,
        lower_q: float = 0.001,
        upper_q: float = 0.999,
        seed: Optional[int] = None,
        min_rows: int = 200,
        cov_jitter: float = 1e-6,
    ):
        from sklearn.covariance import LedoitWolf
        from sklearn.preprocessing import StandardScaler

        X, self.feature_cols = load_benign_feature_frame(dataset_path, min_rows=min_rows)
        self.n_training = len(X)
        self._rng = np.random.default_rng(seed)
        self._nonneg = {c for c in self.feature_cols if float(X[c].min()) >= 0.0}

        self.lower = X.quantile(lower_q)
        self.upper = X.quantile(upper_q)

        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X.values)
        lw = LedoitWolf().fit(Xs)
        cov = lw.covariance_.astype(np.float64)
        self._L = _cov_matrix_sqrt(cov, jitter=cov_jitter)
        self._lw_shrinkage = float(getattr(lw, "shrinkage_", np.nan))

    def sample(self) -> Dict[str, float]:
        d = len(self.feature_cols)
        z = self._rng.standard_normal(d)
        x_s = self._L @ z
        raw = self._scaler.inverse_transform(x_s.reshape(1, -1))[0]
        syn = pd.Series(raw, index=self.feature_cols)
        syn = syn.clip(lower=self.lower, upper=self.upper)
        for c in self._nonneg:
            syn[c] = max(0.0, float(syn[c]))
        return {c: float(syn[c]) for c in self.feature_cols}
