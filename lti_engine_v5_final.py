"""
Latent Transportable Interventions (LTI) Engine - v5 Final
Pure Mathematical Engine for One-Shot Causal Direction Discovery.
"""

import numpy as np
import pandas as pd
import warnings
from scipy.signal import savgol_filter
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

warnings.filterwarnings("ignore")

def preprocess_series(series: pd.Series) -> pd.Series:
    """Apply Savitzky-Golay filter to smooth noise while preserving morphology."""
    w = min(11, len(series) if len(series) % 2 != 0 else len(series) - 1)
    if w < 3:
        return series
    smoothed = savgol_filter(series.dropna().values, window_length=w, polyorder=2)
    return pd.Series(smoothed, index=series.index)

def classify_role(series: pd.Series) -> str:
    """
    Classify variable role into Effort (E), Flow (F), or Quantity (Q)
    based on derivative sparsity and zero-crossings.
    """
    s = preprocess_series(series)
    grad = np.gradient(s.values)
    
    epsilon = 1e-9
    sparsity = np.max(np.abs(grad)) / (np.mean(np.abs(grad)) + epsilon)
    
    # Zero crossings to detect oscillatory flow
    zero_crossings = np.where(np.diff(np.signbit(grad)))[0]
    has_reversals = len(zero_crossings) > 0
    
    if sparsity > 3 and has_reversals:
        return "F"  # Flow (Pulse/Impulse)
    elif sparsity > 3:
        return "E"  # Effort (Step function)
    else:
        return "Q"  # Quantity (Smooth accumulation)

def hysteresis_area(x: pd.Series, y: pd.Series) -> float:
    """
    Computes Phase-Space Hysteresis Area using the Shoelace formula.
    Positive Area -> X causes Y (Counter-clockwise)
    Negative Area -> Y causes X (Clockwise)
    """
    # Min-Max Normalization
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-9)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-9)
    
    x_val = x_norm.values
    y_val = y_norm.values
    
    # Shoelace Formula for signed area
    area = 0.5 * np.sum(x_val[:-1] * y_val[1:] - x_val[1:] * y_val[:-1])
    return area

def _granger_pvalue(cause: pd.Series, effect: pd.Series, max_lag: int) -> float:
    try:
        data = pd.concat([effect, cause], axis=1).dropna()
        if len(data) < max_lag * 4:
            return 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        return min(res[lag][0]["ssr_ftest"][1] for lag in res)
    except Exception:
        return 1.0

def _transfer_entropy(x: np.ndarray, y: np.ndarray, n_bins: int = 6, lag: int = 1) -> float:
    def _bin(arr):
        edges = np.percentile(arr, np.linspace(0, 100, n_bins + 1)[1:-1])
        return np.digitize(arr, edges).astype(np.int32)

    x_d, y_t, y_l = _bin(x[:-lag]), _bin(y[lag:]), _bin(y[:-lag])

    def _entropy(a, b=None, c=None):
        if b is None and c is None:
            _, cnt = np.unique(a, return_counts=True)
        elif c is None:
            idx = a * (n_bins + 1) + b
            _, cnt = np.unique(idx, return_counts=True)
        else:
            idx = a * (n_bins + 1)**2 + b * (n_bins + 1) + c
            _, cnt = np.unique(idx, return_counts=True)
        p = cnt / cnt.sum()
        return -np.sum(p * np.log2(p + 1e-12))

    try:
        te = _entropy(y_t, y_l) - _entropy(y_l) - _entropy(y_t, y_l, x_d) + _entropy(y_l, x_d)
        return float(np.clip(te / (_entropy(y_t) + 1e-9), 0.0, 1.0))
    except Exception:
        return 0.0

def _peak_lag_correlation(x: pd.Series, y: pd.Series, max_lag: int) -> tuple[float, int]:
    best_corr, best_lag = 0.0, 1
    for lag in range(1, max_lag + 1):
        c = x.shift(lag).corr(y)
        if not np.isnan(c) and abs(c) > best_corr:
            best_corr, best_lag = abs(c), lag
    return best_corr, best_lag

def robust_causal_direction(name_a: str, series_a: pd.Series, name_b: str, series_b: pd.Series) -> tuple[str, float, str]:
    """
    The main LTI Engine function.
    Returns: (Direction, Confidence Score, Method Used)
    """
    aligned = pd.concat([series_a.rename(name_a), series_b.rename(name_b)], axis=1, join="inner").dropna()

    if len(aligned) < 30:
        return "UNDETERMINED", 0.0, "insufficient_data"

    def _make_stationary(s: pd.Series) -> pd.Series:
        p = adfuller(s.dropna(), autolag="AIC")[1]
        return s.diff().dropna() if p > 0.05 else s

    sa, sb = _make_stationary(aligned[name_a]), _make_stationary(aligned[name_b])
    aligned2 = pd.concat([sa, sb], axis=1, join="inner").dropna()
    
    if len(aligned2) < 24:
        return "UNDETERMINED", 0.0, "insufficient_obs"

    max_lag = min(12, len(aligned2) // 8)
    a_arr, b_arr = aligned2[name_a].values, aligned2[name_b].values

    # 1. Granger Causality
    score_granger_ab = max(0.0, 1.0 - _granger_pvalue(aligned2[name_a], aligned2[name_b], max_lag))
    score_granger_ba = max(0.0, 1.0 - _granger_pvalue(aligned2[name_b], aligned2[name_a], max_lag))

    # 2. Transfer Entropy
    te_ab = _transfer_entropy(a_arr, b_arr, lag=1)
    te_ba = _transfer_entropy(b_arr, a_arr, lag=1)

    # 3. Lag-Correlation
    corr_ab, _ = _peak_lag_correlation(aligned2[name_a], aligned2[name_b], max_lag)
    corr_ba, _ = _peak_lag_correlation(aligned2[name_b], aligned2[name_a], max_lag)

    # Weighted Evidence Aggregation
    evidence_ab = (score_granger_ab * 0.45) + (te_ab * 0.35) + (corr_ab * 0.20)
    evidence_ba = (score_granger_ba * 0.45) + (te_ba * 0.35) + (corr_ba * 0.20)

    delta = evidence_ab - evidence_ba

    # 4. Tie-Breaker: Phase-Space Hysteresis
    if abs(delta) < 0.05:
        h_area = hysteresis_area(aligned[name_a], aligned[name_b])
        confidence = max(evidence_ab, evidence_ba) + min(abs(h_area), 0.2) # Boost confidence slightly with hysteresis
        
        if h_area > 0.01:
            return f"{name_a} → {name_b}", confidence, "hysteresis_tie_breaker"
        elif h_area < -0.01:
            return f"{name_b} → {name_a}", confidence, "hysteresis_tie_breaker"
        else:
            return "BIDIRECTIONAL", max(evidence_ab, evidence_ba), "ensemble_tie"

    if delta > 0:
        return f"{name_a} → {name_b}", evidence_ab, "ensemble_empirical"
    else:
        return f"{name_b} → {name_a}", evidence_ba, "ensemble_empirical"