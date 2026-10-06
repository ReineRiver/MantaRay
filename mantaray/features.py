"""Classic light-curve shape features for the non-deep-learning baseline (random forest).

Computed from the same 200-bin folded, normalised curves the CNN sees (no period: folding
removed it for the CNN too, so the comparison is fair). These are the standard Fourier
decomposition parameters used for variable-star classification (e.g. Simon & Lee 1981)
plus simple shape statistics.
"""
import numpy as np
from scipy.stats import kurtosis, skew

FEATURE_NAMES = [
    "A1", "R21", "R31", "R41", "R51", "sin_phi21", "cos_phi21", "sin_phi31", "cos_phi31",
    "dominant_harmonic", "harmonic_variance_fraction", "skewness", "kurtosis",
    "min", "max", "max_over_abs_min", "frac_below_-2", "frac_above_2", "n_dips",
]


def _n_dips(x, thr=-1.5):
    below = x < thr
    if below.all() or not below.any():
        return float(below.all())
    starts = below & ~np.roll(below, 1)  # circular: count runs
    return float(starts.sum())


def shape_features(X):
    """X: (n, N_BINS) normalised curves -> (n, len(FEATURE_NAMES)) feature matrix."""
    X = np.asarray(X, float)
    n_bins = X.shape[1]
    F = np.fft.rfft(X, axis=1)
    amps = 2 * np.abs(F[:, 1:6]) / n_bins          # harmonics k = 1..5 cycles per fold
    ph = np.angle(F[:, 1:6])
    A1 = amps[:, 0] + 1e-9
    phi21 = ph[:, 1] - 2 * ph[:, 0]                  # phase-origin independent combinations
    phi31 = ph[:, 2] - 3 * ph[:, 0]
    var = X.var(1) + 1e-9
    feats = np.column_stack([
        amps[:, 0], amps[:, 1] / A1, amps[:, 2] / A1, amps[:, 3] / A1, amps[:, 4] / A1,
        np.sin(phi21), np.cos(phi21), np.sin(phi31), np.cos(phi31),
        np.argmax(amps, 1) + 1, (amps ** 2).sum(1) / 2 / var,
        skew(X, axis=1), kurtosis(X, axis=1),
        X.min(1), X.max(1), X.max(1) / (np.abs(X.min(1)) + 1e-9),
        (X < -2).mean(1), (X > 2).mean(1), [_n_dips(x) for x in X],
    ])
    return np.nan_to_num(feats)
