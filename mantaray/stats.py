"""Statistics for the paper: confidence intervals and significance against random chance.

Unit of analysis = one STAR. A star can be in the test set of several seeds, so its
values are first averaged over the seeds it appeared in; stars are then independent.
  * 95% confidence interval: bootstrap over stars (2000 resamples)
  * significance vs chance: one-sided Wilcoxon signed-rank test of (value - chance) per star
  * method comparisons: paired Wilcoxon test on the same stars
"""
import numpy as np
from scipy.stats import wilcoxon


def _star_means(stars, cols):
    d = stars[["tic"] + list(cols)].dropna()
    return d.groupby("tic")[list(cols)].mean()


def _boot_ci(v, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), (n_boot, len(v)))
    m = v[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _wilcoxon(diff, alternative):
    diff = diff[np.abs(diff) > 1e-12]
    if len(diff) < 10:
        return float("nan")
    return float(wilcoxon(diff, alternative=alternative).pvalue)


def vs_chance(stars, value_col, chance_col):
    g = _star_means(stars, (value_col, chance_col))
    if len(g) == 0:
        return {"n_stars": 0}
    v, c = g[value_col].to_numpy(float), g[chance_col].to_numpy(float)
    lo, hi = _boot_ci(v)
    return {"n_stars": int(len(v)), "mean": float(v.mean()), "ci95_low": lo, "ci95_high": hi,
            "chance": float(c.mean()), "enrichment_over_chance": float(v.mean() / c.mean()) if c.mean() > 0 else np.nan,
            "p_value_vs_chance": _wilcoxon(v - c, "greater")}


def paired(stars, col_a, col_b):
    """Is method A better than method B on the same stars? (two-sided)"""
    g = _star_means(stars, (col_a, col_b))
    if len(g) == 0:
        return {"n_stars": 0}
    d = (g[col_a] - g[col_b]).to_numpy(float)
    lo, hi = _boot_ci(d)
    return {"n_stars": int(len(d)), "mean_difference": float(d.mean()), "ci95_low": lo, "ci95_high": hi,
            "p_value": _wilcoxon(d, "two-sided")}
