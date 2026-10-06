"""Turn one raw TESS 2-minute light curve into one row of the dataset.

No network code here, so everything can be tested offline. The download side lives
in build_tess_dataset.py.

Pipeline, identical for every class (to avoid giving the CNN a processing shortcut):
  1. clean     : drop NaNs, convert to relative flux, clip only UPWARD outliers
                 (flares, cosmic rays); dips are kept because eclipses are dips
  2. period    : EB    -> catalogue period (Lomb-Scargle often finds half the EB period)
                 DSCT  -> strongest Lomb-Scargle peak (must be 5-80 cycles/day)
                 Noise -> a RANDOM period, log-uniform over 0.0125-12.5 days (the whole range of the
                          trained classes): a quiet star has no real period, so the period must not
                          tell the model "quiet" (seeded by the TIC number, so it is reproducible)
  3. fold+bin  : phase-fold on that period, average into N_BINS phase bins
  4. checks    : EB    -> primary eclipse detected at >= 5 sigma; eclipse mask built
                 DSCT  -> dominant peak S/N >= 10 inside the delta Scuti range
                 ROT   -> dominant peak S/N >= 10 between 0.08 and 5 cycles/day (spot rotation)
                 Noise -> no peak with S/N >= 4 (Breger et al. 1993 criterion) at 0.5-100 c/d, no peak
                          that would pass the rotator selection (S/N >= 10 at 0.08-5 c/d), no deep dips

Never-seen test types (not used for training; see UNSEEN_KINDS): folded on the catalogue
period when there is one (RR Lyrae, ellipsoidal, gamma Dor), on a box-search (BLS) period
for dip/transit stars, otherwise on the strongest periodogram peak.
"""
import numpy as np
from astropy.timeseries import LombScargle

from mantaray.data import N_BINS

FREQ_MIN, FREQ_MAX = 0.5, 100.0      # cycles/day searched by the periodogram (2-min Nyquist = 360)
DSCT_FMIN, DSCT_FMAX = 5.0, 80.0      # delta Scuti pressure-mode range
DSCT_MIN_SNR = 10.0
QUIET_MAX_SNR = 4.0
EB_MIN_SIGMA = 5.0                    # primary eclipse depth in units of binned noise
EB_PERIOD_RANGE = (0.2, 10.0)         # days: short-period EBs, several orbits per 27-day sector
DETACHED_MAX_MASK = 0.30              # eclipse mask covering < 30% of phase = detached-like
ROT_FMIN, ROT_FMAX = 0.08, 5.0        # spot rotation: periods 0.2-12.5 days
ROT_MIN_SNR = 10.0
UNSEEN_KINDS = ("GDOR", "RR", "ELL", "DIP")   # types the classifier is never trained on
UNSEEN_MIN_SNR = 4.0
QUIET_PERIOD_RANGE = (1 / DSCT_FMAX, 1 / ROT_FMIN)   # days: random fold period of quiet stars, 0.0125-12.5 d


# ====================================================================== basics

def to_float(q):
    """astropy Quantity / masked array / ndarray -> float ndarray with NaN for masked values."""
    v = getattr(q, "value", q)
    if hasattr(v, "filled"):
        v = v.filled(np.nan)
    return np.asarray(v, dtype=float)


def robust_std(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def point_noise(f):
    """Per-point white noise from point-to-point differences (insensitive to slow signals)."""
    return robust_std(np.diff(f)) / np.sqrt(2)


def clean(time, flux, return_index=False):
    """Relative flux with NaNs and upward outliers removed.

    return_index=True also returns the indices (into the input arrays) of the kept
    points, so other columns (e.g. centroids) can be filtered identically.
    """
    t, f = to_float(time), to_float(flux)
    idx = np.flatnonzero(np.isfinite(t) & np.isfinite(f))
    if len(idx) < 1000:
        return (None, None, None) if return_index else (None, None)
    idx = idx[np.argsort(t[idx])]
    t, f = t[idx], f[idx]
    f = f / np.median(f) - 1.0
    keep = f < np.median(f) + 5 * max(robust_std(f), point_noise(f) * 3)
    if return_index:
        return t[keep], f[keep], idx[keep]
    return t[keep], f[keep]


# ====================================================================== periodogram

def amplitude_spectrum(t, f, fmin=FREQ_MIN, fmax=FREQ_MAX, oversample=5):
    T = t.max() - t.min()
    freq = np.arange(fmin, fmax, 1.0 / (oversample * T))
    power = LombScargle(t, f, normalization="psd").power(freq, method="fast")
    return freq, np.sqrt(4 * np.clip(power, 0, None) / len(t))


def top_peak(freq, amp, window=5.0):
    """Highest peak and its S/N against the mean amplitude within +/- window cycles/day (Breger et al. 1993)."""
    i = int(np.argmax(amp))
    near = np.abs(freq - freq[i]) < window
    return float(freq[i]), float(amp[i]), float(amp[i] / np.mean(amp[near]))


# ====================================================================== fold + bin

def fold_bin(t, f, period, n=N_BINS, t0=None):
    """Mean flux in n phase bins (t0 = first timestamp unless given). Empty bins filled circularly."""
    phase = ((t - (t[0] if t0 is None else t0)) / period) % 1.0
    idx = np.minimum((phase * n).astype(int), n - 1)
    cnt = np.bincount(idx, minlength=n)
    tot = np.bincount(idx, weights=f, minlength=n)
    b = np.full(n, np.nan)
    good = cnt > 0
    b[good] = tot[good] / cnt[good]
    if not good.all():
        x = np.flatnonzero(good)
        b[~good] = np.interp(np.flatnonzero(~good), np.r_[x - n, x, x + n], np.r_[b[x], b[x], b[x]])
    return b, cnt


def _grow(d, i, thr):
    """Contiguous circular run of bins around i where d > thr."""
    n, m = len(d), np.zeros(len(d), bool)
    m[i] = True
    for step in (1, -1):
        j = i
        for _ in range(n):
            j = (j + step) % n
            if d[j] > thr and not m[j]:
                m[j] = True
            else:
                break
    return m


def eclipse_mask(b, sigma_bin, min_sigma=EB_MIN_SIGMA):
    """Find primary (and secondary) eclipse in a binned folded curve.

    Eclipse = contiguous run of bins around the minimum that are deeper than
    max(10% of the eclipse depth, 3 sigma). Returns (mask, info) or (None, reason).
    """
    d = np.median(b) - b                          # positive inside dips
    i1 = int(np.argmax(d))
    depth1 = d[i1]
    if depth1 < min_sigma * sigma_bin:
        return None, f"no primary eclipse >= {min_sigma:.0f} sigma"
    m1 = _grow(d, i1, max(0.1 * depth1, 3 * sigma_bin))
    # secondary: deepest point away from the primary (primary region widened by 0.05 in phase)
    excl = m1.copy()
    for s in range(1, int(0.05 * len(b)) + 1):
        excl |= np.roll(m1, s) | np.roll(m1, -s)
    d2 = np.where(excl, -np.inf, d)
    i2 = int(np.argmax(d2))
    depth2 = d2[i2]
    m2 = np.zeros_like(m1)
    if np.isfinite(depth2) and depth2 >= min_sigma * sigma_bin:
        m2 = _grow(d, i2, max(0.1 * depth2, 3 * sigma_bin)) & ~m1
    mask = m1 | m2
    info = {"depth1_sigma": float(depth1 / sigma_bin), "depth2_sigma": float(max(depth2, 0) / sigma_bin),
            "primary_bin": i1, "mask_fraction": float(mask.mean())}
    return mask, info


# ====================================================================== one star

def bls_period(t, f, pmin=0.5, pmax=12.0):
    """Box Least Squares period search for transit/eclipse-like dips. Returns (period, depth S/N)."""
    from astropy.timeseries import BoxLeastSquares
    pmax = min(pmax, (t.max() - t.min()) / 2)
    periods = np.exp(np.linspace(np.log(pmin), np.log(pmax), 3000))
    dy = np.full_like(f, point_noise(f))   # per-point noise, so depth S/N is meaningful
    res = BoxLeastSquares(t, f, dy=dy).power(periods, [0.04, 0.08, 0.15], objective="snr")
    i = int(np.argmax(res.power))
    return float(res.period[i]), float(res.depth_snr[i])


def quiet_period(t, f, seed, tries=20):
    """Random fold period for a quiet star: log-uniform over QUIET_PERIOD_RANGE, at least 2 cycles in the
    light curve. A period that leaves phase bins nearly empty (e.g. close to a multiple of the 2-min
    cadence) is redrawn, so a quiet star is never rejected just because of its random period."""
    lo, hi = QUIET_PERIOD_RANGE
    hi = min(hi, (t.max() - t.min()) / 2)
    rng = np.random.default_rng(None if seed is None else int(seed))
    for _ in range(tries):
        P = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        if fold_bin(t, f, P)[1].min() >= 5:
            break
    return P


def process_star(kind, time, flux, catalogue_period=None, seed=None):
    """kind in {"EB", "DSCT", "ROT", "Noise"} or one of UNSEEN_KINDS.
    seed: quiet stars only - seeds the random fold period (the build uses the TIC number).
    Returns dict with status "ok" or "rejected" + reason."""
    t, f = clean(time, flux)
    if t is None:
        return {"status": "rejected", "reason": "fewer than 1000 good points"}
    out = {"n_points": int(len(t)), "baseline_days": float(t.max() - t.min()),
           "point_noise": float(point_noise(f))}
    freq, amp = amplitude_spectrum(t, f)
    f1, a1, snr1 = top_peak(freq, amp)
    out.update({"ls_freq": f1, "ls_amp": a1, "ls_snr": snr1})

    if kind == "EB":
        P = float(catalogue_period) if catalogue_period is not None else np.nan
        if not (EB_PERIOD_RANGE[0] <= P <= EB_PERIOD_RANGE[1]):
            return {**out, "status": "rejected", "reason": f"period {P} outside {EB_PERIOD_RANGE}"}
        if out["baseline_days"] < 2 * P:
            return {**out, "status": "rejected", "reason": "fewer than 2 orbits in the sector"}
    elif kind == "DSCT":
        if not (DSCT_FMIN <= f1 <= DSCT_FMAX):
            return {**out, "status": "rejected", "reason": f"dominant frequency {f1:.2f} c/d not delta Scuti"}
        if snr1 < DSCT_MIN_SNR:
            return {**out, "status": "rejected", "reason": f"dominant peak S/N {snr1:.1f} < {DSCT_MIN_SNR}"}
        P = 1.0 / f1
    elif kind == "ROT":
        fr, ar = amplitude_spectrum(t, f, ROT_FMIN, ROT_FMAX)
        f1r, a1r, snr1r = top_peak(fr, ar)
        out.update({"ls_freq": f1r, "ls_amp": a1r, "ls_snr": snr1r})
        if snr1r < ROT_MIN_SNR:
            return {**out, "status": "rejected", "reason": f"rotation peak S/N {snr1r:.1f} < {ROT_MIN_SNR}"}
        P = 1.0 / f1r
        if out["baseline_days"] < 2 * P:
            return {**out, "status": "rejected", "reason": "fewer than 2 rotations in the sector"}
    elif kind == "Noise":
        if snr1 >= QUIET_MAX_SNR:
            return {**out, "status": "rejected", "reason": f"variable: peak S/N {snr1:.1f} at {f1:.2f} c/d"}
        fs, as_ = amplitude_spectrum(t, f, ROT_FMIN, ROT_FMAX)
        f1s, _, snr1s = top_peak(fs, as_)
        out.update({"slow_freq": f1s, "slow_snr": snr1s})
        if snr1s >= ROT_MIN_SNR:
            return {**out, "status": "rejected", "reason": f"slow variable: peak S/N {snr1s:.1f} at {f1s:.2f} c/d"}
        P = quiet_period(t, f, seed)
        out["period_source"] = "random"
    elif kind in UNSEEN_KINDS:
        cp = float(catalogue_period) if catalogue_period is not None and np.isfinite(catalogue_period) else np.nan
        if kind == "DIP":
            P, snr_b = bls_period(t, f)
            out.update({"bls_period": P, "bls_snr": snr_b, "period_source": "BLS"})
            if snr_b < 7:
                return {**out, "status": "rejected", "reason": f"no clear periodic dip (BLS S/N {snr_b:.1f})"}
        elif 0.05 <= cp <= out["baseline_days"] / 2:
            P = cp
            out["period_source"] = "catalogue"
        else:
            fa, aa = amplitude_spectrum(t, f, ROT_FMIN, FREQ_MAX)
            f1a, a1a, snr1a = top_peak(fa, aa)
            out.update({"ls_freq": f1a, "ls_amp": a1a, "ls_snr": snr1a, "period_source": "periodogram"})
            if snr1a < UNSEEN_MIN_SNR:
                return {**out, "status": "rejected", "reason": f"no significant variability (S/N {snr1a:.1f})"}
            P = 1.0 / f1a
    else:
        raise ValueError(kind)

    b, cnt = fold_bin(t, f, P)
    sigma_bin = out["point_noise"] / np.sqrt(max(np.median(cnt), 1))
    out.update({"period": P, "sigma_bin": float(sigma_bin), "min_points_per_bin": int(cnt.min())})
    if cnt.min() < 5:
        return {**out, "status": "rejected", "reason": "phase coverage too sparse"}

    mask = np.zeros(N_BINS, bool)
    if kind == "EB":
        mask, info = eclipse_mask(b, sigma_bin)
        if mask is None:
            return {**out, "status": "rejected", "reason": info}
        out.update(info)
        out["detached_like"] = bool(info["mask_fraction"] < DETACHED_MAX_MASK)
    if kind == "Noise":
        # no hidden eclipses/transits: 30-min averages must not dip below 6 sigma
        k = 15
        nb = len(f) // k
        f30 = f[: nb * k].reshape(nb, k).mean(1)
        if (np.median(f30) - f30.min()) > 6 * out["point_noise"] / np.sqrt(k):
            return {**out, "status": "rejected", "reason": "dip found (possible eclipse/transit)"}
    out.update({"status": "ok", "flux": b.astype(np.float32).tolist(), "mask": np.flatnonzero(mask).tolist()})
    return out
