"""Vetting tests: is the signal real, periodic, and coming from the target star?

Standard checks used when vetting TESS signals, applied to every accepted star
(no network code here, so all of it is testable offline):

1. Split-half test (EB, DSCT)
   Fold the first and second half of the sector separately. A real periodic signal
   repeats: eclipse depth (EB) or pulsation amplitude (DSCT) must agree within a factor 2
   and be detected (>= 3 sigma) in BOTH halves. Dust passing in front of a star,
   one-off glitches and other non-repeating events fail.
2. Odd-even test (EB)
   Compare alternate primary eclipses. Different depths mean the catalogue period is
   half the true period (the "primary" is really primary + secondary), or the dips are
   not eclipses.
3. Centroid source-offset test (EB, DSCT), after Bryson et al. 2013 (Kepler)
   When any star in the aperture varies, the flux-weighted image position shifts. Dividing
   that shift by the fractional depth (or amplitude) of the signal estimates how far from
   the image centre the VARYING source sits. A source >= 1 TESS pixel (21") away, with a
   significant shift, means the signal most likely comes from a neighbouring star.
   (A shift alone is not enough: in a crowded aperture the image also moves when the target
   itself varies; then the implied offset is small.)
4. Shape flag (EB)
   "Binary" with no secondary eclipse and a smooth dip covering > 40% of the orbit:
   may be a rotating spotted star or a pulsator rather than an eclipsing binary.
"""
import numpy as np

from mantaray.tess import N_BINS, clean, fold_bin, point_noise, robust_std, to_float

SPLIT_RATIO = (0.5, 2.0)        # allowed depth/amplitude ratio between the two halves
SPLIT_MIN_SIGMA = 3.0
ODDEVEN_MAX_SIGMA = 3.0
ODDEVEN_RATIO = (0.8, 1.25)
CENTROID_MAX_SNR = 5.0
PIXEL_ARCSEC = 21.0
MAX_SOURCE_OFFSET = 21.0        # arcsec: one TESS pixel


def _detrend(c, window=721):
    """Remove slow pointing drift from a centroid series with a running median (~1 day at 2 min)."""
    from scipy.ndimage import median_filter
    c = np.asarray(c, float)
    ok = np.isfinite(c)
    if ok.sum() < window:
        return c - np.nanmedian(c)
    out = np.full_like(c, np.nan)
    out[ok] = c[ok] - median_filter(c[ok], size=window, mode="nearest")
    return out


def _sine_amp(t, y, f):
    """Least-squares amplitude of a sinusoid at frequency f, and its 1-sigma error."""
    ok = np.isfinite(y)
    t, y = t[ok], y[ok] - np.mean(y[ok])
    A = np.column_stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    amp = float(np.hypot(*coef))
    sig = float(robust_std(y - A @ coef) * np.sqrt(2.0 / max(len(y), 1)))
    return amp, sig


def _depth(f, sel, out_sel, noise):
    n = int(sel.sum())
    if n < 3 or out_sel.sum() < 3:
        return np.nan, np.inf
    return float(np.median(f[out_sel]) - np.mean(f[sel])), float(noise / np.sqrt(n))


def _primary_region(mask_bins, primary_bin, n=N_BINS):
    """Bins of the mask that form the contiguous run containing the primary eclipse."""
    m = np.zeros(n, bool)
    m[list(mask_bins)] = True
    reg = np.zeros(n, bool)
    if not m[primary_bin]:
        return reg
    reg[primary_bin] = True
    for step in (1, -1):
        j = primary_bin
        for _ in range(n):
            j = (j + step) % n
            if m[j] and not reg[j]:
                reg[j] = True
            else:
                break
    return reg


def vet_star(kind, time, flux, centroid_col, centroid_row, period, crowdsap=np.nan,
             mask_bins=(), primary_bin=None, mask_fraction=np.nan, depth2_sigma=np.nan, ls_freq=np.nan):
    """Run the vetting tests on one star. Returns a dict of measurements + pass/fail flags."""
    t, f, idx = clean(time, flux, return_index=True)
    if t is None:
        return {"vet_status": "error", "vet_reason": "too few points"}
    cc, cr = to_float(centroid_col)[idx], to_float(centroid_row)[idx]
    noise = point_noise(f)
    t0 = t[0]                       # same phase zero-point as the dataset (mantaray.tess.fold_bin)
    phase = ((t - t0) / period) % 1.0
    pbin = np.minimum((phase * N_BINS).astype(int), N_BINS - 1)
    first = t < np.median(t)
    out = {"vet_status": "ok", "crowdsap": float(crowdsap) if crowdsap is not None else np.nan}
    reasons = []

    if kind == "EB":
        inmask = np.isin(pbin, list(mask_bins))
        prim = _primary_region(mask_bins, int(primary_bin)) if primary_bin is not None else np.zeros(N_BINS, bool)
        inprim = prim[pbin]
        # 1. split-half
        d1, s1 = _depth(f, inprim & first, ~inmask & first, noise)
        d2, s2 = _depth(f, inprim & ~first, ~inmask & ~first, noise)
        ratio = d1 / d2 if d2 and np.isfinite(d2) and d2 > 0 else np.nan
        ok_split = bool(d1 > SPLIT_MIN_SIGMA * s1 and d2 > SPLIT_MIN_SIGMA * s2
                        and SPLIT_RATIO[0] <= ratio <= SPLIT_RATIO[1])
        out.update({"split_depth_ratio": ratio, "split_sigma_half1": d1 / s1, "split_sigma_half2": d2 / s2,
                    "split_pass": ok_split})
        if not ok_split:
            reasons.append("eclipse not repeated in both halves")
        # 2. odd-even (cycle number counted from the primary eclipse centre)
        ph0 = (primary_bin + 0.5) / N_BINS if primary_bin is not None else 0.0
        cyc = np.floor((t - t0) / period - ph0 + 0.5).astype(int)
        do, so = _depth(f, inprim & (cyc % 2 == 0), ~inmask, noise)
        de, se = _depth(f, inprim & (cyc % 2 == 1), ~inmask, noise)
        oe_sig = abs(do - de) / np.hypot(so, se) if np.isfinite(do) and np.isfinite(de) else np.nan
        oe_ratio = do / de if de and np.isfinite(de) and de > 0 else np.nan
        ok_oe = bool(np.isfinite(oe_sig) and (oe_sig < ODDEVEN_MAX_SIGMA or ODDEVEN_RATIO[0] <= oe_ratio <= ODDEVEN_RATIO[1]))
        out.update({"oddeven_sigma": oe_sig, "oddeven_ratio": oe_ratio, "oddeven_pass": ok_oe})
        if not ok_oe:
            reasons.append("odd and even eclipses differ (period may be wrong)")
        # 3. centroid shift in vs out of eclipse, relative to the eclipse depth
        signal, _ = _depth(f, inprim, ~inmask, noise)
        snr2, shift = 0.0, 0.0
        for c in (cc, cr):
            c = _detrend(c)
            ok = np.isfinite(c)
            if (inprim & ok).sum() > 3 and (~inmask & ok).sum() > 3:
                d = np.mean(c[inprim & ok]) - np.mean(c[~inmask & ok])
                s = robust_std(c[ok]) / np.sqrt((inprim & ok).sum())
                snr2 += (d / s) ** 2 if s > 0 else 0.0
                shift += d ** 2
        # 4. shape flag
        sinus = bool(np.nan_to_num(mask_fraction) > 0.4 and np.nan_to_num(depth2_sigma) < 5)
        out["no_secondary_smooth_dip"] = sinus
        if sinus:
            reasons.append("no secondary eclipse, smooth wide dip (possibly not eclipsing)")
    elif kind in ("DSCT", "ROT"):     # coherent periodic signal: pulsation or spot rotation
        f1 = float(ls_freq) if np.isfinite(ls_freq) else 1.0 / period
        a1, e1 = _sine_amp(t[first], f[first], f1)
        a2, e2 = _sine_amp(t[~first], f[~first], f1)
        ratio = a1 / a2 if a2 > 0 else np.nan
        ok_split = bool(a1 > SPLIT_MIN_SIGMA * e1 and a2 > SPLIT_MIN_SIGMA * e2
                        and SPLIT_RATIO[0] <= ratio <= SPLIT_RATIO[1])
        out.update({"split_amp_ratio": ratio, "split_sigma_half1": a1 / e1 if e1 else np.nan,
                    "split_sigma_half2": a2 / e2 if e2 else np.nan, "split_pass": ok_split})
        if not ok_split:
            reasons.append("pulsation not repeated in both halves" if kind == "DSCT"
                           else "rotation signal changed by more than 2x between halves")
        signal, _ = _sine_amp(t, f, f1)
        snr2, shift = 0.0, 0.0
        for c in (cc, cr):
            c = _detrend(c)
            if np.isfinite(c).sum() > 100:
                a, e = _sine_amp(t, c, f1)
                snr2 += (a / e) ** 2 if e > 0 else 0.0
                shift += a ** 2
    else:  # Noise: quiet in both halves (no dip test needed again; done when the star was accepted)
        out["split_pass"] = True
        snr2, shift, signal = 0.0, 0.0, np.nan

    if kind in ("EB", "DSCT", "ROT"):
        csnr = float(np.sqrt(snr2))
        shift_as = float(np.sqrt(shift) * PIXEL_ARCSEC)
        offset = shift_as / signal if np.isfinite(signal) and signal > 0 else np.nan
        flag = bool(csnr > CENTROID_MAX_SNR and np.isfinite(offset) and offset >= MAX_SOURCE_OFFSET)
        out.update({"centroid_snr": csnr, "centroid_shift_arcsec": shift_as, "signal_depth_or_amp": float(signal),
                    "source_offset_arcsec": float(offset), "centroid_flag": flag})
        if flag:
            reasons.append("signal source >= 1 TESS pixel from the target (likely a neighbouring star)")
    out["vet_pass"] = not reasons
    out["vet_reason"] = "; ".join(reasons)
    return out


def fold_halves(t, f, period, n=N_BINS):
    """Folded curves of the first and second half (for figures)."""
    first = t < np.median(t)
    return fold_bin(t[first], f[first], period, n, t0=t[0])[0], fold_bin(t[~first], f[~first], period, n, t0=t[0])[0]
