"""Train and explain the 1D-CNN on real TESS stars (dataset built by build_tess_dataset.py).

Classes: quiet stars (Noise), eclipsing binaries (EB), delta Scuti stars (DSCT) and spotted rotators (ROT).
Default model (--model period): the folded light curve plus log10(period), the period entering through a
small network whose output is added to the shape logits (mantaray/model.py), so every decision splits
exactly into a shape term and a period term. --model shape trains the shape-only CNN.

For each seed (new stratified 70/15/15 split, new model):
  * accuracy, and a random-forest baseline on Fourier/shape features (+ period for the period model);
  * explanations (Grad-CAM conv1-3, Integrated Gradients) scored against the physics, with per-star
    bootstrap confidence intervals and Wilcoxon tests against chance; deletion test; sanity check;
  * results split by blending (CROWDSAP) and robustness to added noise;
  * synthetic-to-real transfer (models trained only on synthetic curves);
  * never-seen star types (gamma Dor, RR Lyrae, ellipsoidal, dip/transit) and a confidence-based
    "unknown" flag;
  * detection limit: eclipses and pulsations of known size injected into real quiet stars;
  * period model: shape/period split of every decision and the conflict test.

Usage: see README.md.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mantaray import report
from mantaray.data import TESS_CLASSES, load_csv, make_dataset, normalise
from mantaray.evaluate import (METHODS, confusion, deletion_summary, explain_all, fig_deletion, fig_examples,
                               fig_stability, physics_metrics, pm, print_physics_table, stack, summarise)
from mantaray.explain import (GradCAM1D, deletion_curve, dilate, explain_chunked, extrema_share,
                              integrated_gradients, mask_mass, pointing_game, sanity_check)
from mantaray.features import FEATURE_NAMES, shape_features
from mantaray.model import LightCurveCNN, extra_context, predict, proba, train
from mantaray.stats import paired, vs_chance

DATA = Path("data/tess")
OUT = Path("outputs/tess")
NAMES = list(TESS_CLASSES[:3])              # class names of the current run (set in main)
CROWD_BINS = [(0.99, 1.01, "isolated (CROWDSAP >= 0.99)"), (0.90, 0.99, "mild blending (0.90-0.99)"),
              (0.0, 0.90, "strong blending (< 0.90)")]
NOISE_LEVELS = (0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0)  # added noise, in units of each normalised curve's std
CONTACT_FRAC = 0.4                          # share of contact binaries in the "with contacts" synthetic set
INJ_SNR = (0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16)  # injected depth / amplitude in units of the star's per-bin noise
UNKNOWN_QUANTILE = 0.05                     # "unknown" = less confident than 95% of validation stars
USE_PERIOD = True                           # main model: shape + log-period (set in main from --model)


# ====================================================================== data

def load(vetted=False, classes=None, seed=0):
    """Returns X (normalised), Xraw (relative flux), y, M, meta, held-out set for the open-set test."""
    X, y = load_csv(DATA / "tess_dataset.csv")
    Xraw = pd.read_csv(DATA / "tess_dataset.csv").iloc[:, 1:].to_numpy(np.float32)
    meta = pd.read_csv(DATA / "tess_meta.csv")
    M = np.load(DATA / "tess_masks.npy")
    assert len(X) == len(meta) == len(M), "dataset, meta and masks are out of sync: rerun the export"
    vet_path = DATA / "tess_vetting.csv"
    if vet_path.exists():
        vet = pd.read_csv(vet_path)
        if len(vet) == len(meta) and (vet["tic"].to_numpy() == meta["tic"].to_numpy()).all():
            meta["vet_pass"] = (vet["vet_status"] == "ok") & vet["vet_pass"].astype(str).str.lower().eq("true")
        elif vetted:
            raise SystemExit("tess_vetting.csv does not match the dataset: run  python vet_tess.py  again.")
    elif vetted:
        raise SystemExit("No data/tess/tess_vetting.csv yet: run  python vet_tess.py  first.")
    present = [TESS_CLASSES[c] for c in sorted(np.unique(y))]
    names = present if classes is None else [c for c in TESS_CLASSES if c in classes and c in present]
    assert names[:3] == ["Noise", "EB", "DSCT"], "Noise, EB and DSCT must always be trained"
    keep = np.isin(y, [TESS_CLASSES.index(c) for c in names])
    held = ~keep                                 # classes left out of training -> never-seen test stars
    if "exclude" in meta:                        # quiet stars excluded by the quiet-star checks
        ex = meta["exclude"].astype(str).str.lower().eq("true").to_numpy()
        keep &= ~ex
        held &= ~ex
    if vetted:
        ok = meta["vet_pass"].to_numpy(bool)
        keep &= ok
        held &= ok
        n = min(int((keep & (y == c)).sum()) for c in np.unique(y[keep]))  # rebalance the classes
        rng = np.random.default_rng(seed)
        idx = np.concatenate([rng.choice(np.flatnonzero(keep & (y == c)), n, replace=False) for c in np.unique(y[keep])])
        keep = np.zeros(len(y), bool)
        keep[idx] = True
    held_out = {"X": X[held], "kind": meta.loc[held, "kind"].to_numpy(),
                "logp": np.log10(meta.loc[held, "period"].to_numpy(float))}
    idx = np.flatnonzero(keep)
    return X[idx], Xraw[idx], y[idx], M[idx], meta.iloc[idx].reset_index(drop=True), names, held_out


def load_unseen(held_out):
    """Never-seen test stars: unseen_dataset.csv (gamma Dor, RR Lyrae, ellipsoidal, dips) + held-out classes."""
    Xs, kinds, lps = [held_out["X"]], [held_out["kind"]], [held_out["logp"]]
    path = DATA / "unseen_dataset.csv"
    if path.exists():
        u = pd.read_csv(path)
        Xs.append(normalise(u.iloc[:, 1:].to_numpy(float)).astype(np.float32))
        kinds.append(u["kind"].to_numpy())
        um = DATA / "unseen_meta.csv"
        lps.append(np.log10(pd.read_csv(um)["period"].to_numpy(float)) if um.exists() else np.full(len(u), np.nan))
    X, k, lp = np.concatenate(Xs), np.concatenate(kinds), np.concatenate(lps)
    return (X, k, lp) if len(k) else (None, None, None)


def split(y, seed, frac=(0.7, 0.15)):
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        a, b = int(frac[0] * len(idx)), int((frac[0] + frac[1]) * len(idx))
        tr += idx[:a].tolist()
        va += idx[a:b].tolist()
        te += idx[b:].tolist()
    return np.array(tr), np.array(va), np.array(te)


# ====================================================================== per-seed analyses

def per_star_explanations(S, X, y, M, ok, detached):
    """Per-star values for every method (NaN where a metric does not apply)."""
    n = len(y)
    near = dilate(M, 4)
    eb = ok & (y == 1) & detached
    cols = {"eb_hit_chance": np.where(eb, near.mean(1), np.nan),
            "eb_share_chance": np.where(eb, M.mean(1), np.nan)}
    waves = (("dsct", 2), ("rot", 3)) if (y == 3).any() else (("dsct", 2),)
    for name, c in waves:
        cols[f"{name}_maxmin_chance"] = np.full(n, np.nan)
    for m in METHODS:
        cols[f"eb_hit_{m}"] = np.where(eb, pointing_game(S[m], near), np.nan)
        cols[f"eb_share_{m}"] = np.where(eb, mask_mass(S[m], M), np.nan)
        for name, c in waves:
            sel = ok & (y == c)
            cols[f"{name}_maxmin_{m}"] = np.full(n, np.nan)
            if sel.any():
                share, chance = extrema_share(S[m][sel], X[sel])
                cols[f"{name}_maxmin_{m}"][sel] = share
                cols[f"{name}_maxmin_chance"][sel] = chance
    return cols


def noise_robustness(model, X, y, M, detached, seed, E=None):
    """Add white noise to the test curves (simulating fainter stars) and re-test. E: extra inputs (period)."""
    rng = np.random.default_rng(seed + 99)
    rows = []
    near = dilate(M, 4)
    for level in NOISE_LEVELS:
        Xn = normalise(X + rng.normal(0, level, X.shape)).astype(np.float32) if level else X
        pred = predict(model, Xn, E)
        ok = pred == y
        eb = ok & (y == 1) & detached
        ds = ok & (y == 2)
        s_gc = explain_chunked(_gradcam_conv2, model, Xn, E, y)
        s_ig = explain_chunked(integrated_gradients, model, Xn, E, y)
        row = {"noise_level": level, "accuracy": float(ok.mean()),
               **{f"accuracy_{NAMES[c]}": float(ok[y == c].mean()) for c in range(len(NAMES))}}
        for name, s in (("gradcam_conv2", s_gc), ("intgrad", s_ig)):
            row[f"EB_peak_on_eclipse_{name}"] = float(pointing_game(s[eb], near[eb]).mean()) if eb.any() else np.nan
            row[f"DSCT_share_on_max_min_{name}"] = float(extrema_share(s[ds], Xn[ds])[0].mean()) if ds.any() else np.nan
        row["EB_peak_chance"] = float(near[eb].mean()) if eb.any() else np.nan
        rows.append(row)
    return rows


def _gradcam_conv2(model, X, y):
    g = GradCAM1D(model, model.conv2)
    try:
        return g(X, y)
    finally:
        g.remove()


def random_forest(X, y, tr, va, te, seed, logp=None):
    """Baseline on classic Fourier/shape features; with the period model it also gets log-period (fair)."""
    from sklearn.ensemble import RandomForestClassifier
    F = shape_features(X)
    names = list(FEATURE_NAMES)
    if logp is not None:
        F = np.c_[F, logp]
        names.append("log_period")
    fit = np.r_[tr, va]
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    rf.fit(F[fit], y[fit])
    pred = rf.predict(F[te])
    return {"accuracy": float((pred == y[te]).mean()),
            "accuracy_by_class": {NAMES[c]: float((pred[y[te] == c] == c).mean()) for c in range(len(NAMES))},
            "feature_importance": dict(zip(names, rf.feature_importances_.round(4).tolist()))}


def open_set(model, Xva, Xte, yte, Xu, ku, Eva=None, Ete=None, Eu=None):
    """Never-seen stars: predicted class shares, 'unknown' rate, and how well confidence separates them."""
    from sklearn.metrics import roc_auc_score
    thr = float(np.quantile(proba(model, Xva, Eva).max(1), UNKNOWN_QUANTILE))
    p_te = proba(model, Xte, Ete)
    out = {"confidence_threshold": thr,
           "known_test_flagged_unknown": float((p_te.max(1) < thr).mean()),
           "known_test_accuracy_when_confident": float((p_te.argmax(1) == yte)[p_te.max(1) >= thr].mean()),
           "types": {}}
    p_u = proba(model, Xu, Eu)
    for kind in sorted(set(ku)):
        sel = ku == kind
        conf = p_u[sel].max(1)
        pred = p_u[sel].argmax(1)
        out["types"][kind] = {"n": int(sel.sum()), "flagged_unknown": float((conf < thr).mean()),
                              "mean_confidence": float(conf.mean()),
                              **{f"called_{NAMES[c]}": float((pred == c).mean()) for c in range(len(NAMES))},
                              **{f"called_{NAMES[c]}_confidently": float(((pred == c) & (conf >= thr)).mean())
                                 for c in range(len(NAMES))}}
    labels = np.r_[np.ones(len(p_te)), np.zeros(len(p_u))]
    out["auroc_known_vs_unseen"] = float(roc_auc_score(labels, np.r_[p_te.max(1), p_u.max(1)]))
    return out


def _eclipse(n, depth, rng, hw=0.04, ingress=0.5):
    c = rng.uniform(0, 1)
    ph = (np.arange(n) + 0.5) / n
    d1 = np.minimum(np.abs(ph - c) % 1, 1 - np.abs(ph - c) % 1)
    d2 = np.minimum(np.abs(ph - (c + 0.5) % 1) % 1, 1 - np.abs(ph - (c + 0.5) % 1) % 1)
    shape = np.clip((hw - d1) / (ingress * hw), 0, 1) + 0.4 * np.clip((hw - d2) / (ingress * hw), 0, 1)
    return -depth * shape, (d1 < hw) | (d2 < hw)


def injection_test(model, Xraw_q, sigma_bin, seed, repeats=2, pools=None):
    """Inject eclipses / pulsations of known size into real quiet stars; measure what the model sees.

    pools (period model): {class: standardised log-periods of training stars}. Eclipses get a binary
    period; pulsations a delta Scuti period (sine_*) and, separately, a rotator period (rotsine_*)."""
    rng = np.random.default_rng(seed + 7)
    n_bins = Xraw_q.shape[1]
    ph = (np.arange(n_bins) + 0.5) / n_bins
    rows = []
    for snr in INJ_SNR:
        Xe, Me, Xs = [], [], []
        for _ in range(repeats):
            for x, s in zip(Xraw_q, sigma_bin):
                sig, mask = _eclipse(n_bins, snr * s, rng)
                Xe.append(x + sig)
                Me.append(mask)
                a, p0 = snr * s, rng.uniform(0, 2 * np.pi)
                Xs.append(x + a * np.sin(2 * np.pi * ph + p0) + 0.3 * a * np.sin(4 * np.pi * ph + 2 * p0))
        Xe, Me, Xs = (normalise(np.array(Xe)).astype(np.float32), np.array(Me),
                      normalise(np.array(Xs)).astype(np.float32))
        n = len(Xe)
        draw = lambda c: rng.choice(pools[c], n)[:, None].astype(np.float32) if pools else None  # noqa: E731
        Ee, Es = draw(1), draw(2)
        pe, ps = predict(model, Xe, Ee), predict(model, Xs, Es)
        row = {"snr": snr, "eclipse_called_EB": float((pe == 1).mean()), "eclipse_called_variable": float((pe != 0).mean()),
               "sine_called_DSCT": float((ps == 2).mean()), "sine_called_variable": float((ps != 0).mean())}
        if len(NAMES) > 3:
            row["sine_called_ROT"] = float((ps == 3).mean())
            if pools and 3 in pools:
                pr = predict(model, Xs, draw(3))
                row.update({"rotsine_called_ROT": float((pr == 3).mean()), "rotsine_called_DSCT": float((pr == 2).mean()),
                            "rotsine_called_variable": float((pr != 0).mean())})
        eb = pe == 1
        if eb.any():
            s_ig = explain_chunked(integrated_gradients, model, Xe[eb], None if Ee is None else Ee[eb],
                                   np.ones(eb.sum(), int))
            row["eclipse_IG_peak_on_injected"] = float(pointing_game(s_ig, dilate(Me[eb], 4)).mean())
            row["eclipse_IG_peak_chance"] = float(dilate(Me[eb], 4).mean())
        else:
            row["eclipse_IG_peak_on_injected"] = row["eclipse_IG_peak_chance"] = np.nan
        rows.append(row)
    return rows


def run_seed(seed, X, Xraw, y, M, meta, epochs, unseen, keep=False):
    tr, va, te = split(y, seed)
    K = len(NAMES)
    logp = np.log10(meta["period"].to_numpy(float))
    mu, sd = float(logp[tr].mean()), float(logp[tr].std())
    Z = ((logp - mu) / sd)[:, None].astype(np.float32) if USE_PERIOD else None
    ez = (lambda idx: None) if Z is None else (lambda idx: Z[idx])  # noqa: E731
    t1 = time.time()
    model = train(LightCurveCNN(n_classes=K, n_extra=1 if USE_PERIOD else 0), X[tr], y[tr], X[va], y[va],
                  epochs=epochs, seed=seed, verbose=False, extra=ez(tr), extra_val=ez(va))
    Xte, yte, Mte, mte = X[te], y[te], M[te], meta.iloc[te].reset_index(drop=True)
    Ete = ez(te)
    pred = predict(model, Xte, Ete)
    ok = pred == yte
    detached = mte["detached_like"].astype(str).str.lower().eq("true").values
    r = {"train_seconds": round(time.time() - t1, 1), "test_accuracy": float(ok.mean()),
         "confusion": confusion(yte, pred, K),
         "accuracy_by_class": {NAMES[c]: float(ok[yte == c].mean()) for c in range(K)},
         "EB_accuracy_detached_like": float(ok[(yte == 1) & detached].mean()) if ((yte == 1) & detached).any() else float("nan"),
         "EB_accuracy_contact_like": float(ok[(yte == 1) & ~detached].mean()) if ((yte == 1) & ~detached).any() else float("nan")}
    S = explain_chunked(explain_all, model, Xte, Ete, yte)
    r["physics"] = {n: physics_metrics(s, Xte, yte, Mte, ok, eb_sel=detached) for n, s in S.items()}
    r["deletion"] = {}
    for c in range(1, K):
        sel = ok & (yte == c)
        with extra_context(model, None if Ete is None else Ete[sel]):
            d = {n: deletion_curve(model, Xte[sel], yte[sel], S[n][sel], seed=seed)
                 for n in ("gradcam_conv2", "gradcam_conv3", "intgrad")}
        r["deletion"][NAMES[c]] = {**{n: v["salient"] for n, v in d.items()}, "random": d["intgrad"]["random"]}
    with extra_context(model, None if Ete is None else Ete[ok]):
        r["sanity_spearman_vs_random_model"] = {
            l: sanity_check(model, l, Xte[ok], yte[ok], seed=seed) for l in ("conv2", "conv3")}
    r["noise_robustness"] = noise_robustness(model, Xte, yte, Mte, detached, seed, Ete)
    r["random_forest"] = random_forest(X, y, tr, va, te, seed, logp if USE_PERIOD else None)
    if unseen[0] is not None:
        Xu, ku, lpu = unseen
        Eu = ((lpu - mu) / sd)[:, None].astype(np.float32) if USE_PERIOD else None
        r["open_set"] = open_set(model, X[va], Xte, yte, Xu, ku, ez(va), Ete, Eu)
    q = te[y[te] == 0]
    pools = {c: Z[tr][y[tr] == c, 0] for c in range(1, K)} if USE_PERIOD else None
    r["injection"] = injection_test(model, Xraw[q], meta["sigma_bin"].to_numpy(float)[q], seed, pools=pools)
    r["quiet_sigma_bin_ppm"] = float(np.median(meta["sigma_bin"].to_numpy(float)[q]) * 1e6)
    if USE_PERIOD and K == 4:   # exact shape/period split of every decision, and the conflict test
        import run_fixes
        with torch.no_grad():
            model.eval()
            Fte = model.features(torch.as_tensor(Xte), torch.as_tensor(Ete)).numpy()
        r["period_terms"] = run_fixes.period_terms(model, Fte, yte, mu, sd)
        r["conflict"] = run_fixes.conflict_test(model, {"X": X, "y": y}, te, Z, tr, seed)

    p = proba(model, Xte, Ete)
    stars = pd.DataFrame({"tic": mte["tic"], "true": [NAMES[c] for c in yte],
                          "pred": [NAMES[c] for c in pred], "p_true": p[np.arange(len(yte)), yte],
                          "crowdsap": pd.to_numeric(mte["crowdsap"], errors="coerce"), "detached_like": detached,
                          "vet_pass": mte["vet_pass"].values if "vet_pass" in mte else np.nan,
                          **per_star_explanations(S, Xte, yte, Mte, ok, detached), "seed": seed})
    stars["eb_peak_on_eclipse_intgrad"] = stars["eb_hit_intgrad"]
    stars["eb_peak_on_eclipse_gradcam_conv2"] = stars["eb_hit_gradcam_conv2"]
    ex = (Xte, yte, Mte, S, ok, mte) if keep else None
    return r, stars, ex


def transfer(X, y, seed):
    """Models trained only on synthetic curves (3 classes), evaluated on the TESS stars.

    Accuracies use the stars of the three synthetic classes; predictions are kept for all stars.
    'pred' = detached binaries only; 'pred_contact' = contact binaries added (CONTACT_FRAC).
    """
    b = 1000 * (seed + 1)
    out = {}
    for key, frac in (("pred", 0.0), ("pred_contact", CONTACT_FRAC)):
        Xs, ys, _ = make_dataset(2000, seed=b + 1, contact_frac=frac)
        Xv, yv, _ = make_dataset(500, seed=b + 2, contact_frac=frac)
        out[key] = predict(train(LightCurveCNN(), Xs, ys, Xv, yv, seed=seed, verbose=False), X)
    s3 = y < 3
    pred = out["pred"]
    return {"accuracy": float((pred[s3] == y[s3]).mean()), "confusion": confusion(y[s3], pred[s3]),
            "accuracy_by_class": {NAMES[c]: float((pred[y == c] == c).mean()) for c in range(3)},
            "accuracy_with_contacts": float((out["pred_contact"][s3] == y[s3]).mean()),
            "confusion_with_contacts": confusion(y[s3], out["pred_contact"][s3]), **out}


def transfer_table(meta, y, transfers):
    """Per-star predictions of the synthetic-trained models (one row per star per seed)."""
    syn = ["Noise", "EB", "DSCT"]
    rows = []
    for seed, t in enumerate(transfers):
        rows.append(pd.DataFrame({"tic": meta["tic"], "true": [NAMES[c] for c in y],
                                  "detached_like": meta["detached_like"].astype(str).str.lower().eq("true"),
                                  "pred_synthetic_model": [syn[c] for c in t["pred"]],
                                  "pred_synthetic_contact_model": [syn[c] for c in t["pred_contact"]],
                                  "seed": seed}))
    return pd.concat(rows, ignore_index=True)


# ====================================================================== pooled analyses

def crowding_table(stars):
    rows = []
    for lo, hi, label in CROWD_BINS:
        s = stars[(stars.crowdsap >= lo) & (stars.crowdsap < hi)]
        eb = s[s.true == "EB"]
        rows.append({"group": label, "n_star_tests": int(len(s)),
                     "accuracy": float((s.true == s.pred).mean()) if len(s) else float("nan"),
                     "EB_accuracy": float((eb.true == eb.pred).mean()) if len(eb) else float("nan"),
                     "EB_peak_on_eclipse_intgrad": float(eb.eb_peak_on_eclipse_intgrad.mean()) if len(eb) else float("nan"),
                     "EB_peak_on_eclipse_gradcam_conv2": float(eb.eb_peak_on_eclipse_gradcam_conv2.mean()) if len(eb) else float("nan")})
    return rows


def significance_table(stars):
    rows = []
    metrics = [("EB: heatmap peak on eclipse", "eb_hit_", "eb_hit_chance"),
               ("EB: share of heatmap inside eclipse", "eb_share_", "eb_share_chance"),
               ("DSCT: share on light max/min", "dsct_maxmin_", "dsct_maxmin_chance")]
    if "rot_maxmin_chance" in stars:
        metrics.append(("ROT: share on light max/min", "rot_maxmin_", "rot_maxmin_chance"))
    for m in METHODS:
        for metric, prefix, chance in metrics:
            if prefix + m in stars:
                rows.append({"metric": metric, "method": m, **vs_chance(stars, prefix + m, chance)})
    comps = []
    for a, b in (("intgrad", "gradcam_conv3"), ("gradcam_conv2", "gradcam_conv3"), ("intgrad", "gradcam_conv2")):
        for metric, prefix in (("EB: heatmap peak on eclipse", "eb_hit_"), ("DSCT: share on light max/min", "dsct_maxmin_")):
            comps.append({"metric": metric, "comparison": f"{a} minus {b}", **paired(stars, prefix + a, prefix + b)})
    return pd.DataFrame(rows), pd.DataFrame(comps)


def snr50(rows, key):
    """Injected S/N at which the recovery fraction first crosses 50% (log-linear interpolation)."""
    s = np.array([r["snr"]["mean"] for r in rows])
    v = np.array([r[key]["mean"] for r in rows])
    for i in range(1, len(s)):
        if v[i - 1] < 0.5 <= v[i]:
            return float(np.exp(np.interp(0.5, [v[i - 1], v[i]], np.log([s[i - 1], s[i]]))))
    return float(s[0]) if v[0] >= 0.5 else float("nan")


def _safe(name, fn, *args):
    """Draw one figure; a plotting problem must never stop the run or lose the results."""
    try:
        fn(*args)
    except Exception as e:  # noqa: BLE001
        print(f"  (figure {name} skipped: {type(e).__name__}: {e})")


def draw_figures(summary, stars, tp, X, M, meta, sig, comps):
    synth_path = Path("outputs/results.json")
    synth = json.loads(synth_path.read_text()) if synth_path.exists() else None
    _safe("fig5", report.tess_overview, OUT / "fig5_tess_overview.png", summary, synth, stars, tp)
    mis_path = OUT / "misclassified.csv"
    mis = pd.read_csv(mis_path) if mis_path.exists() else None
    _safe("fig6", report.misclassified, OUT / "fig6_tess_misclassified.png", X, meta, M, mis)
    if "confusion_summed_over_seeds(rows=true)" in summary:
        _safe("fig3", report.confusion, OUT / "fig3_tess_confusion.png", summary)
    if sig is not None and len(sig):
        _safe("fig7", report.significance, OUT / "fig7_tess_significance.png", sig)
    if "noise_robustness" in summary:
        _safe("fig8", report.noise_robustness, OUT / "fig8_tess_noise_robustness.png",
              summary["noise_robustness"], NOISE_LEVELS)
    if "random_forest" in summary:
        _safe("fig9", report.baseline, OUT / "fig9_tess_baseline.png", summary)
    vet_path = DATA / "tess_vetting.csv"
    if vet_path.exists():
        _safe("fig0", report.vetting, OUT / "fig0_tess_vetting.png", pd.read_csv(vet_path))
    if "open_set" in summary:
        _safe("fig10", report.open_set, OUT / "fig10_tess_never_seen.png", summary["open_set"], summary["class_names"])
    if "injection" in summary:
        _safe("fig11", report.detection_limit, OUT / "fig11_tess_detection_limit.png", summary["injection"],
              summary["quiet_sigma_bin_ppm"]["mean"], summary.get("detection_limit", {}))
    if "period_terms" in summary:
        import run_fixes
        _safe("fig17", report.fix_period, OUT / "fig17_tess_period_split.png",
              {"B_period": {"period_terms": summary["period_terms"]}, "class_names": summary["class_names"]},
              run_fixes.PERIOD_GRID, run_fixes._period_ranges(), "B_period", summary.get("conflict"))


# ====================================================================== main

def main():
    global OUT, NAMES, USE_PERIOD
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("period", "shape"), default="period",
                    help="period (default, the final model): folded curve + log-period; shape: folded curve only")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40, help="more epochs than synthetic: the real set is smaller")
    ap.add_argument("--vetted", action="store_true", help="use only stars that passed vet_tess.py")
    ap.add_argument("--classes", default=None,
                    help="comma-separated classes to train on (default: all in the dataset); "
                         "left-out classes are added to the never-seen test stars")
    ap.add_argument("--plots-only", action="store_true", help="redraw figures from saved results, no retraining")
    args = ap.parse_args()
    classes = args.classes.split(",") if args.classes else None
    USE_PERIOD = args.model == "period"
    X, Xraw, y, M, meta, NAMES, held = load(vetted=args.vetted, classes=classes)
    OUT = Path("outputs/tess" + ("_vetted" if args.vetted else "") + ("_period" if USE_PERIOD else "")
               + (f"_{len(NAMES)}class" if classes else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    unseen = load_unseen(held)

    if args.plots_only:
        summary = json.loads((OUT / "results.json").read_text())
        stars = pd.read_csv(OUT / "per_star_tests.csv")
        tp_path = OUT / "transfer_predictions.csv"
        tp = pd.read_csv(tp_path) if tp_path.exists() else None
        if tp is None or "pred_synthetic_contact_model" not in tp:
            print("Recomputing synthetic-model predictions on the TESS stars ...")
            tp = transfer_table(meta, y, [transfer(X, y, s) for s in range(summary["n_seeds"])])
            tp.to_csv(tp_path, index=False)
        sig_path, comp_path = OUT / "significance.csv", OUT / "method_comparisons.csv"
        sig = pd.read_csv(sig_path) if sig_path.exists() else None
        comps = pd.read_csv(comp_path) if comp_path.exists() else None
        if sig is None and "eb_hit_intgrad" in stars:
            sig, comps = significance_table(stars)
        summary.setdefault("class_names", NAMES)
        draw_figures(summary, stars, tp, X, M, meta, sig, comps)
        print(f"Figures redrawn in {OUT}/")
        return

    counts = {NAMES[c]: int((y == c).sum()) for c in range(len(NAMES))}
    n_det = int(meta["detached_like"].astype(str).str.lower().eq("true").sum())
    n_unseen = {} if unseen[0] is None else pd.Series(unseen[1]).value_counts().to_dict()
    print(f"Model: {'1D-CNN + period (final model)' if USE_PERIOD else '1D-CNN, shape only'}")
    print(f"TESS dataset{' (vetted stars only)' if args.vetted else ''}: {len(y)} stars {counts}; "
          f"detached-like EBs with eclipse masks: {n_det}; never-seen test stars: {n_unseen or 'none'}")
    t0 = time.time()
    runs, all_stars, ex, transfers = [], [], None, []
    for seed in range(args.seeds):
        r, stars, e = run_seed(seed, X, Xraw, y, M, meta, args.epochs, unseen, keep=(seed == 0))
        runs.append(r)
        all_stars.append(stars)
        ex = ex or e
        transfers.append(transfer(X, y, seed))
        extra = f" | never-seen AUROC {r['open_set']['auroc_known_vs_unseen']:.3f}" if "open_set" in r else ""
        print(f"seed {seed + 1}/{args.seeds}: CNN {r['test_accuracy']:.3f} | random forest "
              f"{r['random_forest']['accuracy']:.3f} | synthetic-trained CNN {transfers[-1]['accuracy']:.3f} "
              f"(with contact binaries {transfers[-1]['accuracy_with_contacts']:.3f}){extra}")

    stars = pd.concat(all_stars, ignore_index=True)
    skip = ("deletion", "confusion", "noise_robustness", "random_forest", "open_set", "injection", "period_terms",
            "conflict")
    summary = summarise(stack([{k: v for k, v in r.items() if k not in skip} for r in runs]))
    summary["class_names"] = NAMES
    summary["confusion_summed_over_seeds(rows=true)"] = np.sum([r["confusion"] for r in runs], 0).tolist()
    summary["deletion_fractions"] = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
    summary["deletion"], D = deletion_summary(runs)
    summary["noise_robustness"] = summarise(stack([r["noise_robustness"] for r in runs]))
    summary["random_forest"] = summarise(stack([r["random_forest"] for r in runs]))
    if "open_set" in runs[0]:
        summary["open_set"] = summarise(stack([r["open_set"] for r in runs]))
    summary["injection"] = summarise(stack([r["injection"] for r in runs]))
    summary["detection_limit"] = {k: snr50(summary["injection"], k) for k in
                                  ("eclipse_called_EB", "eclipse_called_variable", "sine_called_variable",
                                   "rotsine_called_variable") if k in summary["injection"][0]}
    for key in ("period_terms", "conflict"):
        if key in runs[0]:
            summary[key] = summarise(stack([r[key] for r in runs]))
    summary["model"] = "1D-CNN + log-period" if USE_PERIOD else "1D-CNN, shape only"
    summary["random_forest_inputs"] = "shape features + period" if USE_PERIOD else "shape features"
    summary["crowding_pooled_over_seeds"] = crowding_table(stars)
    keep_t = ("accuracy", "accuracy_by_class", "accuracy_with_contacts")
    summary["synthetic_to_real_transfer"] = summarise(stack([{k: t[k] for k in keep_t} for t in transfers]))
    summary["synthetic_to_real_confusion_summed"] = np.sum([t["confusion"] for t in transfers], 0).tolist()
    summary["synthetic_with_contacts_to_real_confusion_summed"] = np.sum([t["confusion_with_contacts"] for t in transfers], 0).tolist()
    tp = transfer_table(meta, y, transfers)
    tp.to_csv(OUT / "transfer_predictions.csv", index=False)
    summary["dataset"] = {"counts": counts, "detached_like_EBs": n_det, "vetted_only": args.vetted,
                          "never_seen_test_stars": {k: int(v) for k, v in n_unseen.items()}}
    summary["n_seeds"] = args.seeds
    summary["runtime_seconds"] = round(time.time() - t0, 1)

    sig, comps = significance_table(stars)
    sig.to_csv(OUT / "significance.csv", index=False)
    comps.to_csv(OUT / "method_comparisons.csv", index=False)
    wrong = stars[stars.true != stars.pred]
    if len(wrong):
        mis = (wrong.groupby(["tic", "true", "pred"]).size().rename("times_misclassified").reset_index()
               .sort_values("times_misclassified", ascending=False))
        mis.to_csv(OUT / "misclassified.csv", index=False)
    else:
        (OUT / "misclassified.csv").unlink(missing_ok=True)
    stars.to_csv(OUT / "per_star_tests.csv", index=False)

    Xte, yte, Mte, S, ok, mte = ex
    titles = [f"TIC {t}" for t in mte["tic"]]
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    _safe("fig1", fig_examples, OUT / "fig1_tess_examples.png", Xte, yte, Mte, S, ok, None, titles, NAMES)
    _safe("fig2", fig_deletion, OUT / "fig2_tess_deletion.png", D, summary["deletion_fractions"])
    _safe("fig4", fig_stability, OUT / "fig4_tess_stability.png", summary["physics"])
    draw_figures(summary, stars, tp, X, M, meta, sig, comps)

    T = summary["synthetic_to_real_transfer"]
    print(f"\n=== TESS results over {args.seeds} seed(s), {summary['model']}: mean ± std ===")
    print(f"CNN accuracy: {pm(summary['test_accuracy'])}   by class: " +
          ", ".join(f"{k} {pm(v)}" for k, v in summary["accuracy_by_class"].items()))
    print(f"Random-forest baseline ({summary['random_forest_inputs']}): {pm(summary['random_forest']['accuracy'])}   by class: " +
          ", ".join(f"{k} {pm(v)}" for k, v in summary["random_forest"]["accuracy_by_class"].items()))
    print(f"EB accuracy, detached-like: {pm(summary['EB_accuracy_detached_like'])}; "
          f"contact-like: {pm(summary['EB_accuracy_contact_like'])}")
    print(f"Synthetic-trained CNN on TESS (Noise/EB/DSCT stars): {pm(T['accuracy'])}; with contact binaries "
          f"in training: {pm(T['accuracy_with_contacts'])}")
    print_physics_table(summary["physics"])
    if "ROT_share_on_max_min" in summary["physics"]["intgrad"]:
        print("ROT share on light max/min: " + ", ".join(
            f"{m} {pm(summary['physics'][m]['ROT_share_on_max_min'])}" for m in METHODS)
            + f"  (chance {pm(summary['physics']['intgrad']['ROT_share_on_max_min_chance'])})")
    print("\nSignificance vs chance (per star, 95% CI, one-sided Wilcoxon):")
    for r in sig.itertuples():
        if r.n_stars:
            print(f"  {r.metric:38s} {r.method:14s} {100 * r.mean:5.1f}% [{100 * r.ci95_low:5.1f}, {100 * r.ci95_high:5.1f}]"
                  f"  chance {100 * r.chance:5.1f}%  p = {r.p_value_vs_chance:.1e}  (n = {r.n_stars})")
    if "open_set" in summary:
        O = summary["open_set"]
        print(f"\nNever-seen stars: AUROC known vs never-seen {pm(O['auroc_known_vs_unseen'])}; "
              f"known test stars wrongly flagged 'unknown' {pm(O['known_test_flagged_unknown'])}")
        for kind, d in O["types"].items():
            calls = ", ".join(f"{NAMES[c]} {100 * d[f'called_{NAMES[c]}']['mean']:.0f}%" for c in range(len(NAMES)))
            print(f"  {kind:6s} (n={d['n']['mean']:.0f}): flagged unknown {pm(d['flagged_unknown'])} | called: {calls}")
    L = summary["detection_limit"]
    ppm = summary["quiet_sigma_bin_ppm"]["mean"]
    print(f"\nDetection limit (injected into real quiet stars; 50% recovery; typical per-bin noise {ppm:.0f} ppm):")
    for k, v in L.items():
        print(f"  {k:26s} S/N {v:5.2f}  = {v * ppm:7.0f} ppm = {1.0857 * v * ppm / 1000:6.2f} mmag")
    if "period_terms" in summary:
        T = summary["period_terms"]["by_class"]
        print("\nDecision split (correct test stars): shape term alone right | period term alone right | period share of margin")
        for c in NAMES:
            print(f"  {c:5s} {pm(T[c]['shape_alone_correct'])} | {pm(T[c]['period_alone_correct'])} | "
                  f"{pm(T[c]['period_share_of_margin'])}")
        C = summary["conflict"]
        print("Conflict test: share that keeps its own class when given another class's period")
        for c in NAMES:
            print(f"  {c:5s} " + "  ".join(f"{d} period {100 * C[f'{c}_given_{d}_period']['kept_own_class']['mean']:5.1f}%"
                                           for d in NAMES if d != c))
    print("\nAdded-noise robustness (accuracy | EB peak on eclipse, IG):")
    for row in summary["noise_robustness"]:
        print(f"  noise {row['noise_level']['mean']:.2f}: accuracy {pm(row['accuracy'])} | "
              f"{pm(row['EB_peak_on_eclipse_intgrad'])}")
    print(f"\nRuntime {summary['runtime_seconds']} s. Results and figures in {OUT}/")


if __name__ == "__main__":
    main()
