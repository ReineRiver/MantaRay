"""Shortcut tests and three candidate fixes for the rotator / delta Scuti confusion (vetted 4-class stars).

Phase folding removes the period, and folded rotators and delta Scuti stars look alike. Four model
variants are trained on the same splits and seeds so they can be compared star by star:
  plain   shape only
  A       S/N-matched training: variable training stars are shown at lower S/N by adding the folded
          curve of a random quiet training star (real TESS noise)
  B       shape + log10(period), the period entering as an additive per-class logit term
  A+B     both
and four "unknown" detectors are compared (C): maximum softmax (Hendrycks & Gimpel 2017), energy
(Liu et al. 2020), Mahalanobis distance (Lee et al. 2018) and k-nearest-neighbour distance (Sun et al. 2022).

Tests for every variant: accuracy (with random-forest baselines), real stars made fainter, sinusoids
injected into real quiet stars with delta Scuti-like or rotator-like periods, the conflict test,
never-seen star types, explanation metrics, and the exact shape/period split of each decision.

Usage: see README.md.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_tess as rt
from mantaray import report
from mantaray.data import normalise
from mantaray.evaluate import METHODS, explain_all, physics_metrics, pm, stack, summarise
from mantaray.explain import dilate, integrated_gradients, pointing_game
from mantaray.features import shape_features
from mantaray.model import LightCurveCNN, logits as model_logits, train

DATA = rt.DATA
OUT = Path("outputs/fixes_vetted")
NAMES = ["Noise", "EB", "DSCT", "ROT"]
VARIANTS = {"plain": (False, False), "A_snr_matched": (True, False), "B_period": (False, True), "AB_both": (True, True)}
DETECTORS = ("max_softmax", "energy", "mahalanobis", "knn")
FAINT_SNR = (30, 15, 8, 5, 3, 2)    # target S/N per bin (signal rms / noise per phase bin)
AUG_P = 0.5                         # share of variable training stars shown at a lower S/N (variant A)
FLOOR_QUANTILE = 0.05               # lowest training S/N used by the augmentation
KNN_K = 10
CHUNK = 512                         # explanation batch (model.context must match it)


# ====================================================================== data

def signal_snr(Xraw, sigma_bin):
    """Signal rms / noise per phase bin, from the folded relative flux and the measured bin noise."""
    return np.sqrt(np.clip(Xraw.var(1) - sigma_bin ** 2, 0, None)) / sigma_bin


def load_data():
    X, Xraw, y, M, meta, names, _ = rt.load(vetted=True)
    if names != NAMES:
        raise SystemExit(f"run_fixes.py needs the 4-class dataset (found {names}).")
    sig = meta["sigma_bin"].to_numpy(float)
    u = pd.read_csv(DATA / "unseen_dataset.csv")
    um = pd.read_csv(DATA / "unseen_meta.csv")
    assert (u["kind"].to_numpy() == um["kind"].to_numpy()).all(), "unseen dataset and meta are out of sync"
    return {"X": X, "Xraw": Xraw, "y": y, "M": M, "meta": meta, "sig": sig, "snr": signal_snr(Xraw, sig),
            "logp": np.log10(meta["period"].to_numpy(float)),
            "detached": meta["detached_like"].astype(str).str.lower().eq("true").to_numpy(),
            "Xu": normalise(u.iloc[:, 1:].to_numpy(float)).astype(np.float32), "ku": u["kind"].to_numpy(),
            "logpu": np.log10(um["period"].to_numpy(float))}


def snr_augment(X, Xraw, sig, y, floor, seed):
    """Variant A: returns augment(curves, idx) that lowers the S/N of half the variable stars in a batch.

    In a normalised curve (unit variance) the noise per bin is sn = sigma_bin / std(raw flux) and the
    signal variance is 1 - sn^2. To reach a target S/N we add noise of variance
    (signal rms / target)^2 - sn^2, then normalise again (median 0, std 1).
    The added noise is REAL TESS noise: the folded curve of a random quiet training star (unit std),
    rolled by a random phase - so it carries the same red noise and systematics as the quiet stars,
    not just white noise."""
    bank = torch.as_tensor(X[y == 0])
    sn = torch.as_tensor(np.clip(sig / (Xraw.std(1) + 1e-12), 1e-6, 0.999), dtype=torch.float32)
    srms = torch.sqrt(1 - sn ** 2)
    snr = srms / sn
    var = torch.as_tensor(y > 0)
    g = torch.Generator().manual_seed(seed + 123)
    lf = float(np.log(floor))

    def augment(xb, idx):
        n = len(idx)
        s, r, e = sn[idx], snr[idx], srms[idx]
        apply = var[idx] & (torch.rand(n, generator=g) < AUG_P) & (r > floor)
        u = torch.rand(n, generator=g)
        target = torch.exp(lf + u * (torch.log(r.clamp(min=floor)) - lf))
        add = torch.sqrt(torch.clamp((e / target) ** 2 - s ** 2, min=0))
        noise = bank[torch.randint(0, len(bank), (n,), generator=g)]
        noise = torch.roll(noise, shifts=int(torch.randint(0, noise.shape[1], (1,), generator=g)), dims=1)
        xn = xb + add[:, None] * noise
        xn = xn - xn.median(1, keepdim=True).values
        xn = xn / (xn.std(1, keepdim=True) + 1e-8)
        return torch.where(apply[:, None], xn, xb)

    return augment


def make_fainter(Xraw, sig, snr, target, rng):
    """Add white noise to raw folded curves so their S/N per bin drops to `target`; returns normalised curves."""
    s2 = (snr * sig) ** 2
    add = np.sqrt(np.clip(s2 / target ** 2 - sig ** 2, 0, None))
    return normalise(Xraw + add[:, None] * rng.normal(size=Xraw.shape)).astype(np.float32)


# ====================================================================== helpers for the period input

def chunked(fn, model, X, E, *args):
    """Run an explanation function in chunks with model.context = the matching extra inputs."""
    if E is None:
        return fn(model, X, *args)
    parts = []
    for i in range(0, len(X), CHUNK):
        model.context = torch.as_tensor(E[i:i + CHUNK], dtype=torch.float32)
        parts.append(fn(model, X[i:i + CHUNK], *[a[i:i + CHUNK] for a in args]))
    model.context = None
    if isinstance(parts[0], dict):
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return np.concatenate(parts)


@torch.no_grad()
def feats_logits(model, X, E):
    model.eval()
    f = model.features(torch.as_tensor(X), None if E is None else torch.as_tensor(E, dtype=torch.float32))
    return f.numpy(), model.logits_from_features(f).numpy()


# ====================================================================== "unknown" detectors (C)

class Detectors:
    """Scores where HIGHER = more like the training stars. Fitted on the training stars' features."""

    def __init__(self, Ftr, ytr):
        from sklearn.covariance import LedoitWolf
        self.mu = np.stack([Ftr[ytr == c].mean(0) for c in np.unique(ytr)])
        self.prec = LedoitWolf().fit(Ftr - self.mu[ytr]).precision_
        self.bank = Ftr / (np.linalg.norm(Ftr, axis=1, keepdims=True) + 1e-12)

    def __call__(self, F, L):
        p = np.exp(L - L.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        d = F[:, None, :] - self.mu[None]
        maha = np.einsum("ncd,de,nce->nc", d, self.prec, d).min(1)
        Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
        dist = np.sqrt(np.clip(2 - 2 * Fn @ self.bank.T, 0, None))
        knn = np.sort(dist, 1)[:, KNN_K - 1]
        lse = L.max(1) + np.log(np.exp(L - L.max(1, keepdims=True)).sum(1))
        return {"max_softmax": p.max(1), "energy": lse, "mahalanobis": -maha, "knn": -knn}


def open_set(det, va, te, yte, pred_te, un, ku, pred_u):
    from sklearn.metrics import roc_auc_score
    out = {}
    for name in DETECTORS:
        thr = float(np.quantile(va[name], rt.UNKNOWN_QUANTILE))
        s_te, s_u = te[name], un[name]
        conf = s_te >= thr
        r = {"auroc_known_vs_unseen": float(roc_auc_score(np.r_[np.ones(len(s_te)), np.zeros(len(s_u))], np.r_[s_te, s_u])),
             "known_test_flagged_unknown": float((~conf).mean()),
             "known_test_accuracy_when_confident": float((pred_te == yte)[conf].mean()) if conf.any() else float("nan"),
             "types": {}}
        for kind in sorted(set(ku)):
            sel = ku == kind
            flag = s_u[sel] < thr
            r["types"][kind] = {"n": int(sel.sum()), "flagged_unknown": float(flag.mean()),
                                **{f"called_{NAMES[c]}": float((pred_u[sel] == c).mean()) for c in range(4)},
                                **{f"called_{NAMES[c]}_confidently": float(((pred_u[sel] == c) & ~flag).mean())
                                   for c in range(4)}}
        out[name] = r
    return out


# ====================================================================== tests

def faint_test(model, D, te, E, seed):
    """Real variable test stars made fainter: what are they called at each S/N?"""
    rng = np.random.default_rng(seed + 31)
    y, rows = D["y"], []
    for level in (None,) + FAINT_SNR:
        row = {"snr": float("inf") if level is None else level}
        for c in (1, 2, 3):
            idx = te[(y[te] == c) & ((D["snr"][te] > level) if level else True)]
            if len(idx) == 0:
                row.update({f"{NAMES[c]}_n": 0, **{f"{NAMES[c]}_called_{NAMES[k]}": float("nan") for k in range(4)}})
                continue
            Xf = D["X"][idx] if level is None else make_fainter(D["Xraw"][idx], D["sig"][idx], D["snr"][idx], level, rng)
            p = model_logits(model, Xf, None if E is None else E[idx]).argmax(1).numpy()
            row[f"{NAMES[c]}_n"] = int(len(idx))
            for k in range(4):
                row[f"{NAMES[c]}_called_{NAMES[k]}"] = float((p == k).mean())
        rows.append(row)
    return rows


def injection(model, D, te, Z, tr, use_period, seed, repeats=2):
    """Eclipses and pulsations injected into the real quiet test stars (as in run_tess.injection_test).

    Pulsations are given either a delta Scuti-like or a rotator-like period (drawn from the training
    stars); eclipses an eclipsing-binary period. Shape-only models never see the period."""
    rng = np.random.default_rng(seed + 7)
    y = D["y"]
    q = te[y[te] == 0]
    Xq, sq = D["Xraw"][q], D["sig"][q]
    pools = {c: Z[tr][y[tr] == c, 0] for c in (1, 2, 3)}
    n_bins = Xq.shape[1]
    ph = (np.arange(n_bins) + 0.5) / n_bins
    rows = []
    for snr in rt.INJ_SNR:
        Xe, Me, Xs = [], [], []
        for _ in range(repeats):
            for x, s in zip(Xq, sq):
                sig, mask = rt._eclipse(n_bins, snr * s, rng)
                Xe.append(x + sig)
                Me.append(mask)
                a, p0 = snr * s, rng.uniform(0, 2 * np.pi)
                Xs.append(x + a * np.sin(2 * np.pi * ph + p0) + 0.3 * a * np.sin(4 * np.pi * ph + 2 * p0))
        Xe, Me, Xs = normalise(np.array(Xe)).astype(np.float32), np.array(Me), normalise(np.array(Xs)).astype(np.float32)
        n = len(Xe)
        Ee = rng.choice(pools[1], n)[:, None].astype(np.float32) if use_period else None
        pe = model_logits(model, Xe, Ee).argmax(1).numpy()
        row = {"snr": snr, "eclipse_called_EB": float((pe == 1).mean()), "eclipse_called_variable": float((pe != 0).mean())}
        for arm, c in (("dsct_period", 2), ("rot_period", 3)):
            Es = rng.choice(pools[c], n)[:, None].astype(np.float32) if use_period else None
            ps = model_logits(model, Xs, Es).argmax(1).numpy()
            row.update({f"sine_{arm}_called_DSCT": float((ps == 2).mean()), f"sine_{arm}_called_ROT": float((ps == 3).mean()),
                        f"sine_{arm}_called_variable": float((ps != 0).mean())})
        eb = pe == 1
        if eb.any():
            s_ig = chunked(integrated_gradients, model, Xe[eb], None if Ee is None else Ee[eb], np.ones(eb.sum(), int))
            row["eclipse_IG_peak_on_injected"] = float(pointing_game(s_ig, dilate(Me[eb], 4)).mean())
        else:
            row["eclipse_IG_peak_on_injected"] = float("nan")
        rows.append(row)
    return rows


def conflict_test(model, D, te, Z, tr, seed):
    """Period variants: test stars given periods drawn from another class's training stars."""
    rng = np.random.default_rng(seed + 11)
    y, out = D["y"], {}
    for c in range(4):
        idx = te[y[te] == c]
        for d in range(4):
            E = rng.choice(Z[tr][y[tr] == d, 0], len(idx))[:, None].astype(np.float32)
            p = model_logits(model, D["X"][idx], E).argmax(1).numpy()
            out[f"{NAMES[c]}_given_{NAMES[d]}_period"] = {"kept_own_class": float((p == c).mean()),
                                                          "called_period_class": float((p == d).mean())}
    return out


PERIOD_GRID = np.round(np.arange(-2.2, 1.31, 0.05), 3)   # log10(period / day) for the learned period prior


@torch.no_grad()
def period_terms(model, Fte, yte, mu, sd):
    """Period variants: the logits are exactly shape_term + period_term, so each decision splits in two.

    Returns (a) the learned period prior P(class | period alone) on a grid of periods and
    (b) per class: share of correct test stars that the shape term alone / the period term alone
    would get right, and the period's share of the winning margin over the runner-up class."""
    model.eval()
    z = torch.as_tensor(((PERIOD_GRID - mu) / sd)[:, None], dtype=torch.float32)
    prior = torch.softmax(model.extra_head(z), 1).numpy()
    F = torch.as_tensor(Fte)
    Ls, Lp = model.head(F[:, :32]).numpy(), model.extra_head(F[:, 32:]).numpy()
    L = Ls + Lp
    pred = L.argmax(1)
    runner = np.argsort(L, 1)[:, -2]
    i = np.arange(len(L))
    ms, mp = Ls[i, pred] - Ls[i, runner], Lp[i, pred] - Lp[i, runner]
    out = {"prior": {NAMES[c]: prior[:, c].tolist() for c in range(4)}, "by_class": {}}
    for c in range(4):
        sel = (yte == c) & (pred == c)
        out["by_class"][NAMES[c]] = {
            "shape_alone_correct": float((Ls[sel].argmax(1) == c).mean()) if sel.any() else float("nan"),
            "period_alone_correct": float((Lp[sel].argmax(1) == c).mean()) if sel.any() else float("nan"),
            "period_share_of_margin": float((np.abs(mp[sel]) / (np.abs(ms[sel]) + np.abs(mp[sel]) + 1e-12)).mean())
            if sel.any() else float("nan")}
    return out


def random_forests(D, tr, va, te, seed):
    from sklearn.ensemble import RandomForestClassifier
    F = shape_features(D["X"])
    lp = D["logp"][:, None]
    fit, y = np.r_[tr, va], D["y"]
    out = {}
    for name, feats in (("shape", F), ("shape_plus_period", np.c_[F, lp]), ("period_only", lp)):
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1)
        rf.fit(feats[fit], y[fit])
        p = rf.predict(feats[te])
        out[name] = {"accuracy": float((p == y[te]).mean()),
                     "accuracy_by_class": {NAMES[c]: float((p[y[te] == c] == c).mean()) for c in range(4)}}
    return out


def run_variant(name, D, tr, va, te, Z, Zu, floor, seed, epochs, mu=0.0, sd=1.0):
    snr_match, use_period = VARIANTS[name]
    X, y = D["X"], D["y"]
    E = Z if use_period else None
    Eu = Zu if use_period else None
    aug = snr_augment(X[tr], D["Xraw"][tr], D["sig"][tr], y[tr], floor, seed) if snr_match else None
    t1 = time.time()
    model = train(LightCurveCNN(4, n_extra=1 if use_period else 0), X[tr], y[tr], X[va], y[va], epochs=epochs,
                  seed=seed, verbose=False, extra=None if E is None else E[tr],
                  extra_val=None if E is None else E[va], augment=aug)
    sub = (lambda idx: None) if E is None else (lambda idx: E[idx])
    Ftr, _ = feats_logits(model, X[tr], sub(tr))
    Fva, Lva = feats_logits(model, X[va], sub(va))
    Fte, Lte = feats_logits(model, X[te], sub(te))
    Fu, Lu = feats_logits(model, D["Xu"], Eu)
    pred = Lte.argmax(1)
    yte = y[te]
    ok = pred == yte
    r = {"train_seconds": round(time.time() - t1, 1), "test_accuracy": float(ok.mean()),
         "accuracy_by_class": {NAMES[c]: float(ok[yte == c].mean()) for c in range(4)},
         "confusion": [[int(((yte == a) & (pred == b)).sum()) for b in range(4)] for a in range(4)]}
    S = chunked(explain_all, model, X[te], sub(te), yte)
    r["physics"] = {m: physics_metrics(S[m], X[te], yte, D["M"][te], ok, eb_sel=D["detached"][te]) for m in METHODS}
    det = Detectors(Ftr, y[tr])
    r["open_set"] = open_set(det, det(Fva, Lva), det(Fte, Lte), yte, pred, det(Fu, Lu), D["ku"], Lu.argmax(1))
    r["faint"] = faint_test(model, D, te, E, seed)
    r["injection"] = injection(model, D, te, Z, tr, use_period, seed)
    if use_period:
        r["conflict"] = conflict_test(model, D, te, Z, tr, seed)
        r["period_terms"] = period_terms(model, Fte, yte, mu, sd)
    return r


def run_seed(seed, D, epochs):
    y = D["y"]
    tr, va, te = rt.split(y, seed)
    mu, sd = D["logp"][tr].mean(), D["logp"][tr].std()
    Z = ((D["logp"] - mu) / sd)[:, None].astype(np.float32)
    Zu = ((D["logpu"] - mu) / sd)[:, None].astype(np.float32)
    floor = float(np.quantile(D["snr"][tr][y[tr] > 0], FLOOR_QUANTILE))
    out = {"snr_floor": floor, "random_forest": random_forests(D, tr, va, te, seed),
           "quiet_sigma_bin_ppm": float(np.median(D["sig"][te][y[te] == 0]) * 1e6)}
    for name in VARIANTS:
        out[name] = run_variant(name, D, tr, va, te, Z, Zu, floor, seed, epochs, mu, sd)
    return out


# ====================================================================== summary

def detection_limits(rows):
    keys = ["eclipse_called_EB"] + [f"sine_{a}_called_variable" for a in ("dsct_period", "rot_period")]
    return {k: rt.snr50(rows, k) for k in keys}


def print_summary(S):
    print(f"\n=== Shortcut tests and fixes over {S['n_seeds']} seed(s), vetted 4-class stars "
          f"({S.get('dataset_counts', {}).get('Noise', '?')} per class; quiet stars folded on "
          f"{S.get('quiet_fold', 'periodogram peak')}): mean ± std ===")
    print(f"S/N floor used by variant A: {S['snr_floor']['mean']:.2f} (signal rms / noise per bin)")
    rf = S["random_forest"]
    print("Random forest: " + "; ".join(f"{k.replace('_', ' ')} {pm(v['accuracy'])} (ROT {pm(v['accuracy_by_class']['ROT'])})"
                                        for k, v in rf.items()))
    print(f"\n{'variant':15s} {'accuracy':>15s} {'ROT':>15s} {'DSCT':>15s}  {'DSCT at S/N 3 called ROT':>24s}")
    for v in VARIANTS:
        R = S[v]
        f3 = next(r for r in R["faint"] if r["snr"]["mean"] == 3)
        print(f"{v:15s} {pm(R['test_accuracy']):>15s} {pm(R['accuracy_by_class']['ROT']):>15s} "
              f"{pm(R['accuracy_by_class']['DSCT']):>15s}  {pm(f3['DSCT_called_ROT']):>24s}")
    print("\nInjected sinusoids called 'rotator' (weak S/N 3 -> strong S/N 16), by the period they were given:")
    for v in VARIANTS:
        rows = S[v]["injection"]
        lo = next(r for r in rows if r["snr"]["mean"] == 3)
        hi = next(r for r in rows if r["snr"]["mean"] == 16)
        print(f"  {v:15s} δ Sct period: {100 * lo['sine_dsct_period_called_ROT']['mean']:5.1f}% -> "
              f"{100 * hi['sine_dsct_period_called_ROT']['mean']:5.1f}%   rotator period: "
              f"{100 * lo['sine_rot_period_called_ROT']['mean']:5.1f}% -> {100 * hi['sine_rot_period_called_ROT']['mean']:5.1f}%")
    print("\nNever-seen stars, AUROC known vs never-seen (higher = better 'unknown' flag):")
    print(f"  {'variant':15s} " + " ".join(f"{d:>16s}" for d in DETECTORS))
    for v in VARIANTS:
        print(f"  {v:15s} " + " ".join(f"{pm(S[v]['open_set'][d]['auroc_known_vs_unseen']):>16s}" for d in DETECTORS))
    print("\nExplanations (correct test stars): IG peak on eclipse | Grad-CAM conv2 on δ Sct max/min | IG, GC2 on ROT max/min")
    for v in VARIANTS:
        P = S[v]["physics"]
        print(f"  {v:15s} {pm(P['intgrad']['EB_peak_on_eclipse'])} | {pm(P['gradcam_conv2']['DSCT_share_on_max_min'])} | "
              f"{pm(P['intgrad']['ROT_share_on_max_min'])}, {pm(P['gradcam_conv2']['ROT_share_on_max_min'])} "
              f"(chance {pm(P['intgrad']['ROT_share_on_max_min_chance'])})")
    print("\nDetection limit, S/N per bin at 50% recovery (eclipse called binary | sinusoid called variable, δ Sct / rotator period):")
    ppm = S["quiet_sigma_bin_ppm"]["mean"]
    for v in VARIANTS:
        L = S[v]["detection_limit"]
        print(f"  {v:15s} eclipse {L['eclipse_called_EB']:.2f} ({L['eclipse_called_EB'] * ppm:.0f} ppm) | sinusoid "
              f"{L['sine_dsct_period_called_variable']:.2f} / {L['sine_rot_period_called_variable']:.2f}")
    for v in ("B_period", "AB_both"):
        T = S[v]["period_terms"]["by_class"]
        print(f"\nDecision split, {v} (correct test stars): shape term alone right | period term alone right | "
              f"period share of the margin")
        for c in NAMES:
            print(f"  {c:5s} {pm(T[c]['shape_alone_correct'])} | {pm(T[c]['period_alone_correct'])} | "
                  f"{pm(T[c]['period_share_of_margin'])}")
    for v in ("B_period", "AB_both"):
        C = S[v]["conflict"]
        print(f"\nConflict test, {v}: share that keeps its own class when given another class's period")
        for c in NAMES:
            print(f"  {c:5s} " + "  ".join(f"{d} period {100 * C[f'{c}_given_{d}_period']['kept_own_class']['mean']:5.1f}%"
                                           for d in NAMES if d != c))


def draw(S):
    ref = Path("outputs/tess_vetted_3class/results.json")
    ref3 = json.loads(ref.read_text())["open_set"]["auroc_known_vs_unseen"]["mean"] if ref.exists() else None
    rt._safe("fig13", report.fix_shortcut, OUT / "fig13_fix_shortcut.png", S)
    rt._safe("fig14", report.fix_accuracy, OUT / "fig14_fix_accuracy.png", S)
    rt._safe("fig15", report.fix_unknown, OUT / "fig15_fix_unknown.png", S, ref3)
    rt._safe("fig16", report.fix_explanations, OUT / "fig16_fix_explanations.png", S)
    rt._safe("fig17", report.fix_period, OUT / "fig17_fix_period.png", S, PERIOD_GRID, _period_ranges())


def _period_ranges():
    """5-95% log-period range of each training class and never-seen type (for fig17)."""
    out = {}
    try:
        m = pd.read_csv(DATA / "tess_meta.csv")
        u = pd.read_csv(DATA / "unseen_meta.csv")
        for df in (m, u):
            for k, g in df.groupby("kind"):
                lp = np.log10(g["period"].to_numpy(float))
                out[k] = [float(np.quantile(lp, 0.05)), float(np.quantile(lp, 0.95))]
    except (OSError, KeyError):
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--plots-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.plots_only:
        S = json.loads((OUT / "results.json").read_text())
        draw(S)
        print_summary(S)
        return
    D = load_data()
    src = D["meta"].loc[D["y"] == 0, "period_source"] if "period_source" in D["meta"] else pd.Series(dtype=str)
    quiet_fold = "random" if (src.astype(str) == "random").all() and len(src) else "periodogram peak"
    print(f"Quiet stars folded on: {quiet_fold} periods")
    print(f"Vetted stars: {len(D['y'])} ({', '.join(f'{n} {int((D['y'] == c).sum())}' for c, n in enumerate(NAMES))}); "
          f"never-seen: {pd.Series(D['ku']).value_counts().to_dict()}")
    t0 = time.time()
    runs = []
    for seed in range(a.seeds):
        r = run_seed(seed, D, a.epochs)
        runs.append(r)
        print(f"seed {seed + 1}/{a.seeds}: " + " | ".join(
            f"{v} {r[v]['test_accuracy']:.3f} (AUROC knn {r[v]['open_set']['knn']['auroc_known_vs_unseen']:.3f})"
            for v in VARIANTS) + f"  [{time.time() - t0:.0f} s]")
    S = {}
    for key in ("snr_floor", "random_forest", "quiet_sigma_bin_ppm"):
        S[key] = summarise(stack([r[key] for r in runs]))
    for v in VARIANTS:
        R = [{k: x for k, x in r[v].items() if k != "confusion"} for r in runs]
        S[v] = summarise(stack(R))
        S[v]["confusion_summed_over_seeds(rows=true)"] = np.sum([r[v]["confusion"] for r in runs], 0).tolist()
        S[v]["detection_limit"] = detection_limits(S[v]["injection"])
    S["class_names"] = NAMES
    S["quiet_fold"] = quiet_fold
    S["dataset_counts"] = {n: int((D["y"] == c).sum()) for c, n in enumerate(NAMES)}
    S["variants"] = {k: {"snr_matched": a_, "period_input": b_} for k, (a_, b_) in VARIANTS.items()}
    S["n_seeds"] = a.seeds
    S["runtime_seconds"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(S, indent=2))
    draw(S)
    print_summary(S)
    print(f"\nRuntime {S['runtime_seconds']} s. Results and figures in {OUT}/")


if __name__ == "__main__":
    main()
