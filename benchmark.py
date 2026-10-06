"""Fair benchmark: every model on the same stars, splits and metrics, plus an outside catalogue check.

1. Same splits and metrics (5 seeds, the stratified 70/15/15 splits of run_tess.py):
     1D-CNN (shape), 1D-CNN + period, random forest (shape; shape + period; period only) and
     gradient boosting (shape + period). Accuracy, balanced accuracy, macro-F1, per-class
     precision / recall / F1, star-level bootstrap 95% CIs and a paired McNemar test.
2. Stars never used for training: stars that failed vetting, vetted stars left out when the classes
   were balanced, and accuracy against TESS magnitude and signal-to-noise.
3. Outside check against Gao, Chen, Wang & Liu 2025 (ApJS 276, 57; 72,505 TESS 2-min periodic
   variables), matched by TIC: label agreement, prediction agreement, period agreement, and whether
   any quiet star is listed as a periodic variable. Their Table 5 is downloaded once to data/external/.

Usage: see README.md.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_fixes as F
import run_tess as rt
from mantaray import report
from mantaray.evaluate import pm, stack, summarise
from mantaray.features import shape_features
from mantaray.model import LightCurveCNN, logits as model_logits, train

OUT = Path("outputs/benchmark")
EXT = Path("data/external")
GAO_FILE = EXT / "gao2025_table5.txt"
GAO_URL = "https://nadc.china-vo.org/res/file_upload/download?id=48032"
NAMES = F.NAMES
MODELS = ("cnn_shape", "cnn_period", "rf_shape", "rf_shape_period", "gb_shape_period", "rf_period_only")
GAO_TO_OURS = {"EA": "EB", "EB": "EB", "EW": "EB", "DSCT": "DSCT", "HADS": "DSCT", "ROT": "ROT",
               "RRAB": "RR", "RRCD": "RR", "RRC": "RR", "CEPHEIDS": "other", "CEP": "other", "GCAS": "other",
               "UV": "other", "YSO": "other"}
MAG_BINS = [(0, 8, "< 8"), (8, 9, "8-9"), (9, 10, "9-10"), (10, 11, "10-11"), (11, 12, "11-12"), (12, 30, "> 12")]
SNR_BINS = [(0, 5, "< 5"), (5, 10, "5-10"), (10, 30, "10-30"), (30, 100, "30-100"), (100, 1e9, "> 100")]
N_BOOT = 1000
GAO_CONFIDENT = 0.5      # Gao et al. "correct classification probability" counted as a confident label


# ====================================================================== data

def load_all():
    """Vetted, balanced stars (the benchmark set) + the extra test sets."""
    D = F.load_data()                                    # vetted, balanced, 4 classes
    X, Xraw, y, M, meta, names, _ = rt.load(vetted=False)  # every row (quiet stars that failed re-folding excluded)
    key = lambda m, yy: [f"{int(t)}_{int(c)}" for t, c in zip(m["tic"], yy)]  # noqa: E731
    in_bench = np.isin(key(meta, y), key(D["meta"], D["y"]))
    vet = meta["vet_pass"].fillna(False).astype(bool).to_numpy() if "vet_pass" in meta else np.zeros(len(y), bool)
    sig = meta["sigma_bin"].to_numpy(float)
    extra = {"X": X, "y": y, "meta": meta, "logp": np.log10(meta["period"].to_numpy(float)),
             "snr": F.signal_snr(Xraw, sig), "failed_vetting": ~vet, "unused_vetted": vet & ~in_bench}
    return D, extra


def features(X, logp):
    S = shape_features(X)
    return {"shape": S, "shape_period": np.c_[S, logp], "period_only": logp[:, None]}


# ====================================================================== metrics

def metrics(y, p, w=None):
    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support
    lab = list(range(4))
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=lab, sample_weight=w, zero_division=0)
    return {"accuracy": float(np.average(y == p, weights=w)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p, sample_weight=w)),
            "macro_f1": float(f1_score(y, p, labels=lab, average="macro", sample_weight=w, zero_division=0)),
            "per_class": {NAMES[c]: {"precision": float(pr[c]), "recall": float(rc[c]), "f1": float(f1[c])}
                          for c in lab}}


def bootstrap_ci(star, y, p, rng):
    """95% CI of accuracy / balanced accuracy / macro-F1, resampling STARS (a star can be in several
    seeds' test sets; all its rows go in or out together)."""
    ids, inv = np.unique(star, return_inverse=True)
    out = {k: [] for k in ("accuracy", "balanced_accuracy", "macro_f1")}
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))[inv].astype(float)
        if w.sum() == 0:
            continue
        m = metrics(y, p, w)
        for k in out:
            out[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def mcnemar(y, pa, pb):
    """Paired test on the same test rows: do models a and b make different numbers of mistakes?"""
    from scipy.stats import binomtest
    a_only = int(((pa == y) & (pb != y)).sum())
    b_only = int(((pa != y) & (pb == y)).sum())
    n = a_only + b_only
    p = float(binomtest(a_only, n, 0.5).pvalue) if n else 1.0
    return {"right_only_first": a_only, "right_only_second": b_only, "p_value": p}


def binned_accuracy(values, ok, bins):
    rows = []
    for lo, hi, lab in bins:
        sel = (values >= lo) & (values < hi)
        rows.append({"bin": lab, "n": int(sel.sum()), "accuracy": float(ok[sel].mean()) if sel.any() else float("nan")})
    return rows


# ====================================================================== one seed

def run_seed(seed, D, E, epochs):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    X, y = D["X"], D["y"]
    tr, va, te = rt.split(y, seed)
    fit = np.r_[tr, va]
    mu, sd = D["logp"][tr].mean(), D["logp"][tr].std()
    z = lambda lp: ((lp - mu) / sd)[:, None].astype(np.float32)  # noqa: E731
    Z = z(D["logp"])
    Fb = features(X, D["logp"])
    Fe = features(E["X"], E["logp"])
    Ze = z(E["logp"])
    preds, extra = {}, {}

    for name, use_p in (("cnn_shape", False), ("cnn_period", True)):
        m = train(LightCurveCNN(4, n_extra=1 if use_p else 0), X[tr], y[tr], X[va], y[va], epochs=epochs, seed=seed,
                  verbose=False, extra=Z[tr] if use_p else None, extra_val=Z[va] if use_p else None)
        preds[name] = model_logits(m, X[te], Z[te] if use_p else None).argmax(1).numpy()
        extra[name] = model_logits(m, E["X"], Ze if use_p else None).argmax(1).numpy()
    for name, fs, make in (
            ("rf_shape", "shape", lambda: RandomForestClassifier(300, min_samples_leaf=2, random_state=seed, n_jobs=-1)),
            ("rf_shape_period", "shape_period",
             lambda: RandomForestClassifier(300, min_samples_leaf=2, random_state=seed, n_jobs=-1)),
            ("gb_shape_period", "shape_period", lambda: HistGradientBoostingClassifier(random_state=seed)),
            ("rf_period_only", "period_only",
             lambda: RandomForestClassifier(300, min_samples_leaf=2, random_state=seed, n_jobs=-1))):
        clf = make().fit(Fb[fs][fit], y[fit])
        preds[name] = clf.predict(Fb[fs][te])
        extra[name] = clf.predict(Fe[fs])

    r = {"metrics": {k: metrics(y[te], p) for k, p in preds.items()}}
    rows = pd.DataFrame({"seed": seed, "row": te, "tic": D["meta"]["tic"].to_numpy()[te], "true": y[te],
                         "tessmag": pd.to_numeric(D["meta"]["tessmag"], errors="coerce").to_numpy()[te],
                         "snr": D["snr"][te], **{f"pred_{k}": v for k, v in preds.items()}})
    erows = pd.DataFrame({"seed": seed, "tic": E["meta"]["tic"].to_numpy(), "true": E["y"],
                          "failed_vetting": E["failed_vetting"], "unused_vetted": E["unused_vetted"],
                          **{f"pred_{k}": v for k, v in extra.items()}})
    return r, rows, erows


# ====================================================================== outside check: Gao et al. 2025

def get_gao():
    """Table 5 of Gao et al. 2025 -> DataFrame (tic, gao_period, gao_prob, gao_type, gao_class)."""
    if not GAO_FILE.exists():
        EXT.mkdir(parents=True, exist_ok=True)
        print(f"Downloading the Gao et al. 2025 catalogue (14 MB) from {GAO_URL} ...")
        try:
            import urllib.request
            req = urllib.request.Request(GAO_URL, headers={"User-Agent": "Mozilla/5.0 (MantaRay research script)"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(GAO_FILE.with_suffix(".part"), "wb") as fh:
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    fh.write(block)
            GAO_FILE.with_suffix(".part").replace(GAO_FILE)
        except Exception as e:  # noqa: BLE001
            print(f"  download failed ({type(e).__name__}: {e}).\n  Download Table5_TESSVar.txt by hand from "
                  f"https://nadc.china-vo.org/res/r101482/ and save it as {GAO_FILE}; the outside check is skipped.")
            return None
    rows = []
    for line in GAO_FILE.read_text(errors="replace").splitlines():
        # CDS fixed-width table (byte positions from the file header): ID 1-10, Per 39-48,
        # classification probability 190-193, Type 195-203. Columns can be blank, so never split on spaces.
        if len(line) < 196 or not line[:10].strip().isdigit():
            continue                                  # header / notes
        typ = line[194:203].strip().upper()
        if typ not in GAO_TO_OURS:
            continue
        try:
            num = lambda a, b: float(line[a:b]) if line[a:b].strip() else float("nan")  # noqa: E731
            rows.append({"tic": int(line[:10]), "gao_period": num(38, 48), "gao_prob": num(189, 193),
                         "gao_bprp0": num(135, 142), "gao_type": typ, "gao_class": GAO_TO_OURS[typ]})
        except ValueError:
            continue
    if not rows:
        bad = GAO_FILE.with_suffix(".unreadable.txt")
        GAO_FILE.replace(bad)
        print(f"  could not read any rows from the downloaded file (moved to {bad}; it may be a web page instead of "
              f"the table).\n  Download Table5_TESSVar.txt by hand from https://nadc.china-vo.org/res/r101482/ and save "
              f"it as {GAO_FILE}; the outside check is skipped.")
        return None
    g = pd.DataFrame(rows).drop_duplicates("tic")
    print(f"Gao et al. 2025 catalogue: {len(g)} stars read")
    return g


def gao_check(gao, E, test_rows, unseen_meta):
    """Agreement of our labels / predictions with the Gao catalogue, matched by TIC."""
    ours = pd.DataFrame({"tic": E["meta"]["tic"].astype(int), "our_label": [NAMES[c] for c in E["y"]],
                         "our_period": E["meta"]["period"].astype(float), "vetted": ~E["failed_vetting"]})
    if unseen_meta is not None:
        ours = pd.concat([ours, pd.DataFrame({"tic": unseen_meta["tic"].astype(int), "our_label": unseen_meta["kind"],
                                              "our_period": unseen_meta["period"].astype(float),
                                              "vetted": np.nan})], ignore_index=True)
    m = ours.merge(gao, on="tic", how="inner")
    if len(m):
        ratio = m["our_period"] / m["gao_period"]
        m["period_match"] = np.select([np.abs(ratio - 1) < 0.02, np.abs(ratio - 2) < 0.04, np.abs(ratio - 0.5) < 0.01],
                                      ["same", "ours = 2x theirs", "ours = 1/2 theirs"], "different")
    out = {"n_our_stars": int(len(ours)), "n_matched": int(len(m)),
           "matched_by_our_label": m["our_label"].value_counts().to_dict(),
           "label_table": pd.crosstab(m["our_label"], m["gao_type"]).to_dict() if len(m) else {},
           "period_match_by_label": (m.groupby("our_label")["period_match"].value_counts(normalize=True)
                                     .unstack(fill_value=0).to_dict("index") if len(m) else {})}
    agree = {}
    for lab in ("EB", "DSCT", "ROT", "RR"):
        s = m[m["our_label"] == lab]
        c = s[s["gao_prob"] >= GAO_CONFIDENT]
        agree[lab] = {"n": int(len(s)), "same_class_in_gao": float((s["gao_class"] == lab).mean()) if len(s) else float("nan"),
                      "n_gao_confident": int(len(c)),
                      "same_class_when_gao_confident": float((c["gao_class"] == lab).mean()) if len(c) else float("nan"),
                      "their_types": s["gao_type"].value_counts().to_dict()}
    out["label_agreement"] = agree
    # our delta Scuti stars that Gao calls GCAS (gamma Cas = hot Be stars): period, colour and Gao's confidence
    g = m[(m["our_label"] == "DSCT") & (m["gao_type"] == "GCAS")]
    ref = m[(m["our_label"] == "DSCT") & (m["gao_class"] == "DSCT")]
    out["dsct_called_gcas"] = {"n": int(len(g)), "median_period_d": float(g["gao_period"].median()) if len(g) else None,
                               "median_bprp0": float(g["gao_bprp0"].median()) if len(g) else None,
                               "median_gao_prob": float(g["gao_prob"].median()) if len(g) else None,
                               "agreeing_dsct_median_bprp0": float(ref["gao_bprp0"].median()) if len(ref) else None,
                               "agreeing_dsct_median_gao_prob": float(ref["gao_prob"].median()) if len(ref) else None}
    q = m[m["our_label"] == "Noise"]
    out["quiet_stars_in_gao"] = {"n": int(len(q)), "types": q["gao_type"].value_counts().to_dict(),
                                 "tics": q["tic"].tolist()[:50]}
    # model predictions on the benchmark test stars vs Gao labels (Gao class in our 3 variable classes)
    t = test_rows.merge(gao, on="tic", how="inner")
    t = t[t["gao_class"].isin(["EB", "DSCT", "ROT"])]
    gy = t["gao_class"].map({n: i for i, n in enumerate(NAMES)}).to_numpy()
    out["prediction_agreement"] = {k: float((t[f"pred_{k}"].to_numpy() == gy).mean()) if len(t) else float("nan")
                                   for k in MODELS}
    out["prediction_agreement_n_rows"] = int(len(t))
    out["catalogue_label_agreement_on_these_rows"] = float((t["true"].to_numpy() == gy).mean()) if len(t) else float("nan")
    return out, m


# ====================================================================== summary

def summarise_all(runs, rows, erows, gao_out):
    S = {"n_seeds": len(runs), "models": list(MODELS), "class_names": NAMES}
    S["per_seed"] = {k: summarise(stack([r["metrics"][k] for r in runs])) for k in MODELS}
    rng = np.random.default_rng(0)
    S["pooled"] = {}
    for k in MODELS:
        p = rows[f"pred_{k}"].to_numpy()
        S["pooled"][k] = {**metrics(rows["true"].to_numpy(), p),
                          "ci95": bootstrap_ci(rows["tic"].to_numpy(), rows["true"].to_numpy(), p, rng)}
    best_classic = max(("rf_shape_period", "gb_shape_period"), key=lambda k: S["pooled"][k]["macro_f1"])
    S["mcnemar"] = {f"cnn_period vs {best_classic}": mcnemar(rows["true"].to_numpy(), rows["pred_cnn_period"].to_numpy(),
                                                             rows[f"pred_{best_classic}"].to_numpy()),
                    "cnn_period vs cnn_shape": mcnemar(rows["true"].to_numpy(), rows["pred_cnn_period"].to_numpy(),
                                                       rows["pred_cnn_shape"].to_numpy())}
    S["harder_stars"] = {}
    for subset in ("failed_vetting", "unused_vetted"):
        e = erows[erows[subset]]
        S["harder_stars"][subset] = {"n_stars": int(e["tic"].nunique()),
                                     "by_class_n": {NAMES[c]: int((e[e.seed == 0]["true"] == c).sum()) for c in range(4)},
                                     **{k: {"accuracy": float((e[f"pred_{k}"] == e["true"]).mean()) if len(e) else float("nan"),
                                            "by_class": {NAMES[c]: float((e[e.true == c][f"pred_{k}"] == c).mean())
                                                         for c in range(4) if (e.true == c).any()}}
                                        for k in MODELS}}
    S["by_magnitude"] = {k: binned_accuracy(rows["tessmag"].to_numpy(), (rows[f"pred_{k}"] == rows["true"]).to_numpy(),
                                            MAG_BINS) for k in ("cnn_period", "rf_shape_period", "cnn_shape")}
    var = rows["true"].to_numpy() > 0                     # S/N is meaningless for quiet stars
    S["by_snr"] = {k: binned_accuracy(rows["snr"].to_numpy()[var], (rows[f"pred_{k}"] == rows["true"]).to_numpy()[var],
                                      SNR_BINS) for k in ("cnn_period", "rf_shape_period", "cnn_shape")}
    S["gao"] = gao_out
    return S


LABELS = {"cnn_shape": "1D-CNN, shape only", "cnn_period": "1D-CNN + period (ours)", "rf_shape": "random forest, shape",
          "rf_shape_period": "random forest, shape + period", "gb_shape_period": "gradient boosting, shape + period",
          "rf_period_only": "random forest, period only"}


def print_summary(S):
    print(f"\n=== Benchmark over {S['n_seeds']} seeds: same stars, same splits (pooled test rows, 95% CI) ===")
    print(f"{'model':36s} {'accuracy':>22s} {'balanced acc.':>22s} {'macro-F1':>22s}")
    for k in MODELS:
        P = S["pooled"][k]
        f = lambda key: f"{100 * P[key]:5.1f} [{100 * P['ci95'][key][0]:5.1f}, {100 * P['ci95'][key][1]:5.1f}]"  # noqa: E731
        print(f"{LABELS[k]:36s} {f('accuracy'):>22s} {f('balanced_accuracy'):>22s} {f('macro_f1'):>22s}")
    print("\nPer-class F1 (pooled):")
    print(f"  {'model':36s} " + " ".join(f"{n:>7s}" for n in NAMES))
    for k in MODELS:
        print(f"  {LABELS[k]:36s} " + " ".join(f"{100 * S['pooled'][k]['per_class'][n]['f1']:6.1f}%" for n in NAMES))
    print("\nPaired McNemar tests (same test rows):")
    for name, t in S["mcnemar"].items():
        print(f"  {name:40s} right only with the first: {t['right_only_first']:3d}, only with the second: "
              f"{t['right_only_second']:3d}, p = {t['p_value']:.3f}")
    print("\nHarder stars (each seed's models; never used for training):")
    for subset, lab in (("failed_vetting", "stars that FAILED vetting"), ("unused_vetted", "vetted stars never used")):
        H = S["harder_stars"][subset]
        print(f"  {lab} ({H['n_stars']} stars: {H['by_class_n']})")
        for k in ("cnn_period", "rf_shape_period", "cnn_shape"):
            bc = ", ".join(f"{c} {100 * v:.0f}%" for c, v in H[k]["by_class"].items())
            print(f"    {LABELS[k]:34s} {100 * H[k]['accuracy']:5.1f}%   ({bc})")
    print("\nAccuracy by TESS magnitude (pooled test rows):")
    for k in ("cnn_period", "rf_shape_period"):
        print(f"  {LABELS[k]:34s} " + "  ".join(f"{r['bin']}: {100 * r['accuracy']:.0f}% (n={r['n']})"
                                                 for r in S["by_magnitude"][k] if r["n"]))
    print("Accuracy by S/N per bin (pooled test rows, variable stars only):")
    for k in ("cnn_period", "rf_shape_period"):
        print(f"  {LABELS[k]:34s} " + "  ".join(f"{r['bin']}: {100 * r['accuracy']:.0f}% (n={r['n']})"
                                                 for r in S["by_snr"][k] if r["n"]))
    G = S.get("gao")
    if G:
        print(f"\nOutside check, Gao et al. 2025: {G['n_matched']} of our {G['n_our_stars']} stars are in their "
              f"catalogue {G['matched_by_our_label']}")
        for lab, a in G["label_agreement"].items():
            if a["n"]:
                print(f"  our {lab:5s} stars: {100 * a['same_class_in_gao']:5.1f}% have the same class in Gao (n = {a['n']}); "
                      f"{100 * a['same_class_when_gao_confident']:5.1f}% where Gao is confident (prob >= {GAO_CONFIDENT}, "
                      f"n = {a['n_gao_confident']})   their types: {a['their_types']}")
        c = G.get("dsct_called_gcas")
        if c and c["n"]:
            print(f"  our δ Sct stars that Gao calls GCAS (gamma Cas, hot Be stars): {c['n']}; median period "
                  f"{c['median_period_d']:.3f} d, colour BP-RP0 {c['median_bprp0']:.2f} (agreeing δ Sct: "
                  f"{c['agreeing_dsct_median_bprp0']:.2f}), Gao probability {c['median_gao_prob']:.2f} "
                  f"(agreeing: {c['agreeing_dsct_median_gao_prob']:.2f})")
        q = G["quiet_stars_in_gao"]
        print(f"  our quiet stars listed as periodic variables by Gao: {q['n']} {q['types'] if q['n'] else ''}")
        print(f"  predictions vs Gao labels on test stars (n = {G['prediction_agreement_n_rows']} rows): " +
              ", ".join(f"{LABELS[k]} {100 * v:.1f}%" for k, v in G["prediction_agreement"].items()
                        if k in ("cnn_period", "rf_shape_period", "cnn_shape")) +
              f"; our catalogue labels {100 * G['catalogue_label_agreement_on_these_rows']:.1f}%")
        for lab, d in G["period_match_by_label"].items():
            print(f"  periods, our {lab}: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in d.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--gao-only", action="store_true", help="redo only the Gao et al. check (uses saved predictions)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.plots_only:
        S = json.loads((OUT / "results.json").read_text())
    elif a.gao_only:
        S = json.loads((OUT / "results.json").read_text())
        _, E = load_all()
        gao = get_gao()
        if gao is not None:
            um = rt.DATA / "unseen_meta.csv"
            S["gao"], matched = gao_check(gao, E, pd.read_csv(OUT / "test_predictions.csv"),
                                          pd.read_csv(um) if um.exists() else None)
            matched.to_csv(OUT / "gao_crossmatch.csv", index=False)
            (OUT / "results.json").write_text(json.dumps(S, indent=2, default=float))
    else:
        t0 = time.time()
        D, E = load_all()
        print(f"Benchmark stars: {len(D['y'])} vetted ({len(D['y']) // 4} per class); extra test stars: "
              f"{int(E['failed_vetting'].sum())} failed vetting, {int(E['unused_vetted'].sum())} vetted but never used")
        runs, rows, erows = [], [], []
        for seed in range(a.seeds):
            r, tr_rows, e_rows = run_seed(seed, D, E, a.epochs)
            runs.append(r)
            rows.append(tr_rows)
            erows.append(e_rows)
            print(f"seed {seed + 1}/{a.seeds}: " + " | ".join(
                f"{k} {r['metrics'][k]['macro_f1']:.3f}" for k in MODELS) + f"  (macro-F1) [{time.time() - t0:.0f} s]")
        rows, erows = pd.concat(rows, ignore_index=True), pd.concat(erows, ignore_index=True)
        rows.to_csv(OUT / "test_predictions.csv", index=False)
        gao = get_gao()
        gao_out = None
        if gao is not None:
            um = rt.DATA / "unseen_meta.csv"
            gao_out, matched = gao_check(gao, E, rows, pd.read_csv(um) if um.exists() else None)
            matched.to_csv(OUT / "gao_crossmatch.csv", index=False)
        S = summarise_all(runs, rows, erows, gao_out)
        S["runtime_seconds"] = round(time.time() - t0, 1)
        (OUT / "results.json").write_text(json.dumps(S, indent=2, default=float))
        table = pd.DataFrame([{"model": LABELS[k], **{m: round(100 * S["pooled"][k][m], 1)
                                                       for m in ("accuracy", "balanced_accuracy", "macro_f1")},
                               **{f"F1_{n}": round(100 * S["pooled"][k]["per_class"][n]["f1"], 1) for n in NAMES}}
                              for k in MODELS])
        table.to_csv(OUT / "benchmark_table.csv", index=False)
    rt._safe("fig19", report.benchmark, OUT / "fig19_benchmark.png", S, LABELS)
    if S.get("gao"):
        rt._safe("fig20", report.gao_check, OUT / "fig20_gao_check.png", S["gao"])
    print_summary(S)
    print(f"\nResults, paper-ready table (benchmark_table.csv) and figures in {OUT}/")


if __name__ == "__main__":
    main()
