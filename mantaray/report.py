"""Figures for the real-TESS results.

run_tess.py   : fig0 vetting, fig3 confusion matrix, fig5 overview, fig6 misclassified stars,
                fig7 significance, fig8 added-noise robustness, fig9 random-forest baseline,
                fig10 never-seen star types, fig11 detection limit, fig17 shape/period split
run_fixes.py  : fig13 shortcut tests, fig14 accuracy, fig15 "unknown" detectors,
                fig16 explanations, fig17 period term
benchmark.py  : fig19 benchmark, fig20 outside check against Gao et al. 2025
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402,F401
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mantaray.data import PHASE  # noqa: E402
from mantaray.evaluate import C1, C2, C3, C_RND, METHODS  # noqa: E402

INK, INK2 = "#0b0b0b", "#52514e"
METHOD_LABELS = ["Grad-CAM\nconv1", "Grad-CAM\nconv2", "Grad-CAM\nconv3", "Integrated\nGradients"]


def _paired_bars(ax, synth, tess, key, chance_key, title):
    """Synthetic vs TESS per method: bar = mean, dots = individual seeds, whisker = std."""
    w = 0.36
    for k, (res, color, label) in enumerate(((synth, C1, "synthetic"), (tess, C2, "real TESS"))):
        if res is None:
            continue
        x = np.arange(len(METHODS)) + (k - 0.5) * w
        means = [res["physics"][m][key]["mean"] for m in METHODS]
        stds = [res["physics"][m][key]["std"] for m in METHODS]
        ax.bar(x, means, w * 0.92, color=color, alpha=0.85, label=label, zorder=2)
        ax.errorbar(x, means, yerr=stds, fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
        for xi, m in zip(x, METHODS):
            v = np.array(res["physics"][m][key]["per_seed"], float)
            ax.scatter(np.full(len(v), xi), v, s=8, color=INK, alpha=0.5, lw=0, zorder=4)
        ch = res["physics"]["intgrad"][chance_key]["mean"]
        ax.plot([-0.5, len(METHODS) - 0.5], [ch, ch], ls=":", lw=1.2, color=color, zorder=1)
        ax.text(len(METHODS) - 0.48, ch, f"chance ({label}) {100 * ch:.0f}%", fontsize=7, color=INK2,
                va="center", ha="left")
    ax.set_xticks(range(len(METHODS)), METHOD_LABELS)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.6, len(METHODS) + 0.9)
    ax.set_title(title, loc="left")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))


def _crowding(ax, rows):
    """Dot plot (axis does not start at 0, so dots instead of bars)."""
    labels = ["isolated\nCROWDSAP ≥ 0.99", "mild blending\n0.90–0.99", "strong blending\n< 0.90"]
    x = np.arange(len(rows))
    series = (("accuracy", C_RND, "classification accuracy (all classes)", "o"),
              ("EB_peak_on_eclipse_intgrad", C2, "EB peak on eclipse: Integrated Gradients", "s"),
              ("EB_peak_on_eclipse_gradcam_conv2", C3, "EB peak on eclipse: Grad-CAM conv2", "D"))
    for key, color, label, marker in series:
        v = [r[key] for r in rows]
        ax.plot(x, v, marker=marker, color=color, lw=2, ms=7, label=label)
    for xi, r in zip(x, rows):
        ax.text(xi, 0.802, f"n = {r['n_star_tests']}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.79, 1.01)
    ax.set_xlim(-0.4, len(rows) - 0.6)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("(c) Blending: light from neighbouring stars in the aperture", loc="left")
    ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 0.06))


def _transfer(ax, transfer_preds, stars):
    """Accuracy by star type: TESS-trained model (test sets) vs synthetic-trained model (all stars)."""
    def groups(df, pred_col):
        det = df["detached_like"].astype(str).str.lower().eq("true")
        out = {}
        for name, sel in (("quiet stars", df.true == "Noise"), ("detached EBs", (df.true == "EB") & det),
                          ("contact-like EBs", (df.true == "EB") & ~det), ("delta Scuti", df.true == "DSCT")):
            s = df[sel]
            out[name] = (float((s[pred_col] == s.true).mean()) if len(s) else np.nan, len(s))
        return out
    series = [(groups(stars, "pred"), C2, "trained on real TESS stars"),
              (groups(transfer_preds, "pred_synthetic_model"), C1, "trained on synthetic: detached binaries only")]
    if "pred_synthetic_contact_model" in transfer_preds:
        series.append((groups(transfer_preds, "pred_synthetic_contact_model"), C3,
                       "trained on synthetic: contact binaries added"))
    names = list(series[0][0])
    y = np.arange(len(names))
    h = 0.8 / len(series)
    for k, (res, color, label) in enumerate(series):
        dy = (k - (len(series) - 1) / 2) * h
        ax.barh(y + dy, [res[n][0] for n in names], h * 0.92, color=color, label=label)
        for yi, n in zip(y, names):
            ax.text(min(res[n][0], 1.0) + 0.01, yi + dy, f"{100 * res[n][0]:.0f}%", va="center", fontsize=7, color=INK)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("classified correctly")
    ax.set_title("(d) Does the synthetic physics transfer to real stars?", loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2 if len(series) < 3 else 1)


def tess_overview(path, tess_summary, synth_summary, stars, transfer_preds):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), constrained_layout=True)
    _paired_bars(axes[0, 0], synth_summary, tess_summary, "EB_peak_on_eclipse",
                 "EB_share_eclipse_plus_contacts_chance",
                 "(a) Binaries: heatmap peak lands on an eclipse")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[0, 0].set_ylabel(f"share of stars (mean ± std over {tess_summary['n_seeds']} seeds)")
    _paired_bars(axes[0, 1], synth_summary, tess_summary, "DSCT_share_on_max_min",
                 "DSCT_share_on_max_min_chance",
                 "(b) δ Scuti: share of heatmap on light maximum / minimum")
    _crowding(axes[1, 0], tess_summary["crowding_pooled_over_seeds"])
    _transfer(axes[1, 1], transfer_preds, stars)
    n = tess_summary["dataset"]["counts"]
    label = {"Noise": "quiet", "EB": "binaries", "DSCT": "δ Scuti", "ROT": "rotators"}
    fig.suptitle("MantaRay on real TESS 2-minute light curves (" + ", ".join(f"{v} {label.get(k, k)}" for k, v in n.items())
                 + f"; {tess_summary['n_seeds']} seeds)", fontsize=11, x=0.01, ha="left")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def misclassified(path, X, meta, M, mis):
    """One panel per misclassified star: folded light curve, eclipse mask, labels and CROWDSAP."""
    if mis is None or not len(mis):
        return
    mis = mis.head(12)
    n = len(mis)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.8 * rows), squeeze=False, constrained_layout=True)
    index = {int(t): i for i, t in enumerate(meta["tic"])}
    for ax, r in zip(axes.flat, mis.itertuples()):
        i = index.get(int(r.tic))
        if i is None:
            ax.axis("off")
            continue
        if M[i].any():
            ax.fill_between(PHASE, 0, 1, where=M[i], transform=ax.get_xaxis_transform(), color="#e8e6e1", lw=0)
        ax.plot(PHASE, X[i], ".", ms=2.5, color=C1)
        m = meta.iloc[i]
        crowd = pd.to_numeric(m.get("crowdsap"), errors="coerce")
        ax.set_title(f"TIC {r.tic}: labelled {r.true}, model said {r.pred}\n"
                     f"P = {m['period']:.4g} d, own light in aperture = {100 * crowd:.0f}%, "
                     f"wrong in {r.times_misclassified} seed(s)", loc="left", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("phase", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Stars the model got wrong (shaded = detected eclipse)", fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def significance(path, sig):
    """Forest plot: per-star mean with 95% CI for every method and metric, against chance."""
    metrics = list(dict.fromkeys(sig["metric"]))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.4 * len(metrics), 3.4), sharey=True, constrained_layout=True)
    colors = dict(zip(METHODS, (C_RND, C3, C1, C2)))
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        d = sig[sig.metric == metric].set_index("method").reindex(METHODS)
        y = np.arange(len(METHODS))[::-1]
        for yi, m in zip(y, METHODS):
            r = d.loc[m]
            if not np.isfinite(r["mean"]):
                continue
            ax.errorbar(r["mean"], yi, xerr=[[r["mean"] - r["ci95_low"]], [r["ci95_high"] - r["mean"]]],
                        fmt="o", color=colors[m], ms=7, capsize=3, lw=2)
            ax.plot([r["chance"]] * 2, [yi - 0.3, yi + 0.3], color=INK2, lw=2)
            p = r["p_value_vs_chance"]
            ax.text(1.02, yi, "p < 1e-10" if p < 1e-10 else f"p = {p:.0e}", va="center", fontsize=7, color=INK2,
                    transform=ax.get_yaxis_transform())
        ax.set_yticks(y, [l.replace("\n", " ") for l in METHOD_LABELS])
        ax.set_xlim(0, 1.02)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(metric, loc="left", fontsize=9)
    np.atleast_1d(axes)[0].set_xlabel("dot = mean per star, bar = 95% CI, grey tick = chance")
    fig.suptitle("Are the explanations better than chance? (one-sided Wilcoxon test per star)",
                 fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _series(rows, key):
    return np.array([r[key]["mean"] for r in rows]), np.array([r[key]["std"] for r in rows])


def noise_robustness(path, rows, levels):
    x = np.array(levels)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    ax = axes[0]
    for key, color, label in (("accuracy_Noise", C_RND, "quiet stars"), ("accuracy_EB", C1, "binaries"),
                              ("accuracy_DSCT", C2, "δ Scuti"), ("accuracy_ROT", C3, "rotators")):
        if key not in rows[0]:
            continue
        m, s = _series(rows, key)
        ax.plot(x, m, "-o", color=color, lw=2, ms=5, label=label)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, lw=0)
    ax.set_title("(a) Classification accuracy", loc="left")
    ax.legend(frameon=False, fontsize=8)
    for ax, metric, title, chance_key in (
            (axes[1], "EB_peak_on_eclipse", "(b) Binaries: heatmap peak on eclipse", "EB_peak_chance"),
            (axes[2], "DSCT_share_on_max_min", "(c) δ Scuti: share on light max/min", None)):
        for name, color, label in (("intgrad", C2, "Integrated Gradients"), ("gradcam_conv2", C3, "Grad-CAM conv2")):
            m, s = _series(rows, f"{metric}_{name}")
            ax.plot(x, m, "-o", color=color, lw=2, ms=5, label=label)
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, lw=0)
        if chance_key:
            ch = np.nanmean(_series(rows, chance_key)[0])
            ax.axhline(ch, color=INK2, lw=1, ls=":")
            ax.text(x[-1], ch + 0.02, "chance", ha="right", fontsize=7, color=INK2)
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    for ax in axes:
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("added noise (× the curve's own scatter)")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle("Robustness: the real test stars made artificially noisier (as if fainter); mean ± std over seeds",
                 fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def baseline(path, summary):
    rf = summary["random_forest"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    label = {"Noise": "quiet stars", "EB": "binaries", "DSCT": "δ Scuti", "ROT": "rotators"}
    keys = list(summary["accuracy_by_class"])
    names = ["all stars"] + [label.get(k, k) for k in keys]
    cnn = [summary["test_accuracy"]] + [summary["accuracy_by_class"][k] for k in keys]
    rfa = [rf["accuracy"]] + [rf["accuracy_by_class"][k] for k in keys]
    y = np.arange(len(names))
    cnn_label = summary.get("model", "1D-CNN")
    rf_label = "random forest on " + summary.get("random_forest_inputs", "shape features")
    for k, (vals, color, label) in enumerate(((cnn, C2, cnn_label), (rfa, C1, rf_label))):
        dy = (k - 0.5) * 0.38
        ax.barh(y + dy, [v["mean"] for v in vals], 0.35, xerr=[v["std"] for v in vals], color=color, label=label,
                error_kw={"elinewidth": 1, "ecolor": INK})
        for yi, v in zip(y, vals):
            ax.text(min(v["mean"], 1) + 0.015, yi + dy, f"{100 * v['mean']:.1f}%", va="center", fontsize=7.5)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("(a) Test accuracy (mean ± std over seeds)", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax = axes[1]
    imp = {k: v["mean"] for k, v in rf["feature_importance"].items()}
    top = sorted(imp, key=imp.get)[-10:]
    ax.barh(range(len(top)), [imp[k] for k in top], color=C1)
    ax.set_yticks(range(len(top)), top)
    ax.set_xlabel("importance in the random forest")
    ax.set_title("(b) Which classic features the random forest relies on", loc="left")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def vetting(path, vet):
    ok = vet[vet.vet_status == "ok"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    kinds = [(k, l) for k, l in (("Noise", "quiet stars"), ("EB", "binaries"), ("DSCT", "δ Scuti"), ("ROT", "rotators"))
             if (ok.kind == k).any()]
    for i, (k, label) in enumerate(kinds):
        s = ok[ok.kind == k]
        n_pass = int(s.vet_pass.astype(str).str.lower().eq("true").sum())
        ax.barh(i, n_pass, color=C3)
        ax.barh(i, len(s) - n_pass, left=n_pass, color=C2)
        ax.text(len(s) + 5, i, f"{n_pass}/{len(s)} pass", va="center", fontsize=8)
    ax.set_yticks(range(len(kinds)), [l for _, l in kinds])
    ax.invert_yaxis()
    ax.set_xlabel("stars")
    ax.set_title("(a) Stars passing all vetting tests", loc="left")
    ax.legend(handles=[matplotlib.patches.Patch(color=C3, label="pass"), matplotlib.patches.Patch(color=C2, label="fail")],
              frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax = axes[1]
    fails = ok[~ok.vet_pass.astype(str).str.lower().eq("true")]
    ex = (fails.assign(reason=fails["vet_reason"].fillna("").astype(str).str.split("; "))
          .explode("reason").reset_index(drop=True))          # one row per (star, reason)
    ex = ex[ex.reason != ""]
    if len(ex):
        tab = pd.crosstab(ex.reason, ex.kind)
        tab = tab.loc[tab.sum(1).sort_values().index]
        left = np.zeros(len(tab))
        for k, color in (("EB", C1), ("DSCT", C2), ("ROT", C3), ("Noise", C_RND)):
            if k in tab:
                ax.barh(range(len(tab)), tab[k], left=left, color=color, label=k)
                left += tab[k].to_numpy()
        ax.set_yticks(range(len(tab)), [r if len(r) < 60 else r[:57] + "..." for r in tab.index], fontsize=8)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlabel("stars failing (a star can fail several tests)")
    ax.set_title("(b) Why stars failed", loc="left")
    fig.savefig(path, dpi=160)
    plt.close(fig)


UNSEEN_LABELS = {"GDOR": "γ Doradus", "RR": "RR Lyrae", "ELL": "ellipsoidal binaries", "DIP": "dip / transit stars",
                 "ROT": "rotators (held out)"}
CLASS_COLORS = {"Noise": C_RND, "EB": C1, "DSCT": C2, "ROT": C3}
CLASS_LABELS = {"Noise": "called quiet", "EB": "called binary", "DSCT": "called δ Scuti", "ROT": "called rotator"}
UNKNOWN_COLOR = "#eda100"


def open_set(path, O, names):
    """(a) what the model calls never-seen star types (confident calls vs 'unknown'); (b) separation."""
    types = list(O["types"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.75 * len(types) + 2.6), constrained_layout=True,
                             gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    y = np.arange(len(types))
    left = np.zeros(len(types))
    for c in names:
        v = np.array([O["types"][k][f"called_{c}_confidently"]["mean"] for k in types])
        ax.barh(y, v, left=left, color=CLASS_COLORS.get(c, C_RND), label=CLASS_LABELS.get(c, c) + " (confident)")
        for yi, (l, w) in enumerate(zip(left, v)):
            if w > 0.06:
                ax.text(l + w / 2, yi, f"{100 * w:.0f}%", ha="center", va="center", fontsize=7.5, color="white")
        left += v
    unk = np.array([O["types"][k]["flagged_unknown"]["mean"] for k in types])
    ax.barh(y, unk, left=left, color=UNKNOWN_COLOR, label='flagged "unknown"')
    for yi, (l, w) in enumerate(zip(left, unk)):
        if w > 0.06:
            ax.text(l + w / 2, yi, f"{100 * w:.0f}%", ha="center", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, [f"{UNSEEN_LABELS.get(k, k)} (n={O['types'][k]['n']['mean']:.0f})" for k in types])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("(a) What the model says about star types it never saw in training", loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax = axes[1]
    vals = [("separation of known vs never-seen\n(AUROC; 0.5 = chance, 1 = perfect)", O["auroc_known_vs_unseen"]),
            ("known test stars wrongly\nflagged 'unknown'", O["known_test_flagged_unknown"]),
            ("accuracy on known stars\nwhen confident", O["known_test_accuracy_when_confident"])]
    for i, (lab, d) in enumerate(vals):
        ax.barh(i, d["mean"], xerr=d["std"], color=[C1, UNKNOWN_COLOR, C2][i], error_kw={"ecolor": INK})
        ax.text(min(d["mean"] + d["std"], 1.05) + 0.03, i, f"{100 * d['mean']:.1f}%" if i else f"{d['mean']:.3f}",
                va="center", fontsize=8)
    ax.set_yticks(range(len(vals)), [v[0] for v in vals], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.2)
    ax.set_title("(b) Can confidence flag them?", loc="left")
    fig.suptitle('"Unknown" = less confident than 95% of the validation stars; mean over seeds',
                 fontsize=9, x=0.01, ha="left")
    fig.savefig(path, dpi=160)
    plt.close(fig)


IG_MIN_CALLED = 0.2   # fig11: hide the IG curve where fewer than 20% of injected eclipses were called binary


def detection_limit(path, rows, sigma_ppm, limits):
    snr = np.array([r["snr"]["mean"] for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), constrained_layout=True)
    for ax, series, title in (
            (axes[0], (("eclipse_called_EB", C1, "-", "called a binary"),
                       ("eclipse_called_variable", C1, "--", "called any variable class"),
                       ("eclipse_IG_peak_on_injected", C3, "-", "IG peak on the injected eclipse")),
             "(a) Injected eclipses"),
            (axes[1], (("sine_called_DSCT", C2, "-", "called δ Scuti"), ("sine_called_ROT", C3, "-", "called rotator"),
                       ("sine_called_variable", C2, "--", "called any variable class"),
                       ("rotsine_called_ROT", C3, "-", "rotator period: called rotator"),
                       ("rotsine_called_variable", C3, "--", "rotator period: called any variable class")),
             "(b) Injected pulsations (sinusoid + harmonic)")):
        two_arms = "rotsine_called_ROT" in rows[0]
        for key, color, ls, label in series:
            if key not in rows[0] or (two_arms and key == "sine_called_ROT"):
                continue
            if two_arms and key.startswith("sine_"):
                label = "δ Sct period: " + label
            m, s = _series(rows, key)
            if key == "eclipse_IG_peak_on_injected":
                # only plot where enough injected eclipses were called binary for the percentage to mean something
                few = _series(rows, "eclipse_called_EB")[0] < IG_MIN_CALLED
                m, s = np.where(few, np.nan, m), np.where(few, np.nan, s)
            ax.plot(snr, m, ls, marker="o", color=color, lw=2, ms=4, label=label)
            ax.fill_between(snr, m - s, m + s, color=color, alpha=0.12, lw=0)
        ax.axhline(0.5, color=INK2, lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.03)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlabel("injected depth or amplitude ÷ the star's own noise per phase bin (S/N)")
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, fontsize=7.5, loc="center right", bbox_to_anchor=(1, 0.3))
        top = ax.secondary_xaxis("top", functions=(lambda x: x * sigma_ppm, lambda x: x / sigma_ppm))
        top.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
        top.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        top.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        top.tick_params(labelsize=7.5)
        top.set_xlabel(f"in ppm for a typical quiet star (noise {sigma_ppm:.0f} ppm per bin)", fontsize=8)
    for ax, key in ((axes[0], "eclipse_called_EB"), (axes[1], "sine_called_variable")):
        if "rotsine_called_variable" in limits and key == "sine_called_variable":
            v2 = limits["rotsine_called_variable"]
            if v2 is not None and np.isfinite(v2):
                ax.axvline(v2, color=C3, lw=1, ls=":")
        v = limits.get(key)
        if v is not None and np.isfinite(v):
            ax.axvline(v, color=INK2, lw=1)
            ax.text(v / 1.06, 0.62, f"50% at S/N {v:.1f}\n= {v * sigma_ppm:.0f} ppm", fontsize=7.5, color=INK2,
                    ha="right")
    fig.suptitle("Detection limit: signals of known size injected into real quiet TESS stars; mean ± std over seeds",
                 fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _m(d):
    return d["mean"], d["std"]


def _faint_x(rows):
    """x positions for the 'made fainter' curves: the original stars are drawn at the right edge."""
    s = np.array([r["snr"]["mean"] for r in rows], float)
    fin = s[np.isfinite(s)]
    return np.where(np.isfinite(s), s, fin.max() * 2.5)


def fix_shortcut(path, S):
    """fig13: is the rotator / delta Scuti label decided by S/N, by period or by shape?"""
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9), constrained_layout=True)
    snr = np.array([r["snr"]["mean"] for r in S["plain"]["injection"]])
    ax = axes[0]
    for v in ("plain", "A_snr_matched"):
        m, s = _series(S[v]["injection"], "sine_dsct_period_called_ROT")
        ax.plot(snr, m, "-o", color=FIX_COLORS[v], lw=2, ms=4, label=FIX_LABELS[v])
        ax.fill_between(snr, m - s, m + s, color=FIX_COLORS[v], alpha=0.12, lw=0)
    ax.set_title("(a) Injected sinusoids, shape-only models", loc="left", fontsize=10)
    ax.set_ylabel('share called "rotator"')
    ax = axes[1]
    for v in ("B_period", "AB_both"):
        for arm, ls, lab in (("rot_period", "-", "rotator period"), ("dsct_period", "--", "δ Sct period")):
            m, s = _series(S[v]["injection"], f"sine_{arm}_called_ROT")
            ax.plot(snr, m, ls, marker="o", color=FIX_COLORS[v], lw=2, ms=4, label=f"{FIX_LABELS[v]}, {lab}")
            ax.fill_between(snr, m - s, m + s, color=FIX_COLORS[v], alpha=0.10, lw=0)
    ax.set_title("(b) Injected sinusoids, models with period", loc="left", fontsize=10)
    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_xlabel("injected amplitude ÷ noise per phase bin")
    for ax, cls, key, title in ((axes[2], "DSCT", "DSCT_called_DSCT", "(c) Real δ Scuti stars made fainter"),
                                (axes[3], "ROT", "ROT_called_ROT", "(d) Real rotators made fainter")):
        for v in FIX_VARIANTS:
            rows = S[v]["faint"]
            x = _faint_x(rows)
            m, s = _series(rows, key)
            ax.plot(x, m, "-o", color=FIX_COLORS[v], lw=2, ms=4, label=FIX_LABELS[v])
            ax.fill_between(x, m - s, m + s, color=FIX_COLORS[v], alpha=0.10, lw=0)
        if cls == "DSCT":
            m, _ = _series(S["plain"]["faint"], "DSCT_called_ROT")
            ax.plot(x, m, ":", color=C_RND, lw=1.5, label='shape only: called "rotator"')
        ax.set_xscale("log")
        ticks = list(x)
        ax.set_xticks(ticks, ["orig." if not np.isfinite(r["snr"]["mean"]) else f"{r['snr']['mean']:g}" for r in rows])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("S/N per bin (signal rms ÷ noise); each point:\nonly the stars bright enough to reach it")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel("share still called its own class" if cls == "DSCT" else "")
    for ax in axes:
        ax.set_ylim(0, 1.03)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle("Is the rotator / δ Scuti call decided by signal-to-noise, by period, or by shape? (mean ± std over seeds)",
                 fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fix_accuracy(path, S):
    """fig14: accuracy per class for the four variants, random-forest baselines, conflict test."""
    names = S["class_names"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.1), constrained_layout=True, gridspec_kw={"width_ratios": [1.2, 1.1, 0.9]})
    ax = axes[0]
    w = 0.2
    groups = names + ["all"]
    for k, v in enumerate(FIX_VARIANTS):
        x = np.arange(len(groups)) + (k - 1.5) * w
        vals = [S[v]["accuracy_by_class"][c] for c in names] + [S[v]["test_accuracy"]]
        ax.bar(x, [d["mean"] for d in vals], w * 0.92, color=FIX_COLORS[v], label=FIX_LABELS[v], zorder=2)
        ax.errorbar(x, [d["mean"] for d in vals], yerr=[d["std"] for d in vals], fmt="none", ecolor=INK,
                    elinewidth=1, capsize=2, zorder=3)
    ax.set_xticks(range(len(groups)), ["quiet", "binary", "δ Scuti", "rotator", "all"])
    ax.set_ylim(0.5, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("test accuracy")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    ax.set_title("(a) 1D-CNN variants", loc="left", fontsize=10)

    ax = axes[1]
    rf = S["random_forest"]
    rows = [(FIX_LABELS[v], S[v]["test_accuracy"], S[v]["accuracy_by_class"]["ROT"], FIX_COLORS[v]) for v in FIX_VARIANTS]
    rows += [("forest: shape", rf["shape"]["accuracy"], rf["shape"]["accuracy_by_class"]["ROT"], INK2),
             ("forest: shape + period", rf["shape_plus_period"]["accuracy"],
              rf["shape_plus_period"]["accuracy_by_class"]["ROT"], INK2),
             ("forest: period only", rf["period_only"]["accuracy"], rf["period_only"]["accuracy_by_class"]["ROT"], INK2)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, acc, rot, col) in zip(y, rows):
        ax.barh(yi + 0.18, acc["mean"], 0.34, color=col, alpha=0.9, zorder=2)
        ax.barh(yi - 0.18, rot["mean"], 0.34, color=col, alpha=0.45, hatch="//", zorder=2)
        ax.text(acc["mean"] + 0.005, yi + 0.18, f"{100 * acc['mean']:.1f}%", va="center", fontsize=7)
        ax.text(rot["mean"] + 0.005, yi - 0.18, f"{100 * rot['mean']:.1f}%", va="center", fontsize=7)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.set_xlim(0.4, 1.08)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(handles=[matplotlib.patches.Patch(color=INK2, label="all classes"),
                       matplotlib.patches.Patch(facecolor=INK2, alpha=0.45, hatch="//", label="rotators")],
              frameon=False, fontsize=7.5, loc="lower right")
    ax.set_title("(b) CNN vs random forest", loc="left", fontsize=10)

    ax = axes[2]
    C = S["B_period"].get("conflict")
    if C:
        A = np.array([[C[f"{c}_given_{d}_period"]["kept_own_class"]["mean"] for d in names] for c in names])
        im = ax.imshow(A, vmin=0, vmax=1, cmap="Blues")
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{100 * A[i, j]:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if A[i, j] > 0.6 else INK)
        lab = ["quiet", "binary", "δ Scuti", "rotator"]
        ax.set_xticks(range(4), lab, fontsize=8)
        ax.set_yticks(range(4), lab, fontsize=8)
        ax.set_xlabel("period taken from a star of this class")
        ax.set_ylabel("light-curve shape of this class")
        fig.colorbar(im, ax=ax, fraction=0.046, format=matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("(c) Conflict test (B: + period): share that\nkeeps its own class", loc="left", fontsize=10)
    fig.suptitle("Accuracy with the fixes (vetted TESS stars, mean ± std over seeds)", fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=160)
    plt.close(fig)


DETECTOR_LABELS = {"max_softmax": "max softmax", "energy": "energy", "mahalanobis": "Mahalanobis\n(features)",
                   "knn": "nearest neighbours\n(features)"}


def fix_unknown(path, S, ref3=None):
    """fig15: four 'unknown' detectors x four variants; per-type flags for the best combination."""
    dets = list(DETECTOR_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.0), constrained_layout=True, gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    w = 0.2
    best, best_v = None, -1
    for k, v in enumerate(FIX_VARIANTS):
        x = np.arange(len(dets)) + (k - 1.5) * w
        m = [S[v]["open_set"][d]["auroc_known_vs_unseen"]["mean"] for d in dets]
        s = [S[v]["open_set"][d]["auroc_known_vs_unseen"]["std"] for d in dets]
        ax.bar(x, m, w * 0.92, color=FIX_COLORS[v], label=FIX_LABELS[v], zorder=2)
        ax.errorbar(x, m, yerr=s, fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
        for d, val in zip(dets, m):
            if val > best_v:
                best, best_v = (v, d), val
    if ref3 is not None:
        ax.axhline(ref3, color=C2, lw=1, ls="--")
        ax.text(len(dets) - 0.5, ref3 + 0.006, f"3-class model, max softmax ({ref3:.3f})", fontsize=7, color=C2, ha="right")
    ax.axhline(0.5, color=INK2, lw=1, ls=":")
    ax.set_xticks(range(len(dets)), [DETECTOR_LABELS[d] for d in dets], fontsize=8)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("AUROC, known vs never-seen")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    ax.set_title("(a) How well each detector flags never-seen stars", loc="left", fontsize=10)

    ax = axes[1]
    v0, d0 = best
    pairs = ((("plain", "max_softmax"), C_RND, f"before: shape only, max softmax"),
             ((v0, d0), FIX_COLORS[v0], f"best: {FIX_LABELS[v0]}, {DETECTOR_LABELS[d0].replace(chr(10), ' ')}"))
    types = list(S["plain"]["open_set"]["max_softmax"]["types"])
    w = 0.38
    for k, ((v, d), col, lab) in enumerate(pairs):
        T = S[v]["open_set"][d]["types"]
        x = np.arange(len(types)) + (k - 0.5) * w
        m = [T[t]["flagged_unknown"]["mean"] for t in types]
        s = [T[t]["flagged_unknown"]["std"] for t in types]
        ax.bar(x, m, w * 0.92, color=col, label=lab, zorder=2)
        ax.errorbar(x, m, yerr=s, fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
        ax.axhline(S[v]["open_set"][d]["known_test_flagged_unknown"]["mean"], color=col, lw=1, ls="--")
    ax.plot([], [], "--", color=INK2, lw=1, label="known stars wrongly flagged")
    ax.set_xticks(range(len(types)), [UNSEEN_LABELS.get(t, t).replace(" ", "\n", 1) for t in types], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel('flagged "unknown"')
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_title('(b) Never-seen types flagged "unknown"', loc="left", fontsize=10)
    fig.suptitle('Detecting star types the model never saw ("unknown" = below the 5% validation quantile; mean ± std over seeds)',
                 fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fix_explanations(path, S):
    """fig16: do the explanations stay on the physics after the fixes?"""
    metrics = (("intgrad", "EB_peak_on_eclipse", None, "binary: IG peak\non the eclipse"),
               ("gradcam_conv2", "DSCT_share_on_max_min", "DSCT_share_on_max_min_chance", "δ Scuti: Grad-CAM conv2\non light max/min"),
               ("intgrad", "ROT_share_on_max_min", "ROT_share_on_max_min_chance", "rotator: IG\non light max/min"),
               ("gradcam_conv2", "ROT_share_on_max_min", "ROT_share_on_max_min_chance", "rotator: Grad-CAM conv2\non light max/min"))
    fig, ax = plt.subplots(figsize=(10, 3.8), constrained_layout=True)
    w = 0.2
    for k, v in enumerate(FIX_VARIANTS):
        x = np.arange(len(metrics)) + (k - 1.5) * w
        P = S[v]["physics"]
        m = [P[meth][key]["mean"] for meth, key, _, _ in metrics]
        s = [P[meth][key]["std"] for meth, key, _, _ in metrics]
        ax.bar(x, m, w * 0.92, color=FIX_COLORS[v], label=FIX_LABELS[v], zorder=2)
        ax.errorbar(x, m, yerr=s, fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
    for i, (meth, key, ck, _) in enumerate(metrics):
        c = S["plain"]["physics"]["intgrad"]["EB_share_eclipse_plus_contacts_chance" if ck is None else ck]["mean"]
        ax.plot([i - 0.45, i + 0.45], [c, c], color=INK, lw=1.2, ls=":", zorder=4)
    ax.plot([], [], ":", color=INK, lw=1.2, label="chance")
    ax.set_xticks(range(len(metrics)), [m[3] for m in metrics], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    ax.set_title("Do the explanations stay on the physics after the fixes? (correct test stars, mean ± std over seeds)",
                 loc="left", fontsize=10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


CLASS_FULL = {"Noise": "quiet", "EB": "binary", "DSCT": "δ Scuti", "ROT": "rotator"}
CLASS_COL = {"Noise": C_RND, "EB": C1, "DSCT": C2, "ROT": C3}


def fix_period(path, S, grid, ranges, variant="B_period", conflict=None):
    """fig17: what the period term learned, how each decision splits into shape and period
    (and, if given, the conflict test: stars given another class's period)."""
    T = S[variant]["period_terms"]
    names = S["class_names"]
    if conflict:
        fig, axes = plt.subplots(1, 3, figsize=(19, 4.2), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1.4, 1, 0.8]})
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    x = 10 ** np.asarray(grid)
    for c in names:
        m = np.array([d["mean"] for d in T["prior"][c]])
        s = np.array([d["std"] for d in T["prior"][c]])
        ax.plot(x, m, color=CLASS_COL[c], lw=2, label=CLASS_FULL[c])
        ax.fill_between(x, m - s, m + s, color=CLASS_COL[c], alpha=0.15, lw=0)
    rows = [(k, CLASS_COL[k], CLASS_FULL[k] + " (trained)") for k in names if k in ranges]
    rows += [(k, INK2, UNSEEN_LABELS.get(k, k) + " (never seen)") for k in ("RR", "GDOR", "ELL", "DIP") if k in ranges]
    for i, (k, col, lab) in enumerate(rows):
        yb = -0.08 - 0.07 * i
        lo, hi = ranges[k]
        ax.plot([10 ** lo, 10 ** hi], [yb, yb], color=col, lw=4, solid_capstyle="butt")
        ax.text(10 ** hi * 1.08, yb, lab, fontsize=6.5, va="center", color=col)
    ax.set_xscale("log")
    ax.set_xlim(x.min(), x.max() * 6)
    ax.set_ylim(-0.1 - 0.07 * len(rows), 1.03)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("period (days)")
    ax.set_ylabel("P(class | period alone)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=2)
    ax.set_title("(a) What the period term learned (bars: 5–95% period range of each type)", loc="left", fontsize=10)

    ax = axes[1]
    w = 0.38
    B = T["by_class"]
    for k, (key, col, lab) in enumerate((("shape_alone_correct", C1, "light-curve shape alone"),
                                         ("period_alone_correct", C2, "period alone"))):
        xx = np.arange(len(names)) + (k - 0.5) * w
        m = [B[c][key]["mean"] for c in names]
        s = [B[c][key]["std"] for c in names]
        ax.bar(xx, m, w * 0.92, color=col, label=lab, zorder=2)
        ax.errorbar(xx, m, yerr=s, fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
    share = [B[c]["period_share_of_margin"]["mean"] for c in names]
    ax.plot(np.arange(len(names)), share, "D", color=INK, ms=5, zorder=4, label="period's share of the decision margin")
    ax.set_xticks(range(len(names)), [CLASS_FULL[c] for c in names])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("share of correctly classified test stars")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    ax.set_title("(b) Would each half have got the star right on its own?", loc="left", fontsize=10)
    if conflict:
        ax = axes[2]
        A = np.array([[conflict[f"{c}_given_{d}_period"]["kept_own_class"]["mean"] for d in names] for c in names])
        im = ax.imshow(A, vmin=0, vmax=1, cmap="Blues")
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{100 * A[i, j]:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if A[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(names)), [CLASS_FULL[c] for c in names], fontsize=8)
        ax.set_yticks(range(len(names)), [CLASS_FULL[c] for c in names], fontsize=8)
        ax.set_xlabel("period taken from a star of this class")
        ax.set_ylabel("light-curve shape of this class")
        fig.colorbar(im, ax=ax, fraction=0.046, format=matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title("(c) Conflict test: share that keeps its own class", loc="left", fontsize=10)
    fig.suptitle("Period input: logit = shape term + period term, so every decision splits exactly in two "
                 "(mean ± std over seeds)", fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def benchmark(path, S, labels):
    models = S["models"]
    names = S["class_names"]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.25, 1.1, 0.9, 1.0]})
    ax = axes[0]
    y = np.arange(len(models))[::-1]
    for yi, k in zip(y, models):
        P = S["pooled"][k]
        lo, hi = P["ci95"]["macro_f1"]
        ax.barh(yi, P["macro_f1"], 0.6, color=BENCH_COL[k], zorder=2)
        ax.errorbar(P["macro_f1"], yi, xerr=[[P["macro_f1"] - lo], [hi - P["macro_f1"]]], fmt="none", ecolor=INK,
                    elinewidth=1, capsize=2, zorder=3)
        ax.plot(P["accuracy"], yi, "|", color=INK, ms=12, mew=1.5, zorder=4)
        ax.text(max(hi, P["accuracy"]) + 0.005, yi, f"{100 * P['macro_f1']:.1f}%", va="center", fontsize=7.5)
    ax.set_yticks(y, [labels[k] for k in models], fontsize=8)
    ax.set_xlim(0.4, 1.06)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("macro-F1 (bar, 95% CI);  | = accuracy")
    ax.set_title("(a) Same stars, same splits", loc="left", fontsize=10)

    ax = axes[1]
    key4 = ["cnn_shape", "cnn_period", "rf_shape_period", "gb_shape_period"]
    w = 0.2
    for k_i, k in enumerate(key4):
        x = np.arange(len(names)) + (k_i - 1.5) * w
        ax.bar(x, [S["pooled"][k]["per_class"][n]["f1"] for n in names], w * 0.92, color=BENCH_COL[k],
               label=labels[k], zorder=2)
    ax.set_xticks(range(len(names)), [CLASS_FULL[n] for n in names])
    ax.set_ylim(0.6, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("F1 per class")
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    ax.set_title("(b) Per class", loc="left", fontsize=10)

    ax = axes[2]
    subsets = (("failed_vetting", "failed\nvetting"), ("unused_vetted", "vetted,\nnever used"))
    k3 = ["cnn_shape", "cnn_period", "rf_shape_period"]
    w = 0.26
    for k_i, k in enumerate(k3):
        x = np.arange(len(subsets)) + (k_i - 1) * w
        v = [S["harder_stars"][s][k]["accuracy"] for s, _ in subsets]
        ax.bar(x, v, w * 0.92, color=BENCH_COL[k], label=labels[k], zorder=2)
        for xi, vi in zip(x, v):
            ax.text(xi, vi + 0.01, f"{100 * vi:.0f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(subsets)), [f"{l}\n({S['harder_stars'][s]['n_stars']} stars)" for s, l in subsets], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("accuracy")
    ax.set_title("(c) Stars never used for training", loc="left", fontsize=10)

    ax = axes[3]
    for k in ("cnn_shape", "cnn_period", "rf_shape_period"):
        rows = [r for r in S["by_magnitude"][k] if r["n"] >= 10]
        ax.plot(range(len(rows)), [r["accuracy"] for r in rows], "-o", color=BENCH_COL[k], lw=2, ms=4, label=labels[k])
    ax.set_xticks(range(len(rows)), [f"{r['bin']}\n(n={r['n']})" for r in rows], fontsize=7.5)
    ax.set_ylim(0.7, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("TESS magnitude (fainter →)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_title("(d) Bright vs faint stars", loc="left", fontsize=10)
    fig.suptitle(f"Fair benchmark on vetted TESS stars ({S['n_seeds']} seeds; pooled test stars; star-level bootstrap CIs)",
                 fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def gao_check(path, G):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True, gridspec_kw={"width_ratios": [0.9, 1.4, 1.0]})
    ax = axes[0]
    A = {k: v for k, v in G["label_agreement"].items() if v["n"]}
    labs = list(A)
    w = 0.38
    for j, (key, nkey, alpha, lab) in enumerate((("same_class_in_gao", "n", 0.45, "all matched stars"),
                                                  ("same_class_when_gao_confident", "n_gao_confident", 1.0,
                                                   "where Gao is confident (prob ≥ 0.5)"))):
        x = np.arange(len(labs)) + (j - 0.5) * w
        v = [A[k].get(key, np.nan) for k in labs]
        ax.bar(x, v, w * 0.92, color=[CLASS_COL.get(k, INK2) for k in labs], alpha=alpha, zorder=2,
               label=lab, edgecolor="none")
        for xi, k, vi in zip(x, labs, v):
            if np.isfinite(vi):
                ax.text(xi, vi + 0.02, f"{100 * vi:.0f}%\n{A[k].get(nkey, 0)}", ha="center", fontsize=6.5)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2)
    q = G["quiet_stars_in_gao"]["n"]
    ax.set_xticks(range(len(labs)), [CLASS_FULL.get(k, UNSEEN_LABELS.get(k, k)) for k in labs], fontsize=8)
    ax.set_xlabel(f"our quiet stars listed as periodic variables by Gao: {q}", fontsize=8)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("same class in Gao et al. 2025 (numbers = stars)")
    ax.set_title("(a) Our catalogue labels vs Gao et al.", loc="left", fontsize=10)

    ax = axes[1]
    T = pd.DataFrame(G["label_table"]).fillna(0)
    if len(T):
        order_r = [r for r in ("EB", "DSCT", "ROT", "RR", "GDOR", "ELL", "DIP", "Noise") if r in T.index]
        order_c = [c for c in ("EA", "EB", "EW", "DSCT", "HADS", "ROT", "RRAB", "RRCD", "CEPHEIDS", "GCAS", "UV", "YSO")
                   if c in T.columns]
        T = T.loc[order_r, order_c]
        Rn = T.div(T.sum(1), axis=0)
        im = ax.imshow(Rn.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for i in range(Rn.shape[0]):
            for j in range(Rn.shape[1]):
                if T.iat[i, j]:
                    ax.text(j, i, f"{int(T.iat[i, j])}", ha="center", va="center", fontsize=7,
                            color="white" if Rn.iat[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(order_c)), order_c, fontsize=7.5, rotation=45)
        ax.set_yticks(range(len(order_r)), [f"{CLASS_FULL.get(r, UNSEEN_LABELS.get(r, r))} (n={int(T.loc[r].sum())})"
                                            for r in order_r], fontsize=7.5)
        ax.set_xlabel("type in Gao et al. 2025")
        fig.colorbar(im, ax=ax, fraction=0.04, format=matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("(b) Matched stars: our label (rows) vs their type (numbers = stars)", loc="left", fontsize=10)

    ax = axes[2]
    P = G["prediction_agreement"]
    ks = ["cnn_shape", "cnn_period", "rf_shape_period"]
    ax.bar(range(len(ks)), [P[k] for k in ks], color=[BENCH_COL[k] for k in ks], zorder=2)
    for i, k in enumerate(ks):
        ax.text(i, P[k] - 0.03, f"{100 * P[k]:.1f}%", ha="center", fontsize=8, color="white")
    c = G["catalogue_label_agreement_on_these_rows"]
    ax.axhline(c, color=INK, ls="--", lw=1)
    ax.text(-0.45, c + 0.008, f"our catalogue labels ({100 * c:.1f}%)", ha="left", fontsize=7, color=INK)
    ax.set_xticks(range(len(ks)), ["CNN, shape", "CNN + period", "forest,\nshape + period"], fontsize=8)
    ax.set_ylim(0.5, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("agrees with Gao et al. label")
    ax.set_title(f"(c) Model predictions vs Gao (test rows, n={G['prediction_agreement_n_rows']})", loc="left", fontsize=10)
    fig.suptitle(f"Outside check: {G['n_matched']} of our {G['n_our_stars']} stars are in the Gao et al. 2025 TESS "
                 "catalogue (their labels are machine-made too: agreement, not ground truth)", fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def confusion(path, summary):
    """fig3: confusion matrix summed over seeds (rows = true class), with per-class recall and overall accuracy."""
    names = summary["class_names"]
    C = np.array(summary["confusion_summed_over_seeds(rows=true)"], float)
    R = C / C.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.6), constrained_layout=True)
    im = ax.imshow(R, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{int(C[i, j])}\n{100 * R[i, j]:.1f}%", ha="center", va="center", fontsize=8,
                    color="white" if R[i, j] > 0.6 else INK)
    lab = [CLASS_FULL.get(n, n) for n in names]
    ax.set_xticks(range(len(names)), lab)
    ax.set_yticks(range(len(names)), lab)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    fig.colorbar(im, ax=ax, fraction=0.046, format=matplotlib.ticker.PercentFormatter(1.0))
    acc = summary["test_accuracy"]
    ax.set_title(f"Test stars, all {summary.get('n_seeds', '?')} seeds pooled: accuracy {100 * acc['mean']:.1f} ± "
                 f"{100 * acc['std']:.1f}%\n({int(np.trace(C))} of {int(C.sum())} correct; {summary.get('model', '')})",
                 loc="left", fontsize=9)
    fig.savefig(path, dpi=160)
    plt.close(fig)
