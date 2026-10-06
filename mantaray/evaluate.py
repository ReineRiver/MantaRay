"""Shared evaluation: explain a test set, score it against the physics, summarise seeds, plot.

Used by run_experiment.py (synthetic) and run_tess.py (real TESS stars), so both are
scored and drawn in exactly the same way.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mantaray.data import CLASS_NAMES, PHASE
from mantaray.explain import (GradCAM1D, dilate, extrema_share, integrated_gradients, mask_mass,
                              pointing_game, spread)

LAYERS = ("conv1", "conv2", "conv3")  # 200, 100, 50 cells; receptive field ~7, ~18, ~50 bins
METHODS = tuple(f"gradcam_{l}" for l in LAYERS) + ("intgrad",)

# colours: categorical slots validated for colour-blind separation; heat = single-hue ramp
C1, C2, C3, C_RND = "#2a78d6", "#eb6834", "#1baf7a", "#9a9890"
HEAT = "Blues"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b"})


# ====================================================================== explanations + metrics

def explain_all(model, X, y):
    S = {}
    for layer in LAYERS:
        g = GradCAM1D(model, getattr(model, layer))
        S[f"gradcam_{layer}"] = g(X, y)
        g.remove()
    S["intgrad"] = integrated_gradients(model, X, y)
    return S


def physics_metrics(S, X, y, M, ok, eb_sel=None):
    """H1 / H2 metrics for one saliency method, each with its random-chance baseline.

    eb_sel: optional boolean array restricting which EBs are scored (e.g. detached EBs
    with a reliable eclipse mask). Defaults to all correctly classified EBs.
    """
    eb = ok & (y == 1) if eb_sel is None else ok & (y == 1) & eb_sel
    ds = ok & (y == 2)
    near = dilate(M, 4)  # eclipse + 4 bins either side (contact points with the flat baseline)
    ext_share, ext_rand = extrema_share(S[ds], X[ds])
    nan = float("nan")
    f = lambda a: float(a.mean()) if a.size else nan  # noqa: E731
    extra = {}
    if (y == 3).any():   # ROT (spotted rotators): light max/min, like the pulsators
        rt = ok & (y == 3)
        r_share, r_rand = extrema_share(S[rt], X[rt]) if rt.any() else (np.array([]), np.array([]))
        extra = {"ROT_share_on_max_min": f(r_share), "ROT_share_on_max_min_chance": f(r_rand),
                 "n_ROT_scored": int(rt.sum())}
    return {**extra,
        "EB_share_in_eclipse": f(mask_mass(S[eb], M[eb])),
        "EB_share_in_eclipse_chance": f(M[eb].mean(1)),
        "EB_share_eclipse_plus_contacts": f(mask_mass(S[eb], near[eb])),
        "EB_share_eclipse_plus_contacts_chance": f(near[eb].mean(1)),
        "EB_peak_on_eclipse": f(pointing_game(S[eb], near[eb])),
        "DSCT_share_on_max_min": f(ext_share),
        "DSCT_share_on_max_min_chance": f(ext_rand),
        "n_EB_scored": int(eb.sum()),
        "n_DSCT_scored": int(ds.sum()),
        "spread": {CLASS_NAMES[c]: f(spread(S[ok & (y == c)])) for c in range(3)},
    }


def confusion(y, pred, k=3):
    return [[int(((y == i) & (pred == j)).sum()) for j in range(k)] for i in range(k)]


# ====================================================================== seed aggregation

def stack(runs):
    """List of identically shaped nested dicts -> same dict with lists of per-seed values at the leaves."""
    first = runs[0]
    if isinstance(first, dict):
        return {k: stack([r[k] for r in runs]) for k in first}
    if isinstance(first, list):
        return [stack([r[i] for r in runs]) for i in range(len(first))]
    return list(runs)


def summarise(x):
    if isinstance(x, dict):
        return {k: summarise(v) for k, v in x.items()}
    if isinstance(x, list) and x and isinstance(x[0], (list, dict)):
        return [summarise(v) for v in x]
    a = np.asarray(x, float)
    ok = np.isfinite(a)
    m = float(a[ok].mean()) if ok.any() else float("nan")
    s = float(a[ok].std(ddof=1)) if ok.sum() > 1 else 0.0
    return {"mean": m, "std": s, "per_seed": a.tolist()}


def deletion_summary(runs):
    out, arrays = {}, {}
    for cls in runs[0]["deletion"]:
        arrays[cls] = {k: np.array([r["deletion"][cls][k] for r in runs]) for k in runs[0]["deletion"][cls]}
        out[cls] = {k: {"mean": v.mean(0).tolist(),
                        "std": (v.std(0, ddof=1) if len(v) > 1 else 0 * v[0]).tolist()}
                    for k, v in arrays[cls].items()}
    return out, arrays


def pm(d):
    return f"{100 * d['mean']:5.1f} ± {100 * d['std']:4.1f}%"


def print_physics_table(P):
    print(f"\n{'method':16s} {'EB peak on eclipse':>20s} {'EB share in eclipse':>20s} {'DSCT share max/min':>20s}")
    for m in METHODS:
        print(f"{m:16s} {pm(P[m]['EB_peak_on_eclipse']):>20s} {pm(P[m]['EB_share_in_eclipse']):>20s} "
              f"{pm(P[m]['DSCT_share_on_max_min']):>20s}")
    c = P["intgrad"]
    print(f"{'chance':16s} {pm(c['EB_share_eclipse_plus_contacts_chance']):>20s} "
          f"{pm(c['EB_share_in_eclipse_chance']):>20s} {pm(c['DSCT_share_on_max_min_chance']):>20s}")


# ====================================================================== figures

def curve(ax, x, s, title, mask=None):
    if mask is not None and mask.any():
        ax.fill_between(PHASE, 0, 1, where=mask, transform=ax.get_xaxis_transform(), color="#e8e6e1", lw=0)
    sc = ax.scatter(PHASE, x, c=s, cmap=HEAT, vmin=0, vmax=1, s=9, lw=0)
    ax.set_title(title, loc="left")
    ax.set_xlim(0, 1)
    return sc


def band(ax, x, arr, color, label, ls="-"):
    m, s = arr.mean(0), (arr.std(0, ddof=1) if len(arr) > 1 else 0 * arr[0])
    ax.plot(x, m, ls, marker="o", color=color, lw=2, ms=4, label=label)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, lw=0)


def fig_examples(path, X, y, M, S, ok, idx=None, titles=None, class_names=CLASS_NAMES):
    """3 x n grid: rows = explanation method, columns = class. idx: optional star index per class."""
    rows = (("gradcam_conv3", "Grad-CAM (conv3)"), ("gradcam_conv2", "Grad-CAM (conv2)"),
            ("intgrad", "Integrated Gradients"))
    full = {"Noise": "Quiet star", "EB": "Eclipsing binary", "DSCT": "δ Scuti star", "ROT": "Spotted rotator"}
    n = len(class_names)
    fig, axes = plt.subplots(3, n, figsize=(3.4 * n, 7), sharex=True, constrained_layout=True, squeeze=False)
    for c in range(n):
        cand = np.flatnonzero(ok & (y == c))
        if not len(cand):
            continue
        i = idx[c] if idx is not None else cand[min(2, len(cand) - 1)]
        name = full.get(class_names[c], class_names[c])
        if titles is not None:
            name += f"\n{titles[i]}"
        for r, (key, label) in enumerate(rows):
            sc = curve(axes[r, c], X[i], S[key][i], name if r == 0 else "", M[i])
        axes[-1, c].set_xlabel("phase")
    for r, (_, label) in enumerate(rows):
        axes[r, 0].set_ylabel(f"{label}\nnormalised flux")
    fig.colorbar(sc, ax=axes, shrink=0.5, label="importance for the decision (0 to 1)")
    fig.suptitle("What the 1D-CNN looks at in each folded light curve (darker = more important; "
                 "grey band = eclipse)", fontsize=10, x=0.01, ha="left")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_deletion(path, D, fractions):
    fig, axes = plt.subplots(1, len(D), figsize=(8, 3), sharey=True, constrained_layout=True)
    f = 100 * np.array(fractions)
    for ax, (name, d) in zip(np.atleast_1d(axes), D.items()):
        band(ax, f, d["gradcam_conv3"], C1, "Grad-CAM conv3")
        band(ax, f, d["gradcam_conv2"], C3, "Grad-CAM conv2")
        band(ax, f, d["intgrad"], C2, "Integrated Gradients")
        band(ax, f, d["random"], C_RND, "random order", "--")
        ax.set_title(name, loc="left")
        ax.set_xlabel("% of phase bins flattened")
    axes = np.atleast_1d(axes)
    axes[0].set_ylabel("P(true class), mean ± std over seeds")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(frameon=False, loc="lower left")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_stability(path, P):
    labels = ["Grad-CAM\nconv1", "Grad-CAM\nconv2", "Grad-CAM\nconv3", "Integrated\nGradients"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
    for ax, key, chance, title in (
            (axes[0], "EB_peak_on_eclipse", "EB_share_eclipse_plus_contacts_chance", "EB: heatmap peak on eclipse"),
            (axes[1], "DSCT_share_on_max_min", "DSCT_share_on_max_min_chance", "DSCT: share on light max/min")):
        for i, m in enumerate(METHODS):
            v = np.array(P[m][key]["per_seed"], float)
            v = v[np.isfinite(v)]
            ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, s=22, color=C1, lw=0, alpha=0.8)
            if len(v):
                ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color="#0b0b0b", lw=2)
        ax.axhline(P["intgrad"][chance]["mean"], color="#52514e", lw=1, ls=":")
        ax.text(3.4, P["intgrad"][chance]["mean"] + 0.02, "chance", ha="right", fontsize=8, color="#52514e")
        ax.set_xticks(range(4), labels)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("one dot per seed; bar = mean")
    fig.savefig(path, dpi=160)
    plt.close(fig)
