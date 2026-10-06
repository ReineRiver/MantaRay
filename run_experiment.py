"""Synthetic pilot: does a 1D-CNN classify phase-folded light curves for physical reasons?

For each seed (fresh synthetic data and a fresh model):
  1. train the CNN on phase-folded light curves (quiet / eclipsing binary / delta Scuti);
  2. explain the test set with Grad-CAM (conv1, conv2, conv3) and Integrated Gradients;
  3. score the explanations against the physics and against chance:
       eclipse attribution (binaries), light-maximum/minimum attribution (delta Scuti),
       deletion test (faithfulness), model-randomisation sanity check;
  4. shortcut dose-response: a fake glitch is planted on every delta Scuti training star at
     increasing strength; the script measures how much each model relies on it and whether
     the explanations point at it.
Results are reported as mean +/- standard deviation over seeds.

Usage: see README.md.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mantaray.data import CLASS_NAMES, N_BINS, add_shortcut, make_dataset
from mantaray.evaluate import (C2, C3, C_RND, band, curve, deletion_summary, explain_all, fig_deletion,
                               fig_examples, fig_stability, physics_metrics, pm, print_physics_table, stack,
                               summarise)
from mantaray.explain import (GradCAM1D, deletion_curve, dilate, integrated_gradients, mask_mass, pointing_game,
                              sanity_check, spearman)
from mantaray.model import LightCurveCNN, accuracy, predict, proba, train

OUT = Path("outputs")

# shortcut experiment settings
LOW_SNR = (0.7, 4)                     # hard regime, where a shortcut is tempting
GLITCH_START, GLITCH_WIDTH = 20, 3     # glitch position in the TEST curves (bins 20-22)
GLITCH_AMPS = (0, 1, 2, 3, 5)          # strength in units of the normalised curve's std

# ====================================================================== one seed

def shortcut_dose(seed, keep):
    """Train one model per glitch strength (low SNR) and measure reliance vs explanation."""
    b = 1000 * (seed + 1)
    Xtr, ytr, _ = make_dataset(1200, seed=b + 11, snr=LOW_SNR)
    Xva, yva, _ = make_dataset(300, seed=b + 12, snr=LOW_SNR)
    Xte, yte, _ = make_dataset(500, seed=b + 13, snr=LOW_SNR)
    glitch = np.zeros((1, N_BINS), bool)
    glitch[0, GLITCH_START:GLITCH_START + GLITCH_WIDTH] = True
    hit_zone = dilate(glitch, 2)                       # glitch +/- 2 bins
    ds, other = yte == 2, yte != 2
    rows, example, clean = [], None, None
    gz = hit_zone.repeat(ds.sum(), 0)

    def peaks(model, X):
        g = GradCAM1D(model, model.conv2)
        s_gc = g(X, yte[ds])
        g.remove()
        s_ig = integrated_gradients(model, X, yte[ds])
        return s_gc, s_ig

    for amp in GLITCH_AMPS:
        put = lambda X, y, cls=2: add_shortcut(X, y, cls, GLITCH_START, GLITCH_WIDTH, amp)  # noqa: E731
        m = train(LightCurveCNN(), put(Xtr, ytr), ytr, put(Xva, yva), yva, epochs=8, seed=seed, verbose=False)
        clean = clean or m  # amp 0 = model trained without any glitch
        Xg = put(Xte, yte)                              # glitch on the DSCT test stars only
        Xall = Xte + amp * glitch.astype(np.float32)    # glitch on EVERY test star
        # independent measures of reliance (no explanation method involved)
        p_with = proba(m, Xg[ds])[:, 2]
        p_without = proba(m, Xte[ds])[:, 2]
        flip = (predict(m, Xall[other]) == 2) & (predict(m, Xte[other]) != 2)
        # explanations of the glitched DSCT stars, by the shortcut model and (control) by the clean model
        s_gc, s_ig = peaks(m, Xg[ds])
        c_gc, c_ig = peaks(clean, Xg[ds])
        row = {
            "glitch_amp": amp,
            "acc_clean_test": accuracy(m, Xte, yte),
            "reliance_flip_rate": float(flip.mean()),
            "reliance_prob_drop": float((p_with - p_without).mean()),
            "gradcam_conv2_peak_on_glitch": float(pointing_game(s_gc, gz).mean()),
            "intgrad_peak_on_glitch": float(pointing_game(s_ig, gz).mean()),
            "gradcam_conv2_glitch_share": float(mask_mass(s_gc, glitch).mean()),
            "intgrad_glitch_share": float(mask_mass(s_ig, glitch).mean()),
            "clean_model_gradcam_conv2_peak_on_glitch": float(pointing_game(c_gc, gz).mean()),
            "clean_model_intgrad_peak_on_glitch": float(pointing_game(c_ig, gz).mean()),
            "clean_model_gradcam_conv2_glitch_share": float(mask_mass(c_gc, glitch).mean()),
            "clean_model_intgrad_glitch_share": float(mask_mass(c_ig, glitch).mean()),
            "glitch_share_chance": GLITCH_WIDTH / N_BINS,
            "peak_on_glitch_chance": float(hit_zone.mean()),
        }
        if amp > 0:  # per star: does a bigger glitch share go with a bigger effect of the glitch?
            row["per_star_spearman_gradcam_conv2"] = spearman(mask_mass(s_gc, glitch), p_with - p_without)
            row["per_star_spearman_intgrad"] = spearman(mask_mass(s_ig, glitch), p_with - p_without)
        rows.append(row)
        if keep and amp == max(GLITCH_AMPS):
            i = int(np.argmax(np.abs(Xg[ds]).max(1) < 3.5 + amp))
            example = (Xg[ds][i], s_gc[i], s_ig[i], glitch[0])
    return rows, example


def run_seed(seed, keep=False):
    b = 1000 * (seed + 1)
    Xtr, ytr, _ = make_dataset(2000, seed=b + 1)
    Xva, yva, _ = make_dataset(500, seed=b + 2)
    Xte, yte, Mte = make_dataset(600, seed=b + 3)
    t1 = time.time()
    model = train(LightCurveCNN(), Xtr, ytr, Xva, yva, seed=seed, verbose=False)
    r = {"train_seconds": round(time.time() - t1, 1), "test_accuracy": accuracy(model, Xte, yte)}
    pred = predict(model, Xte)
    ok = pred == yte  # explain correctly classified stars only
    S = explain_all(model, Xte, yte)
    r["physics"] = {name: physics_metrics(s, Xte, yte, Mte, ok) for name, s in S.items()}
    r["deletion"] = {}
    for c in (1, 2):
        sel = ok & (yte == c)
        d = {n: deletion_curve(model, Xte[sel], yte[sel], S[n][sel], seed=seed)
             for n in ("gradcam_conv2", "gradcam_conv3", "intgrad")}
        r["deletion"][CLASS_NAMES[c]] = {**{n: v["salient"] for n, v in d.items()}, "random": d["intgrad"]["random"]}
    r["sanity_spearman_vs_random_model"] = {
        l: sanity_check(model, l, Xte[ok][:1000], yte[ok][:1000], seed=seed) for l in ("conv2", "conv3")}
    r["shortcut_dose_response"], ex_short = shortcut_dose(seed, keep)
    ex = (Xte, yte, Mte, S, ok, ex_short) if keep else None
    return r, ex


def print_table(summary, n):
    print(f"\n=== Summary over {n} seed(s): mean ± std ===")
    print(f"Test accuracy: {pm(summary['test_accuracy'])}")
    print_physics_table(summary["physics"])
    print("\nShortcut dose-response (low SNR): reliance vs explanation")
    ch = summary["shortcut_dose_response"][0]["glitch_share_chance"]["mean"]
    x = lambda d: f"{d['mean'] / ch:5.1f} ± {d['std'] / ch:4.1f}x"  # noqa: E731
    print("  glitch enrichment = share of the heatmap on the glitch / share expected by chance (1x)")
    print(f"{'glitch amp':>10s} {'flip rate':>16s} {'Grad-CAM enrich':>16s} {'IG enrich':>16s} "
          f"{'clean GC enrich':>16s} {'clean IG enrich':>16s} {'Grad-CAM peak':>16s} {'IG peak':>16s}")
    for row in summary["shortcut_dose_response"]:
        print(f"{row['glitch_amp']['mean']:10.0f} {pm(row['reliance_flip_rate']):>16s} "
              f"{x(row['gradcam_conv2_glitch_share']):>16s} {x(row['intgrad_glitch_share']):>16s} "
              f"{x(row['clean_model_gradcam_conv2_glitch_share']):>16s} "
              f"{x(row['clean_model_intgrad_glitch_share']):>16s} "
              f"{pm(row['gradcam_conv2_peak_on_glitch']):>16s} {pm(row['intgrad_peak_on_glitch']):>16s}")
    print(f"(chance for 'peak on glitch': {pm(summary['shortcut_dose_response'][0]['peak_on_glitch_chance'])})")


# ====================================================================== figures

def fig_shortcut(rows_per_seed, example):
    amps = np.array(GLITCH_AMPS, float)
    get = lambda k: np.array([[row[k] for row in rows] for rows in rows_per_seed])  # noqa: E731
    ch = GLITCH_WIDTH / N_BINS
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), constrained_layout=True)
    ax = axes[0, 0]
    band(ax, amps, 100 * get("reliance_flip_rate"), C_RND, "EB / Noise stars flipped to DSCT", "--")
    ax.set_title("(a) How much the model really relies on the glitch", loc="left")
    ax.set_xlabel("planted glitch strength (× curve std)")
    ax.set_ylabel("% flipped by adding the glitch")
    ax.set_ylim(0, 100)
    ax = axes[0, 1]
    band(ax, amps, get("intgrad_glitch_share") / ch, C2, "Integrated Gradients, shortcut model")
    band(ax, amps, get("gradcam_conv2_glitch_share") / ch, C3, "Grad-CAM conv2, shortcut model")
    band(ax, amps, get("clean_model_intgrad_glitch_share") / ch, C2, "Integrated Gradients, clean model", ":")
    band(ax, amps, get("clean_model_gradcam_conv2_glitch_share") / ch, C3, "Grad-CAM conv2, clean model", ":")
    ax.axhline(1, color="#52514e", lw=1, ls=":")
    ax.text(amps[0], 1.15, "chance", fontsize=8, color="#52514e")
    ax.set_yscale("log")
    ax.set_title("(b) Does the explanation point at the glitch?", loc="left")
    ax.set_xlabel("planted glitch strength (× curve std)")
    ax.set_ylabel("heatmap share on glitch ÷ chance")
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    if example is not None:
        x, s_gc, s_ig, glitch = example
        curve(axes[1, 0], x, s_gc, f"(c) Grad-CAM conv2, shortcut model, glitch {max(GLITCH_AMPS)}×", glitch)
        curve(axes[1, 1], x, s_ig, f"(d) Integrated Gradients, shortcut model, glitch {max(GLITCH_AMPS)}×", glitch)
        for a in axes[1]:
            a.set_xlabel("phase")
        axes[1, 0].set_ylabel("normalised flux")
    fig.savefig(OUT / "fig3_shortcut.png", dpi=160)
    plt.close(fig)


# ====================================================================== main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    runs, ex = [], None
    for seed in range(args.seeds):
        ts = time.time()
        r, e = run_seed(seed, keep=(seed == 0))
        runs.append(r)
        ex = ex or e
        print(f"seed {seed + 1}/{args.seeds}: accuracy {r['test_accuracy']:.3f}, "
              f"model trained in {r['train_seconds']} s, seed total {time.time() - ts:.0f} s")

    summary = summarise(stack([{k: v for k, v in r.items() if k != "deletion"} for r in runs]))
    summary["deletion_fractions"] = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
    summary["deletion"], D = deletion_summary(runs)
    summary["n_seeds"] = args.seeds
    summary["runtime_seconds"] = round(time.time() - t0, 1)

    Xte, yte, Mte, S, ok, ex_short = ex
    fig_examples(OUT / "fig1_examples.png", Xte, yte, Mte, S, ok)
    fig_deletion(OUT / "fig2_deletion.png", D, summary["deletion_fractions"])
    fig_shortcut([r["shortcut_dose_response"] for r in runs], ex_short)
    fig_stability(OUT / "fig4_stability.png", summary["physics"])
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    print_table(summary, args.seeds)
    print(f"\nTotal runtime {summary['runtime_seconds']} s. Figures and results.json in {OUT}/")


if __name__ == "__main__":
    main()
