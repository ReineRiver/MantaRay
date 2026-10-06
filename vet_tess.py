"""Vet every star in the TESS dataset against false variability.

For each star in data/tess/tess_meta.csv the same TESS sector is downloaded again and the tests in
mantaray/vetting.py are run: split-half (dust, one-off glitches), odd-even (wrong period),
centroid source offset (light from a neighbouring star) and a "no secondary eclipse" shape flag.
Results: data/tess/vetting.jsonl (resumable log) and data/tess/tess_vetting.csv (one row per
dataset row, same order as tess_meta.csv).

Usage: see README.md.
"""
import argparse
import json
import socket
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import build_tess_dataset as B
from mantaray.tess import fold_bin
from mantaray.vetting import vet_star

LOG = B.DATA / "vetting.jsonl"


def vet_one(row, flux_row, mask_bins, keep_fits):
    rec = {"kind": row.kind, "tic": int(row.tic)}
    try:
        sr = B.search_2min(f"TIC {int(row.tic)}")
        if len(sr) == 0:
            return {**rec, "vet_status": "error", "vet_reason": "no SPOC 2-min light curve found"}
        lc = B.download_first(sr, sector=row.sector if pd.notna(row.sector) else None)
        cols = lc.colnames if hasattr(lc, "colnames") else []
        cc = lc["centroid_col"] if "centroid_col" in cols else np.full(len(lc), np.nan)
        cr = lc["centroid_row"] if "centroid_row" in cols else np.full(len(lc), np.nan)
        primary = int(np.argmin(np.where(np.isin(np.arange(len(flux_row)), mask_bins), flux_row, np.inf))) \
            if len(mask_bins) else None
        v = vet_star(row.kind, lc.time, lc.flux, cc, cr, float(row.period), row.crowdsap, mask_bins, primary,
                     row.mask_fraction, row.depth2_sigma, row.ls_freq)
        # sanity check: re-folding must reproduce the dataset row (same data, same phase zero-point)
        from mantaray.tess import clean
        t, f = clean(lc.time, lc.flux)
        if t is not None:
            b, _ = fold_bin(t, f, float(row.period))
            v["refold_correlation"] = float(np.corrcoef(b, flux_row)[0, 1])
        del lc
        B.cleanup(int(row.tic), keep_fits)
        return {**rec, **v}
    except Exception as e:
        return {**rec, "vet_status": "error", "vet_reason": f"{type(e).__name__}: {e}"[:300]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--keep-fits", action="store_true")
    args = ap.parse_args()
    B.CACHE.mkdir(parents=True, exist_ok=True)
    B._LOGFILE = (B.DATA / "vetting_log.txt").open("a", encoding="utf-8")
    sys.stdout, sys.stderr = B._MainThreadOnly(sys.stdout), B._MainThreadOnly(sys.stderr)
    socket.setdefaulttimeout(B.NETWORK_TIMEOUT)
    from astroquery import log as aq_log
    aq_log.setLevel("WARNING")
    B.say("Progress is also written to data/tess/vetting_log.txt\n")

    meta = pd.read_csv(B.DATA / "tess_meta.csv")
    flux = pd.read_csv(B.DATA / "tess_dataset.csv").iloc[:, 1:].to_numpy(float)
    masks = np.load(B.DATA / "tess_masks.npy")
    done = {}
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            r = json.loads(line)
            if r.get("vet_status") != "error":  # errors are retried; later lines override earlier ones
                done[(r["kind"], int(r["tic"]))] = r
    todo = [i for i, r in enumerate(meta.itertuples()) if (r.kind, int(r.tic)) not in done]
    B.say(f"{len(meta)} stars in the dataset, {len(done)} already vetted, {len(todo)} to do")

    def job(i):
        return i, vet_one(meta.iloc[i], flux[i], np.flatnonzero(masks[i]).tolist(), args.keep_fits)

    n_done = len(done)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for k in range(0, len(todo), 4 * args.workers):
            for i, rec in pool.map(job, todo[k:k + 4 * args.workers]):
                with B._lock:
                    with LOG.open("a") as fh:
                        fh.write(json.dumps(rec, default=float) + "\n")
                if rec.get("vet_status") != "error":
                    done[(rec["kind"], rec["tic"])] = rec
                    n_done += 1
                tag = "PASS" if rec.get("vet_pass") else ("ERR " if rec.get("vet_status") == "error" else "FAIL")
                B.log(f"[{n_done:4d}/{len(meta)}] {tag} {rec['kind']:5s} TIC {rec['tic']}  {rec.get('vet_reason', '')}")

    # export, aligned with tess_meta.csv
    rows = [done.get((r.kind, int(r.tic)), {"kind": r.kind, "tic": int(r.tic), "vet_status": "missing"})
            for r in meta.itertuples()]
    out = pd.DataFrame(rows)
    out.to_csv(B.DATA / "tess_vetting.csv", index=False)
    ok = out[out.vet_status == "ok"]
    B.say(f"\nVetted {len(ok)}/{len(out)} stars. Passed all tests:")
    for k in ("EB", "DSCT", "ROT", "Noise"):
        s = ok[ok.kind == k]
        if len(s):
            B.say(f"  {k:5s} {int(s.vet_pass.sum()):4d} / {len(s)}")
    if "refold_correlation" in ok:
        bad = (ok.refold_correlation < 0.9).sum()
        B.say(f"Re-fold check: {bad} stars did not reproduce their dataset row (correlation < 0.9)")
    reasons = ok.loc[~ok.vet_pass.astype(bool), "vet_reason"].str.split("; ").explode().value_counts()
    if len(reasons):
        B.say("Why stars failed:\n" + reasons.to_string())
    B.say(f"\nWrote {B.DATA}/tess_vetting.csv")


if __name__ == "__main__":
    main()
