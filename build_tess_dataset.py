"""Build the MantaRay dataset from TESS 2-minute light curves (MAST archive + VizieR catalogues).

Stages (all resumable):
  1. Catalogues
       EB    : TESS Eclipsing Binary catalogue, Prsa et al. 2022, ApJS 258, 16 (VizieR J/ApJS/258/16)
       DSCT  : AAVSO VSX (VizieR B/vsx/vsx), types DSCT, DSCTC and HADS, brighter than mag 12
       ROT and never-seen test types (GDOR, RR, ELL): AAVSO VSX, brighter than mag 12
       Noise : TESS 2-min targets near the DSCT stars, not in VSX or the EB catalogue, passing the
               quiet-star tests in mantaray/tess.py
  2. Download one SPOC 2-minute sector per star (PDCSAP flux), process it (mantaray/tess.py) and
     append the result to data/tess/processed.jsonl; stars already processed are skipped.
  3. Export the accepted stars, balanced across classes:
       data/tess/tess_dataset.csv   (column 0 = label, columns 1-200 = folded flux)
       data/tess/tess_meta.csv      (TIC, sector, period, CROWDSAP, ... per row)
       data/tess/tess_masks.npy     (eclipse mask per row; all False for non-EBs)
       data/tess/unseen_dataset.csv, unseen_meta.csv   (never-seen test types)

Usage: see README.md.
"""
import argparse
import json
import random
import re
import shutil
import socket
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from mantaray.data import N_BINS, TESS_CLASSES
from mantaray.tess import EB_PERIOD_RANGE, UNSEEN_KINDS, process_star

warnings.filterwarnings("ignore")  # astropy/lightkurve are chatty; failures are logged per star

DATA = Path("data/tess")
CACHE = DATA / "_fits_cache"
PROCESSED = DATA / "processed.jsonl"
QUIET_REFOLD = DATA / "quiet_refold.jsonl"   # optional re-processed quiet-star records (see apply_quiet_refold)
REFOLD_KEYS = ("flux", "period", "sigma_bin", "point_noise", "n_points", "ls_freq", "ls_snr", "slow_freq",
               "slow_snr", "period_source", "min_points_per_bin")
DSCT_MAX_MAG = 12.0
LABEL = {name: i for i, name in enumerate(TESS_CLASSES)}  # Noise 0, EB 1, DSCT 2, ROT 3
VSX_MAX_MAG = 12.0
VSX_TYPES = {                       # exact VSX types only (no ':' uncertain, no hybrids)
    "ROT": ["ROT", "BY"],           # spotted rotating stars (BY Draconis type)
    "GDOR": ["GDOR"],               # gamma Doradus pulsators          (never-seen test type)
    "RR": ["RRAB", "RRC"],          # RR Lyrae pulsators               (never-seen test type)
    "ELL": ["ELL"],                 # ellipsoidal (non-eclipsing) binaries (never-seen test type)
}
ALL_KINDS = list(TESS_CLASSES) + list(UNSEEN_KINDS)
_lock = threading.Lock()
_CONSOLE = sys.stdout          # the real console; only log() writes to it
_LOGFILE = None
NETWORK_TIMEOUT = 60           # seconds; a stalled download raises instead of hanging forever


class _MainThreadOnly:
    """Stand-in for sys.stdout/stderr: drops anything printed by worker threads.

    The download libraries print progress bars from every worker thread. On Windows a
    console can freeze (e.g. after a click selects text) and then every print blocks,
    which stalls all downloads. Worker output is useless anyway, so it is discarded.
    """

    def __init__(self, stream):
        self.stream = stream

    def write(self, text):
        if threading.current_thread() is threading.main_thread():
            try:
                return self.stream.write(text)
            except UnicodeEncodeError:
                return self.stream.write(text.encode("ascii", "replace").decode())
        return len(text)

    def flush(self):
        if threading.current_thread() is threading.main_thread():
            self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


# ====================================================================== small helpers

def norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_col(columns, *, exact=(), contains=(), exclude=("err", "e_", "unc", "sig", "flag")):
    """First column whose normalised name equals one of `exact`, else contains one of `contains`."""
    cols = list(columns)
    for c in cols:
        if norm(c) in exact:
            return c
    for c in cols:
        n = norm(c)
        if any(k in n for k in contains) and not any(norm(x) in n for x in exclude if x):
            return c
    return None


def retry(fn, tries=3, wait=5):
    for k in range(tries):
        try:
            return fn()
        except Exception:  # network hiccups: back off and retry
            if k == tries - 1:
                raise
            time.sleep(wait * (k + 1))


def say(msg=""):
    """Main-thread status message: straight to the real console + log file (never via sys.stdout,
    which third-party code may swap out)."""
    log(msg, stamp=False)


def log(msg, stamp=True):
    with _lock:
        line = f"{time.strftime('%H:%M:%S')} {msg}" if stamp else str(msg)
        try:
            _CONSOLE.write(line + "\n")
            _CONSOLE.flush()
        except UnicodeEncodeError:
            _CONSOLE.write(line.encode("ascii", "replace").decode() + "\n")
        if _LOGFILE:
            _LOGFILE.write(line + "\n")
            _LOGFILE.flush()


# ====================================================================== 1. catalogues

def load_eb_catalogue():
    path = DATA / "catalogue_eb.csv"
    if path.exists():
        return pd.read_csv(path)
    from astroquery.vizier import Vizier
    say("Downloading TESS EB catalogue (Prsa et al. 2022) from VizieR ...")
    tables = retry(lambda: Vizier(columns=["**"], row_limit=-1).get_catalogs("J/ApJS/258/16"))
    best = None
    for key in tables.keys():
        df = tables[key].to_pandas()
        tic = find_col(df.columns, exact=("tic", "ticid", "tessid"), contains=("tic",))
        per = find_col(df.columns, exact=("per", "period", "p"), contains=("period",))
        say(f"  table {key}: {len(df)} rows; TIC column = {tic}, period column = {per}")
        if tic and per and (best is None or len(df) > len(best[0])):
            best = (df, tic, per)
    if best is None:
        for key in tables.keys():
            say(f"  columns of {key}: {list(tables[key].colnames)}")
        sys.exit("Could not find TIC + period columns. Paste the lines above to Claude.")
    df, tic, per = best
    morph = find_col(df.columns, contains=("morph",))
    out = pd.DataFrame({"tic": pd.to_numeric(df[tic], errors="coerce"),
                        "period": pd.to_numeric(df[per], errors="coerce"),
                        "morph": pd.to_numeric(df[morph], errors="coerce") if morph else np.nan})
    out = out.dropna(subset=["tic", "period"]).astype({"tic": "int64"}).drop_duplicates("tic")
    out = out[(out.period >= EB_PERIOD_RANGE[0]) & (out.period <= EB_PERIOD_RANGE[1])]
    say(f"  kept {len(out)} EBs with {EB_PERIOD_RANGE[0]} <= P <= {EB_PERIOD_RANGE[1]} d "
          f"(morphology column: {morph})")
    out.to_csv(path, index=False)
    return out


def load_dsct_catalogue():
    path = DATA / "catalogue_dsct_v2.csv"  # v2: adds DSCTC, magnitude limit 12.0
    if path.exists():
        return pd.read_csv(path)
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astroquery.vizier import Vizier
    say("Downloading delta Scuti stars from VSX (VizieR B/vsx/vsx) ...")
    frames = []
    for patterns in (("DSCT*", "HADS*"), ("=DSCT", "=HADS"), ("DSCT", "HADS")):  # VizieR filter syntaxes
        for pattern in patterns:
            try:
                t = retry(lambda: Vizier(columns=["**"], row_limit=-1,
                                         column_filters={"Type": pattern}).get_catalogs("B/vsx/vsx"))
                if len(t):
                    frames.append(t[0].to_pandas())
                    say(f"  Type filter {pattern!r}: {len(t[0])} rows")
            except Exception as e:
                say(f"  Type filter {pattern!r}: query failed ({e})")
        if frames:
            break
    if not frames:
        sys.exit("VSX query returned nothing. Paste the lines above to Claude.")
    df = pd.concat(frames, ignore_index=True)
    c_name = find_col(df.columns, exact=("name",))
    c_type = find_col(df.columns, exact=("type",))
    c_mag = find_col(df.columns, exact=("max", "vmax", "magmax"))
    c_ra = find_col(df.columns, exact=("raj2000", "ra", "radeg"), contains=("raj2000",))
    c_de = find_col(df.columns, exact=("dej2000", "dec", "dedeg"), contains=("dej2000",))
    say(f"  columns used: name={c_name}, type={c_type}, mag={c_mag}, ra={c_ra}, dec={c_de}")
    if not all([c_name, c_type, c_ra, c_de]):
        sys.exit(f"Missing VSX columns. Columns are: {list(df.columns)} -- paste this to Claude.")
    typ = df[c_type].astype(str).str.strip()
    keep = typ.isin(["DSCT", "DSCTC", "HADS", "HADS(B)"])  # exact types only: no ':' (uncertain), no hybrids
    if c_mag:
        keep &= pd.to_numeric(df[c_mag], errors="coerce") < DSCT_MAX_MAG
    df = df[keep]
    ra, de = df[c_ra], df[c_de]
    if ra.dtype == object:  # sexagesimal strings
        sc = SkyCoord(ra.astype(str).values, de.astype(str).values, unit=(u.hourangle, u.deg))
        ra, de = sc.ra.deg, sc.dec.deg
    out = pd.DataFrame({"name": df[c_name].astype(str).values, "type": typ[keep].values,
                        "ra": np.asarray(ra, float), "dec": np.asarray(de, float),
                        "mag": pd.to_numeric(df[c_mag], errors="coerce").values if c_mag else np.nan})
    out = out.drop_duplicates("name")
    say(f"  kept {len(out)} DSCT/HADS stars brighter than {DSCT_MAX_MAG}")
    out.to_csv(path, index=False)
    return out


def load_vsx_bright():
    """All VSX stars brighter than VSX_MAX_MAG (one download, cached), for the ROT and test types."""
    path = DATA / "vsx_bright.csv"
    if path.exists():
        return pd.read_csv(path)
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astroquery.vizier import Vizier
    say(f"Downloading all VSX variables brighter than mag {VSX_MAX_MAG} (VizieR B/vsx/vsx) ...")
    df = None
    try:
        t = retry(lambda: Vizier(columns=["**"], row_limit=-1,
                                 column_filters={"max": f"<{VSX_MAX_MAG}"}).get_catalogs("B/vsx/vsx"))
        if len(t):
            df = t[0].to_pandas()
            say(f"  {len(df)} rows")
    except Exception as e:
        say(f"  magnitude-filtered query failed ({e}); falling back to one query per type")
    if df is None or not len(df):
        frames = []
        for types in VSX_TYPES.values():
            for typ in types:
                try:
                    t = retry(lambda: Vizier(columns=["**"], row_limit=-1,
                                             column_filters={"Type": f"{typ}*"}).get_catalogs("B/vsx/vsx"))
                    if len(t):
                        frames.append(t[0].to_pandas())
                except Exception as e:
                    say(f"  Type {typ}: query failed ({e})")
        if not frames:
            sys.exit("VSX query returned nothing. Paste the lines above to Claude.")
        df = pd.concat(frames, ignore_index=True)
    c_name, c_type = find_col(df.columns, exact=("name",)), find_col(df.columns, exact=("type",))
    c_mag = find_col(df.columns, exact=("max", "vmax", "magmax"))
    c_per = find_col(df.columns, exact=("period", "per"))
    c_ra = find_col(df.columns, exact=("raj2000", "ra", "radeg"), contains=("raj2000",))
    c_de = find_col(df.columns, exact=("dej2000", "dec", "dedeg"), contains=("dej2000",))
    say(f"  columns used: name={c_name}, type={c_type}, mag={c_mag}, period={c_per}, ra={c_ra}, dec={c_de}")
    if not all([c_name, c_type, c_ra, c_de]):
        sys.exit(f"Missing VSX columns. Columns are: {list(df.columns)} -- paste this to Claude.")
    mag = pd.to_numeric(df[c_mag], errors="coerce") if c_mag else pd.Series(np.nan, index=df.index)
    df = df[(mag < VSX_MAX_MAG) | mag.isna() & (c_mag is None)]
    ra, de = df[c_ra], df[c_de]
    if ra.dtype == object:
        sc = SkyCoord(ra.astype(str).values, de.astype(str).values, unit=(u.hourangle, u.deg))
        ra, de = sc.ra.deg, sc.dec.deg
    out = pd.DataFrame({"name": df[c_name].astype(str).values, "type": df[c_type].astype(str).str.strip().values,
                        "ra": np.asarray(ra, float), "dec": np.asarray(de, float),
                        "mag": pd.to_numeric(df[c_mag], errors="coerce").values if c_mag else np.nan,
                        "period": pd.to_numeric(df[c_per], errors="coerce").values if c_per else np.nan})
    out = out.drop_duplicates("name")
    out.to_csv(path, index=False)
    for kind, types in VSX_TYPES.items():
        say(f"  {kind:5s} ({', '.join(types)}): {int(out.type.isin(types).sum())} stars")
    return out


def vsx_jobs(vsx, kind, rng):
    sel = vsx[vsx.type.isin(VSX_TYPES[kind])]
    return [{"kind": kind, "key": r.name, "ra": float(r.ra), "dec": float(r.dec),
             "period": None if pd.isna(r.period) else float(r.period)}
            for r in sel.iloc[rng.permutation(len(sel))].itertuples()]


def dip_jobs(rng):
    """Quiet-star candidates rejected because their light curve had a dip (transits, eclipses)."""
    tics = set()
    if PROCESSED.exists():
        for line in PROCESSED.read_text().splitlines():
            r = json.loads(line)
            if r.get("kind") == "Noise" and "dip found" in str(r.get("reason", "")):
                tics.add(int(r["tic"]))
    tics = sorted(tics)
    return [{"kind": "DIP", "key": t, "tic": t} for t in rng.permutation(tics)]


# ====================================================================== 2. download + process

def search_2min(target, radius=None):
    import lightkurve as lk
    return retry(lambda: lk.search_lightcurve(target, radius=radius, author="SPOC", exptime=120))


def sector_of(row):
    m = re.search(r"Sector\s*(\d+)", str(row["mission"]))
    return int(m.group(1)) if m else 10 ** 6


def download_first(sr, sector=None):
    """Download + read one sector (default: the earliest available) of the first target in a SearchResult.

    Downloads with astroquery directly instead of SearchResult.download(): that method swaps
    sys.stdout for os.devnull while it runs, which is not thread-safe (parallel downloads end
    up writing progress bars into a closed file and die after their first 64 KB block).
    Files are written to a .part file and renamed only when complete, and a file that cannot
    be read is deleted and downloaded again.
    """
    import lightkurve as lk
    from astroquery.mast import Observations
    sectors = [sector_of(r) for r in sr.table]
    pick = sectors.index(int(sector)) if sector is not None and int(sector) in sectors else int(np.argsort(sectors)[0])
    row = sr.table[pick]
    path = CACHE / str(row["productFilename"])
    part = path.with_name(path.name + ".part")

    def fetch():
        part.unlink(missing_ok=True)
        status, msg, _ = Observations.download_file(str(row["dataURI"]), local_path=str(part),
                                                    cache=False, verbose=False)
        if status != "COMPLETE" or not part.exists() or part.stat().st_size < 100_000:
            raise IOError(f"download {status}: {msg}")
        part.replace(path)

    for attempt in range(3):
        if not path.exists():
            retry(fetch)
        try:
            return lk.read(str(path), quality_bitmask="default")
        except Exception:
            path.unlink(missing_ok=True)  # corrupt: delete and download again
            if attempt == 2:
                raise


def tic_of(sr_row_table):
    m = re.search(r"(\d+)", str(sr_row_table["target_name"]))
    return int(m.group(1)) if m else None


def cleanup(tic, keep):
    """Delete this star's FITS files. Best effort: on Windows a file can still be locked."""
    if keep or tic is None:
        return
    for p in sorted(CACHE.rglob(f"*{int(tic):016d}*"), key=lambda q: -len(q.parts)):
        try:
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        except OSError:
            pass


def in_vsx(ra, dec):
    """True if any VSX variable lies within 30 arcsec."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astroquery.vizier import Vizier
    hit = retry(lambda: Vizier(columns=["Name"], row_limit=5).query_region(
        SkyCoord(ra, dec, unit="deg"), radius=30 * u.arcsec, catalog="B/vsx/vsx"))
    return len(hit) > 0 and len(hit[0]) > 0


def handle(job, keep_fits):
    """Process one candidate star end-to-end. Never raises: failures become 'error' records."""
    kind, key = job["kind"], job["key"]
    rec = {"kind": kind, "key": key}
    tic = job.get("tic")
    try:
        if kind == "EB":
            sr = search_2min(f"TIC {tic}")
        elif kind in ("DSCT", "ROT", "GDOR", "RR", "ELL"):   # VSX stars: match coordinates to a TESS target
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            sr = search_2min(SkyCoord(job["ra"], job["dec"], unit="deg"), radius=10 * u.arcsec)
            if len(sr):
                tic = tic_of(sr.table[0])
                sr = sr[np.asarray([tic_of(r) == tic for r in sr.table])]
        elif kind == "DIP":
            sr = search_2min(f"TIC {int(tic)}")
        else:  # Noise
            if in_vsx(job["ra"], job["dec"]):
                return {**rec, "tic": tic, "status": "rejected", "reason": "known variable in VSX"}
            sr = search_2min(f"TIC {tic}")
        if len(sr) == 0:
            return {**rec, "tic": tic, "status": "rejected", "reason": "no SPOC 2-min light curve"}
        lc = download_first(sr)
        meta = {k: lc.meta.get(k) for k in ("SECTOR", "CROWDSAP", "FLFRCSAP", "TESSMAG", "TEFF", "RADIUS")}
        meta = {k.lower(): (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in meta.items()}
        res = process_star(kind, lc.time, lc.flux, job.get("period"), seed=tic)
        del lc
        cleanup(tic, keep_fits)
        return {**rec, "tic": tic, **meta, "catalogue_period": job.get("period"), "morph": job.get("morph"), **res}
    except Exception as e:
        return {**rec, "tic": tic, "status": "error", "reason": f"{type(e).__name__}: {e}"[:300]}


def already_done():
    done, ok = set(), {k: 0 for k in ALL_KINDS}
    if PROCESSED.exists():
        for line in PROCESSED.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "error":
                continue  # retry errors on the next run
            done.add((r["kind"], str(r["key"])))
            if r.get("status") == "ok":
                ok[r["kind"]] += 1
    return done, ok


def run_jobs(kind, jobs, need, workers, keep_fits, done, ok):
    """Process jobs of one class until `need` stars are accepted (or candidates run out)."""
    todo = [j for j in jobs if (kind, str(j["key"])) not in done]
    batch = max(workers * 2, 8)
    i = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while ok[kind] < need and i < len(todo):
            chunk = todo[i:i + batch]
            i += batch
            for rec in pool.map(lambda j: handle(j, keep_fits), chunk):
                with _lock:
                    with PROCESSED.open("a") as fh:
                        fh.write(json.dumps({k: v for k, v in rec.items()}, default=str) + "\n")
                    done.add((kind, str(rec["key"])))
                    if rec.get("status") == "ok":
                        ok[kind] += 1
                tag = "OK " if rec.get("status") == "ok" else rec.get("status", "?")[:3].upper()
                log(f"[{kind:5s} {ok[kind]:4d}/{need}] {tag} TIC {rec.get('tic')}  "
                    f"{rec.get('reason', '') if rec.get('status') != 'ok' else ''}")
    if ok[kind] < need:
        log(f"!! {kind}: only {ok[kind]} accepted, candidates exhausted")


def noise_candidates(dsct, eb_tics, n_wanted, rng, workers):
    """TESS 2-min targets around DSCT positions, minus EBs and the DSCT stars themselves.
    (Known VSX variables are removed later, in parallel, inside handle().)"""
    path = DATA / "candidates_noise.csv"
    if path.exists():
        df = pd.read_csv(path)
        if len(df) >= n_wanted:
            return df
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    say(f"Collecting about {n_wanted} quiet-star candidates near the DSCT fields ...")
    seen, rows = set(eb_tics), []

    def around(i):
        star = dsct.iloc[i]
        try:
            sr = search_2min(SkyCoord(star.ra, star.dec, unit="deg"), radius=0.5 * u.deg)
        except Exception:
            return []
        out = []
        for r in sr.table:
            tic = tic_of(r)
            if tic is None or r["distance"] < 30:  # skip the DSCT star itself
                continue
            out.append((tic, float(r["s_ra"]), float(r["s_dec"])))
        return out

    order = rng.permutation(len(dsct))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for k in range(0, len(order), workers):
            for found in pool.map(around, order[k:k + workers]):
                for tic, ra, dec in found:
                    if tic in seen:
                        continue
                    seen.add(tic)
                    rows.append({"tic": tic, "ra": ra, "dec": dec})
            say(f"  {len(rows)} candidates so far")
            if len(rows) >= n_wanted:
                break
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


# ====================================================================== 3. export

def _dedupe(by, order, seen):
    for k in order:
        uniq = []
        for r in by.get(k, []):
            if r["tic"] not in seen:
                seen.add(r["tic"])
                uniq.append(r)
        by[k] = uniq
    return by


def apply_quiet_refold(rows):
    """Optional: quiet-star records re-processed with the current quiet-star rules (random fold period,
    slow-variability check; mantaray/tess.py) in data/tess/quiet_refold.jsonl replace
    their original fold. Quiet stars that fail the new check, or were not re-processed, stay in the
    file as rows (so tess_vetting.csv stays aligned) but get exclude=True and are not used."""
    if not QUIET_REFOLD.exists():
        return rows
    ref = {}
    for line in QUIET_REFOLD.read_text().splitlines():
        r = json.loads(line)
        if r.get("status") != "error":
            ref[int(r["tic"])] = r
    out = []
    for r in rows:
        if r["kind"] == "Noise":
            q = ref.get(int(r["tic"]))
            if q is None:
                r = {**r, "exclude": True, "exclude_reason": "not re-processed"}
            elif q["status"] == "ok":
                r = {**r, **{k: q[k] for k in REFOLD_KEYS if k in q}, "exclude": False}
            else:
                r = {**r, "exclude": True, "exclude_reason": q.get("reason")}
        out.append(r)
    n_ex = sum(1 for r in out if r.get("exclude"))
    say(f"Quiet stars: random-period fold from {QUIET_REFOLD.name}; {n_ex} excluded "
        f"(failed the slow-variability check or not re-folded)")
    return out


def export(seed=0):
    recs = [json.loads(l) for l in PROCESSED.read_text().splitlines()]
    good = [r for r in recs if r.get("status") == "ok"]
    by = {k: [r for r in good if r["kind"] == k] for k in ALL_KINDS}
    # one row per star: a star can turn up under several kinds; training classes win, in this order
    seen = set()
    by = _dedupe(by, ["EB", "DSCT", "ROT", "Noise"] + list(UNSEEN_KINDS), seen)
    classes = [k for k in TESS_CLASSES if len(by[k])]
    n = min(len(by[k]) for k in classes) if classes else 0
    say("Accepted stars: " + ", ".join(f"{k} {len(by[k])}" for k in TESS_CLASSES)
        + f" -> {n} per class for {len(classes)} classes (balanced)")
    if n == 0:
        say("Nothing to export yet.")
        return
    rnd = random.Random(seed)
    rows = apply_quiet_refold([r for k in classes for r in rnd.sample(by[k], n)])
    flux = np.array([r["flux"] for r in rows], np.float32)
    labels = np.array([LABEL[r["kind"]] for r in rows])
    masks = np.zeros((len(rows), N_BINS), bool)
    for i, r in enumerate(rows):
        masks[i, r.get("mask", [])] = True
    cols = ["label"] + [f"f{i:03d}" for i in range(N_BINS)]
    pd.DataFrame(np.column_stack([labels, flux]), columns=cols).astype({"label": int}).to_csv(
        DATA / "tess_dataset.csv", index=False, float_format="%.7g")
    meta_keys = ["kind", "tic", "key", "sector", "period", "catalogue_period", "morph", "crowdsap", "flfrcsap",
                 "tessmag", "teff", "ls_freq", "ls_snr", "point_noise", "sigma_bin", "n_points",
                 "depth1_sigma", "depth2_sigma", "mask_fraction", "detached_like", "period_source", "bls_snr",
                 "slow_snr", "exclude", "exclude_reason"]
    pd.DataFrame([{k: r.get(k) for k in meta_keys} for r in rows]).to_csv(DATA / "tess_meta.csv", index=False)
    np.save(DATA / "tess_masks.npy", masks)
    say(f"Wrote {len(rows)} stars to {DATA}/tess_dataset.csv (+ tess_meta.csv, tess_masks.npy)")

    unseen = [r for k in UNSEEN_KINDS for r in by[k]]
    if unseen:
        uc = ["kind"] + [f"f{i:03d}" for i in range(N_BINS)]
        pd.DataFrame([[r["kind"]] + list(r["flux"]) for r in unseen], columns=uc).to_csv(
            DATA / "unseen_dataset.csv", index=False, float_format="%.7g")
        pd.DataFrame([{k: r.get(k) for k in meta_keys} for r in unseen]).to_csv(DATA / "unseen_meta.csv", index=False)
        say("Never-seen test stars: " + ", ".join(f"{k} {len(by[k])}" for k in UNSEEN_KINDS)
            + f" -> {DATA}/unseen_dataset.csv")
    reasons = pd.Series([f"{r['kind']}: {str(r.get('reason'))[:60]}" for r in recs if r.get("status") != "ok"])
    if len(reasons):
        say("Most common rejection reasons:")
        say(reasons.str.replace(r"[-\d.]+", "#", regex=True).value_counts().head(12).to_string())


# ====================================================================== main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=500, help="accepted stars wanted per class")
    ap.add_argument("--workers", type=int, default=6, help="parallel downloads")
    ap.add_argument("--keep-fits", action="store_true", help="keep downloaded FITS files (uses ~2 MB per star)")
    ap.add_argument("--export-only", action="store_true", help="skip downloading, just rebuild the CSV")
    ap.add_argument("--unseen-per-type", type=int, default=100,
                    help="stars per never-seen test type (gamma Dor, RR Lyrae, ellipsoidal, dip stars); 0 = skip")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    global _LOGFILE
    _LOGFILE = (DATA / "build_log.txt").open("a", encoding="utf-8")
    sys.stdout, sys.stderr = _MainThreadOnly(sys.stdout), _MainThreadOnly(sys.stderr)
    from astroquery import log as aq_log
    aq_log.setLevel("WARNING")
    socket.setdefaulttimeout(NETWORK_TIMEOUT)
    say("Progress is also written to data/tess/build_log.txt\n")
    CACHE.mkdir(exist_ok=True)
    if not args.export_only:
        rng = np.random.default_rng(42)
        eb, dsct = load_eb_catalogue(), load_dsct_catalogue()
        done, ok = already_done()
        say(f"Already accepted: {ok}")

        eb_jobs = [{"kind": "EB", "key": int(r.tic), "tic": int(r.tic), "period": float(r.period),
                    "morph": None if pd.isna(r.morph) else float(r.morph)}
                   for r in eb.iloc[rng.permutation(len(eb))].itertuples()]
        run_jobs("EB", eb_jobs, args.per_class, args.workers, args.keep_fits, done, ok)

        ds_jobs = [{"kind": "DSCT", "key": r.name, "ra": float(r.ra), "dec": float(r.dec)}
                   for r in dsct.iloc[rng.permutation(len(dsct))].itertuples()]
        run_jobs("DSCT", ds_jobs, args.per_class, args.workers, args.keep_fits, done, ok)

        # quiet stars are rarer than they sound: in the smoke test only ~1 in 3 candidates passed
        cand = noise_candidates(dsct, set(eb.tic.astype(int)), 4 * args.per_class, rng, args.workers)
        no_jobs = [{"kind": "Noise", "key": int(r.tic), "tic": int(r.tic), "ra": float(r.ra), "dec": float(r.dec)}
                   for r in cand.itertuples()]
        run_jobs("Noise", no_jobs, args.per_class, args.workers, args.keep_fits, done, ok)

        # 4th class: spotted rotating stars, from VSX
        vsx = load_vsx_bright()
        run_jobs("ROT", vsx_jobs(vsx, "ROT", rng), args.per_class, args.workers, args.keep_fits, done, ok)

        # never-seen test types (used only to test the classifier, never to train it)
        if args.unseen_per_type > 0:
            for kind in ("GDOR", "RR", "ELL"):
                run_jobs(kind, vsx_jobs(vsx, kind, rng), args.unseen_per_type, args.workers, args.keep_fits, done, ok)
            run_jobs("DIP", dip_jobs(rng), args.unseen_per_type, args.workers, args.keep_fits, done, ok)
    export()


if __name__ == "__main__":
    main()
