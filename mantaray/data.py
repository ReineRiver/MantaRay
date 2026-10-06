"""Phase-folded light curves: synthetic generator with ground-truth masks + CSV loader.

Representation
--------------
Every light curve is PHASE-FOLDED on its own period and binned to N_BINS equal
phase bins (phase 0 -> 1). Folding removes the period from the input, so the
network has to classify on light-curve SHAPE alone; that is exactly the
property Grad-CAM is meant to inspect. Phase is circular, so the model uses
circular padding and every curve gets a random phase shift, which stops the
network from learning "a dip at bin 50" as a position shortcut.

Classes
-------
0  Noise : non-variable star; white noise + slow folded systematics + outliers
1  EB    : eclipsing binary; primary + secondary eclipse (flat-bottomed or
           V-shaped), optional ellipsoidal modulation
2  DSCT  : delta Scuti; dominant radial/non-radial mode folded on its own
           period (fundamental + harmonics); other modes fold into scatter

For EBs we also return a boolean mask of the in-eclipse bins: the physical
ground truth that the Grad-CAM maps are scored against.
"""
import numpy as np

N_BINS = 200
CLASS_NAMES = ["Noise", "EB", "DSCT"]            # synthetic classes
TESS_CLASSES = ["Noise", "EB", "DSCT", "ROT"]   # real-star classes (ROT = spotted rotating stars)
PHASE = (np.arange(N_BINS) + 0.5) / N_BINS  # bin centres


def _circ_dist(centre):
    """|phase - centre| on the circle, shape (n, N_BINS)."""
    d = np.abs(PHASE[None, :] - centre[:, None]) % 1.0
    return np.minimum(d, 1.0 - d)


def _eclipse(dphi, depth, half_width, ingress_frac):
    """Trapezoid eclipse. ingress_frac=1 -> V shape (grazing); small -> flat bottom (total)."""
    t = (half_width[:, None] - dphi) / (ingress_frac[:, None] * half_width[:, None])
    return depth[:, None] * np.clip(t, 0.0, 1.0)


def _outliers(rng, x, sigma, p=0.3):
    """Single-bin spikes (cosmic rays, momentum dumps) in ANY class, so spikes are never a class cue."""
    n = x.shape[0]
    hit = rng.random(n) < p
    for i in np.flatnonzero(hit):
        k = rng.integers(1, 4)
        idx = rng.integers(0, N_BINS, k)
        x[i, idx] += rng.choice([-1, 1], k) * rng.uniform(4, 8, k) * sigma[i]
    return x


def make_noise(rng, n):
    sigma = np.ones(n)
    x = rng.normal(0, 1, (n, N_BINS))
    # folded instrumental systematics: low-order Fourier terms, up to ~1 sigma
    for k in (1, 2, 3):
        amp = rng.uniform(0, 0.6, n) / k
        x += amp[:, None] * np.sin(2 * np.pi * k * PHASE[None, :] + rng.uniform(0, 2 * np.pi, n)[:, None])
    return _outliers(rng, x, sigma), np.zeros((n, N_BINS), bool)


def make_eb(rng, n, snr=(3, 60)):
    hw1 = rng.uniform(0.012, 0.08, n)                 # half-width of primary, in phase
    hw2 = hw1 * rng.uniform(0.8, 1.2, n)
    f1 = rng.uniform(0.3, 1.0, n)                     # ingress fraction (1 = V-shaped)
    f2 = np.clip(f1 * rng.uniform(0.8, 1.2, n), 0.2, 1.0)
    d1 = np.exp(rng.uniform(np.log(0.02), np.log(0.5), n))
    d2 = d1 * rng.uniform(0.0, 0.9, n)
    c1 = rng.uniform(0, 1, n)                         # primary at a random phase
    c2 = (c1 + 0.5 + np.clip(rng.normal(0, 0.03, n), -0.1, 0.1)) % 1.0  # eccentric offset
    dp1, dp2 = _circ_dist(c1), _circ_dist(c2)
    flux = -_eclipse(dp1, d1, hw1, f1) - _eclipse(dp2, d2, hw2, f2)
    # ellipsoidal variation: minima at conjunctions (twice per orbit)
    a_ell = d1 * rng.uniform(0, 0.15, n) * (rng.random(n) < 0.5)
    flux -= a_ell[:, None] * 0.5 * (1 + np.cos(4 * np.pi * (PHASE[None, :] - c1[:, None])))
    snr = np.exp(rng.uniform(np.log(snr[0]), np.log(snr[1]), n))
    sigma = d1 / snr
    x = flux + rng.normal(0, 1, (n, N_BINS)) * sigma[:, None]
    mask = (dp1 < hw1[:, None]) | ((dp2 < hw2[:, None]) & (d2[:, None] > 0.05 * d1[:, None]))
    return _outliers(rng, x, sigma), mask


def make_contact(rng, n, snr=(3, 60)):
    """Contact (W UMa-type) binaries: two stars touching, so the light changes continuously.

    Two nearly equal minima per orbit, rounded or flat-bottomed (total eclipse), with an
    optional difference between the two maxima (O'Connell effect). The "eclipse" mask is
    +/- 0.15 in phase around each minimum.
    """
    A = np.exp(rng.uniform(np.log(0.02), np.log(0.5), n))
    c1 = rng.uniform(0, 1, n)
    p = PHASE[None, :] - c1[:, None]
    flux = -A[:, None] * 0.5 * (1 + np.cos(4 * np.pi * p))                     # two minima per orbit
    flux -= (A * rng.uniform(0, 0.25, n))[:, None] * 0.5 * (1 + np.cos(2 * np.pi * p))  # unequal depths
    flux += (A * rng.uniform(-0.08, 0.08, n))[:, None] * np.sin(2 * np.pi * p)        # O'Connell effect
    flat = np.where(rng.random(n) < 0.5, rng.uniform(0.85, 1.0, n), 10.0)              # total eclipses
    flux = np.maximum(flux, -(A * flat)[:, None])
    snr_ = np.exp(rng.uniform(np.log(snr[0]), np.log(snr[1]), n))
    sigma = A / snr_
    x = flux + rng.normal(0, 1, (n, N_BINS)) * sigma[:, None]
    mask = (_circ_dist(c1) < 0.15) | (_circ_dist((c1 + 0.5) % 1.0) < 0.15)
    return _outliers(rng, x, sigma), mask


def make_dsct(rng, n, snr=(3, 60)):
    a1 = np.exp(rng.uniform(np.log(0.005), np.log(0.3), n))
    a2 = a1 * rng.uniform(0, 0.4, n)                  # harmonics -> HADS-like asymmetry
    a3 = a1 * rng.uniform(0, 0.15, n)
    ph = rng.uniform(0, 2 * np.pi, (3, n))
    p = 2 * np.pi * PHASE[None, :]
    flux = (a1[:, None] * np.sin(p + ph[0][:, None])
            + a2[:, None] * np.sin(2 * p + ph[1][:, None])
            + a3[:, None] * np.sin(3 * p + ph[2][:, None]))
    snr = np.exp(rng.uniform(np.log(snr[0]), np.log(snr[1]), n))
    sigma = np.sqrt((a1 / snr) ** 2 + (a1 * rng.uniform(0, 0.3, n)) ** 2)  # + other modes as scatter
    x = flux + rng.normal(0, 1, (n, N_BINS)) * sigma[:, None]
    return _outliers(rng, x, sigma), np.zeros((n, N_BINS), bool)


def normalise(x):
    """Per-star: subtract median, divide by standard deviation (flux dips stay negative)."""
    x = x - np.median(x, axis=1, keepdims=True)
    return x / (x.std(axis=1, keepdims=True) + 1e-8)


def make_dataset(n_per_class, seed=0, snr=(3, 60), contact_frac=0.0):
    """Returns X (n, N_BINS) float32, y (n,) int64, eclipse_mask (n, N_BINS) bool.

    snr = (min, max) signal-to-noise range for EB and DSCT (eclipse depth or pulsation
    amplitude divided by per-bin noise), drawn log-uniformly.
    contact_frac = share of the EB class made of contact (W UMa) binaries. The default 0
    reproduces the original detached-only pilot exactly.
    """
    rng = np.random.default_rng(seed)
    n_c = int(round(contact_frac * n_per_class))
    parts = [make_noise(rng, n_per_class), make_eb(rng, n_per_class - n_c, snr), make_dsct(rng, n_per_class, snr)]
    if n_c:
        parts.append(make_contact(rng, n_c, snr))
    X = np.concatenate([p[0] for p in parts])
    M = np.concatenate([p[1] for p in parts])
    y = np.r_[np.zeros(n_per_class), np.ones(n_per_class - n_c), np.full(n_per_class, 2), np.ones(n_c)].astype(np.int64)
    order = rng.permutation(len(y))
    return normalise(X[order]).astype(np.float32), y[order], M[order]


def add_shortcut(X, y, target_class, start=20, width=3, amp=2.5):
    """Inject a fake instrumental glitch at a FIXED phase into one class only.

    Simulates a non-physical cue that is correlated with the label (e.g. one class
    observed mostly on one camera). Used to show Grad-CAM can catch a model that
    cheats. Returns a copy.
    """
    X = X.copy()
    X[y == target_class, start:start + width] += amp
    return X


def load_csv(path):
    """Load the project's tabular format: column 0 = class label, columns 1.. = folded flux."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return normalise(data[:, 1:]).astype(np.float32), data[:, 0].astype(np.int64)
