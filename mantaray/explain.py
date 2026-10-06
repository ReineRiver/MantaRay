"""1D Grad-CAM, Integrated Gradients, and the tests that check whether the maps can be trusted.

All functions are batched and vectorised (no per-star Python loops).
Saliency maps are returned as numpy arrays of shape (n_stars, N_BINS),
non-negative, each normalised so its maximum is 1.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F


def _norm(s):
    s = np.clip(s, 0, None)
    return s / (s.max(axis=1, keepdims=True) + 1e-12)


def _circular_upsample(cam, L):
    """Linear interpolation from L' cells to L bins that wraps around phase 1 -> 0."""
    Lp = cam.shape[-1]
    pos = (torch.arange(L, dtype=cam.dtype) + 0.5) * Lp / L - 0.5
    i0 = torch.floor(pos).long()
    w = pos - i0
    return cam[..., i0 % Lp] * (1 - w) + cam[..., (i0 + 1) % Lp] * w


class GradCAM1D:
    """Grad-CAM for 1D signals.

    weights_k = mean_t dScore_c / dA_k(t)          (one weight per channel)
    cam(t)    = ReLU( sum_k weights_k * A_k(t) )    (at the layer's resolution)
    then upsampled (circularly) to the input length.
    """

    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.acts = out
        if out.requires_grad:  # skip during torch.no_grad() inference
            out.register_hook(lambda g: setattr(self, "grads", g))

    def remove(self):
        self.handle.remove()

    def __call__(self, X, target=None, batch=1024):
        self.model.eval()
        out = []
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch])
            logits = self.model(xb)
            t = logits.argmax(1) if target is None else torch.as_tensor(target[i:i + batch])
            self.model.zero_grad()
            logits.gather(1, t[:, None]).sum().backward()
            w = self.grads.mean(dim=2, keepdim=True)
            cam = F.relu((w * self.acts).sum(dim=1))
            out.append(_circular_upsample(cam, xb.shape[-1]).detach().numpy())
        return _norm(np.concatenate(out))


def integrated_gradients(model, X, target, steps=32, batch=512):
    """Integrated Gradients at full input resolution. Baseline = flat light curve (a constant star)."""
    model.eval()
    out = []
    alphas = torch.linspace(1.0 / steps, 1.0, steps)
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i:i + batch])
        tb = torch.as_tensor(target[i:i + batch])
        path = (alphas[:, None, None] * xb[None]).reshape(-1, xb.shape[-1]).requires_grad_(True)
        logits = model(path)
        logits.gather(1, tb.repeat(steps)[:, None]).sum().backward()
        g = path.grad.reshape(steps, *xb.shape).mean(0)
        out.append((xb * g).detach().numpy())
    return _norm(np.concatenate(out))


# ---------------------------------------------------------------- metrics

def mask_mass(sal, mask):
    """Fraction of the saliency that falls inside the physical mask (e.g. the eclipses)."""
    return (sal * mask).sum(1) / (sal.sum(1) + 1e-12)


def dilate(mask, k):
    """Widen a circular boolean mask by k bins on each side."""
    out = mask.copy()
    for s in range(1, k + 1):
        out |= np.roll(mask, s, axis=1) | np.roll(mask, -s, axis=1)
    return out


def pointing_game(sal, mask):
    """Share of stars whose single most-attributed bin falls inside the mask."""
    return np.take_along_axis(mask, sal.argmax(1)[:, None], axis=1)[:, 0]


def extrema_share(sal, X, frac=0.6, k=9):
    """Share of attribution on bins where the smoothed |flux| exceeds frac x its maximum (light max/min)."""
    kern = np.ones(k) / k
    sm = np.stack([np.convolve(np.concatenate([x[-k:], x, x[:k]]), kern, "same")[k:-k] for x in X])
    ext = np.abs(sm) > frac * np.abs(sm).max(1, keepdims=True)
    return mask_mass(sal, ext), ext.mean(1)


def spread(sal):
    """Normalised entropy of the map: 1 = spread evenly over phase, 0 = one single bin."""
    p = sal / (sal.sum(1, keepdims=True) + 1e-12)
    return -(p * np.log(p + 1e-12)).sum(1) / np.log(sal.shape[1])


@torch.no_grad()
def deletion_curve(model, X, y, sal, fractions=(0, 0.05, 0.1, 0.2, 0.3, 0.5), seed=0):
    """Flatten (set to 0 = median flux) the top-k most salient bins and record P(true class).

    A faithful map makes the probability fall much faster than flattening random bins.
    Returns dict with mean probability per fraction for 'salient' and 'random' order.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    n, L = X.shape
    order_sal = np.argsort(-sal, axis=1)
    order_rnd = np.argsort(rng.random((n, L)), axis=1)
    res = {"fractions": list(fractions), "salient": [], "random": []}
    for name, order in (("salient", order_sal), ("random", order_rnd)):
        for f in fractions:
            Xd = X.copy()
            k = int(round(f * L))
            if k:
                np.put_along_axis(Xd, order[:, :k], 0.0, axis=1)
            p = F.softmax(model(torch.as_tensor(Xd)), 1)[torch.arange(n), torch.as_tensor(y)]
            res[name].append(float(p.mean()))
    return res


def _rank(a):
    return np.argsort(np.argsort(a, axis=1), axis=1).astype(float)


def spearman(a, b):
    """Spearman rank correlation between two 1D arrays."""
    ra, rb = _rank(np.asarray(a)[None])[0], _rank(np.asarray(b)[None])[0]
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum() + 1e-12))


def sanity_check(model, layer_name, X, target, seed=0):
    """Model-randomisation test (Adebayo et al. 2018).

    Re-initialise every weight and recompute Grad-CAM. If the maps barely change,
    they describe the input, not what the network learned, and cannot be trusted.
    Returns mean Spearman rank correlation between trained and random-model maps.
    """
    torch.manual_seed(seed)
    rand = copy.deepcopy(model)
    for m in rand.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    ga, gb = GradCAM1D(model, getattr(model, layer_name)), GradCAM1D(rand, getattr(rand, layer_name))
    a, b = ga(X, target), gb(X, target)
    ga.remove(), gb.remove()
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    rho = (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1) + 1e-12)
    return float(np.nanmean(rho))


def explain_chunked(fn, model, X, E, *args, chunk=512):
    """Run an explanation function (e.g. integrated_gradients, evaluate.explain_all) on a model with extra
    inputs E (one row per star): processed in chunks of at most `chunk` stars with model.context = E[chunk],
    so the batches inside the explanation functions always match. E=None: plain call."""
    if E is None or getattr(model, "n_extra", 0) == 0:
        return fn(model, X, *args)
    parts = []
    old = model.context
    try:
        for i in range(0, len(X), chunk):
            model.context = torch.as_tensor(E[i:i + chunk], dtype=torch.float32)
            parts.append(fn(model, X[i:i + chunk], *[a[i:i + chunk] for a in args]))
    finally:
        model.context = old
    if isinstance(parts[0], dict):
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return np.concatenate(parts)
