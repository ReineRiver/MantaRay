"""Lightweight 1D-CNN for phase-folded light curves (about 8k parameters, trains in seconds on a CPU).

Design choices that matter for Grad-CAM:
* padding_mode="circular": phase 0.99 sits next to phase 0.01, so an eclipse
  that wraps around the fold is still one feature.
* Only two pooling steps: the last conv layer keeps 50 of the 200 bins
  (1 cell = 4 bins = 0.02 in phase), fine enough to resolve an eclipse.
* Dilated last layer: receptive field of about 50 bins (0.25 in phase) without more pooling.
* Global average pooling + one linear layer: with this head Grad-CAM on conv3
  is mathematically identical to the original CAM (Zhou et al. 2016).
* Optional extra inputs (n_extra > 0, e.g. the log-period that phase-folding throws away) go
  through their own tiny network (1 -> 16 -> n_classes) whose output is ADDED to the shape logits:
      logit_c = shape_head_c(pooled conv3 features) + period_term_c(log P)
  The period term acts like a learned per-class prior that depends only on the period, so it can be
  plotted on its own, and the convolutional layers (and Grad-CAM / Integrated Gradients on the light
  curve) are unchanged.
  Pass them as model(x, extra), or set model.context (one row per star) before calling code that
  only passes x (the explanation functions); model.context is tiled when the batch is a whole
  multiple of it (the Integrated Gradients path).
"""
from contextlib import contextmanager

import torch
import torch.nn as nn


def _block(c_in, c_out, k, dilation=1):
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, k, padding=dilation * (k // 2), dilation=dilation, padding_mode="circular"),
        nn.BatchNorm1d(c_out),
        nn.ReLU(),
    )


class LightCurveCNN(nn.Module):
    def __init__(self, n_classes=3, n_extra=0):
        super().__init__()
        self.n_extra = n_extra
        self.context = None
        self.conv1 = _block(1, 16, 7)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = _block(16, 32, 5)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = _block(32, 32, 5, dilation=2)  # Grad-CAM target layer
        self.head = nn.Linear(32, n_classes)
        if n_extra:
            self.extra_head = nn.Sequential(nn.Linear(n_extra, 16), nn.ReLU(), nn.Linear(16, n_classes))

    def _extra(self, n, extra):
        e = self.context if extra is None else extra
        if e is None:
            raise ValueError("this model needs extra inputs: call model(x, extra) or set model.context")
        e = torch.as_tensor(e, dtype=torch.float32).reshape(-1, self.n_extra)
        if len(e) != n:
            assert n % len(e) == 0, "batch is not a whole multiple of model.context"
            e = e.repeat(n // len(e), 1)
        return e

    def shape_features(self, x):
        """Pooled conv3 features (input of the shape head)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        return self.conv3(x).mean(dim=2)

    def features(self, x, extra=None):
        """Shape features (+ extra inputs) - the feature space used for 'unknown' detection."""
        f = self.shape_features(x)
        return torch.cat([f, self._extra(len(f), extra)], 1) if self.n_extra else f

    def logits_from_features(self, F):
        out = self.head(F[:, :32])
        return out + self.extra_head(F[:, 32:]) if self.n_extra else out

    def forward(self, x, extra=None):  # x: (B, L) or (B, 1, L)
        return self.logits_from_features(self.features(x, extra))


def train(model, X, y, Xv, yv, epochs=15, lr=3e-3, batch=128, seed=0, verbose=True,
          extra=None, extra_val=None, augment=None):
    """augment: optional function (curves, star indices) -> curves, applied to every training batch."""
    torch.manual_seed(seed)
    X, y = torch.as_tensor(X), torch.as_tensor(y)
    Xv, yv = torch.as_tensor(Xv), torch.as_tensor(yv)
    E = None if extra is None else torch.as_tensor(extra, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(y) + batch - 1) // batch))
    lossf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(y))
        for i in range(0, len(y), batch):
            idx = perm[i:i + batch]
            xb = X[idx] if augment is None else augment(X[idx], idx)
            xb = torch.roll(xb, shifts=int(torch.randint(0, X.shape[1], (1,))), dims=1)  # phase shift
            opt.zero_grad()
            loss = lossf(model(xb, None if E is None else E[idx]), y[idx])
            loss.backward()
            opt.step()
            sched.step()
        if verbose and (ep == epochs - 1 or ep % 5 == 0):
            print(f"  epoch {ep + 1:2d}  loss {loss.item():.3f}  val acc {accuracy(model, Xv, yv, extra_val):.3f}")
    return model


@torch.no_grad()
def logits(model, X, extra=None):
    model.eval()
    return model(torch.as_tensor(X), extra)


def predict(model, X, extra=None):
    return logits(model, X, extra).argmax(1).numpy()


def proba(model, X, extra=None):
    """Class probabilities, shape (n, n_classes)."""
    return torch.softmax(logits(model, X, extra), 1).numpy()


def accuracy(model, X, y, extra=None):
    return float((predict(model, X, extra) == torch.as_tensor(y).numpy()).mean())


@contextmanager
def extra_context(model, extra):
    """Temporarily give the model its extra inputs (one row per star) for code that only passes x."""
    old = model.context
    model.context = None if extra is None else torch.as_tensor(extra, dtype=torch.float32)
    try:
        yield model
    finally:
        model.context = old
