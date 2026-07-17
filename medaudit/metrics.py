"""Hand-written evaluation metrics (pure numpy, no sklearn).

Covers the primitives a reliability audit needs:
  - Calibration : ``ece`` (equal-width / equal-mass), ``brier``, ``reliability_curve``
  - Discrimination : ``auroc`` (binary, mean-rank / Mann-Whitney, tie-safe)
  - Uncertainty : ``cluster_bootstrap`` (resample whole groups, e.g. patients)

Every function is checked against an independent brute-force / definitional
reference in ``tests/test_metrics.py`` before it is trusted.

Conventions: ``probs`` is ``(N, C)`` softmax output; ``labels`` is ``(N,)`` int
class indices; for binary metrics, ``scores`` is ``(N,)`` positive-class scores.
"""
from __future__ import annotations

import numpy as np

__all__ = ["ece", "brier", "reliability_curve", "auroc", "cluster_bootstrap"]


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _top_conf_correct(probs, labels):
    """Per-sample top-label confidence and whether it was correct."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    return conf, correct


def _bin_index(conf, edges):
    """Assign each confidence to exactly one bin ``[edges[i], edges[i+1]]``
    (right-closed) using ``digitize`` on the interior edges — no gaps, no overlaps."""
    n_bins = len(edges) - 1
    idx = np.digitize(conf, edges[1:-1], right=True)
    return np.clip(idx, 0, n_bins - 1)


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def ece(probs, labels, n_bins=15, strategy="equal_width"):
    """Expected Calibration Error (top-label): ``sum_b (n_b/N) * |acc_b - conf_b|``.

    strategy:
      ``equal_width`` — classic fixed-width bins.
      ``equal_mass``  — quantile bins (adaptive-ECE), robust to density skew.

    Note: ECE is a biased estimator; report it alongside Brier, never alone.
    """
    conf, correct = _top_conf_correct(probs, labels)
    N = len(conf)
    if strategy == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "equal_mass":
        edges = np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.unique(edges)                 # ties can collapse quantiles
        edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError(f"unknown strategy: {strategy!r}")
    idx = _bin_index(conf, edges)
    e = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            e += (m.sum() / N) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def brier(probs, labels, n_classes=None):
    """Multiclass Brier score ``mean_i sum_c (p_ic - y_ic)^2`` (y one-hot). Lower is better; 0 = perfect."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    C = n_classes or probs.shape[1]
    onehot = np.eye(C)[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def reliability_curve(probs, labels, n_bins=15):
    """Per-bin ``(mean_conf, mean_acc, count)`` for a reliability diagram; empty bins are ``(nan, nan, 0)``."""
    conf, correct = _top_conf_correct(probs, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = _bin_index(conf, edges)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
        else:
            out.append((float("nan"), float("nan"), 0))
    return out


# --------------------------------------------------------------------------- #
# discrimination
# --------------------------------------------------------------------------- #
def _rankdata_avg(a):
    """Average ranks (1-based, ties averaged), matching ``scipy.stats.rankdata`` default."""
    a = np.asarray(a, dtype=float)
    n = len(a)
    order = np.argsort(a, kind="mergesort")     # stable
    sorted_a = a[order]
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0   # mean rank of the tie group
        i = j + 1
    return ranks


def auroc(scores, labels):
    """Binary AUROC via mean rank (Mann-Whitney U). ``labels`` in {0, 1}, ``scores`` = positive-class score.

    ``(sum(rank_pos) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)``; ties count as 0.5.
    Returns ``nan`` if either class is absent.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rankdata_avg(scores)
    return float((r[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# --------------------------------------------------------------------------- #
# uncertainty: group-cluster bootstrap
# --------------------------------------------------------------------------- #
def cluster_bootstrap(groups, stat_fn, n_boot=2000, seed=42, ci=95, verbose=True,
                      return_valid_frac=False):
    """Group-cluster bootstrap CI: resample whole groups (e.g. patients), not rows.

    Medical images from one patient/subject are not independent, so a naive
    per-row bootstrap understates uncertainty. Each replicate resamples the
    unique groups with replacement and takes ALL rows of the chosen groups.

    Args:
      groups:  ``(N,)`` group id per row (patient/subject/case).
      stat_fn: index array (may repeat) -> scalar statistic; point estimate uses ``arange(N)``.
      n_boot:  number of resamples.
      seed:    RNG seed (reproducible).
      ci:      confidence level as a percentage (e.g. 95).
      verbose: warn if many resamples degenerate to nan (a power signal).
      return_valid_frac: also return the fraction of non-degenerate resamples,
                         so callers can flag an underpowered CI in their report.

    Returns:
      ``(point, lo, hi)``, or ``(point, lo, hi, valid_frac)`` if
      ``return_valid_frac`` — where ``valid_frac`` is len(non-nan) / n_boot.
    """
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    grp2idx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)

    point = float(stat_fn(np.arange(len(groups))))
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([grp2idx[g] for g in sampled])
        boots[b] = stat_fn(idx)

    # Degenerate resamples (a group missing a class -> auroc nan, etc.) must be
    # dropped: a single nan would poison np.percentile and turn the whole CI nan.
    valid = boots[~np.isnan(boots)]
    valid_frac = len(valid) / n_boot if n_boot else 0.0
    if verbose and len(valid) < n_boot * 0.98:
        frac = 1 - valid_frac
        print(f"[medaudit] cluster_bootstrap: {frac:.1%} of resamples were nan "
              f"(subgroup underpowered) -> CI over valid resamples only; treat as inconclusive")
    if len(valid) == 0:
        out = (point, float("nan"), float("nan"))
    else:
        lo = float(np.percentile(valid, (100 - ci) / 2))
        hi = float(np.percentile(valid, 100 - (100 - ci) / 2))
        out = (point, lo, hi)
    return out + (valid_frac,) if return_valid_frac else out
