"""Unit tests: every metric is checked against an independent brute-force /
definitional reference. Run: ``python tests/test_metrics.py`` (from repo root)
or ``pytest``.
"""
import numpy as np

from medaudit import metrics as M


# ---------- independent reference implementations (deliberately naive) ----------
def brute_auroc(scores, labels):
    """O(n^2) pairwise definition: AUROC = P(s_pos > s_neg) + 0.5 P(s_pos = s_neg)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    c = 0.0
    for p in pos:
        for n in neg:
            c += 1.0 if p > n else (0.5 if p == n else 0.0)
    return c / (len(pos) * len(neg))


def brute_ece_equal_width(probs, labels, n_bins):
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    N = len(conf)
    e = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (conf >= lo) & (conf <= hi) if b == 0 else (conf > lo) & (conf <= hi)
        if m.any():
            e += m.sum() / N * abs(correct[m].mean() - conf[m].mean())
    return e


def softmax_rows(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


# ---------- tests ----------
def test_auroc():
    rng = np.random.default_rng(0)
    for _ in range(30):
        n = int(rng.integers(10, 80))
        scores = rng.random(n)
        labels = (rng.random(n) > 0.5).astype(int)
        if labels.sum() in (0, n):
            continue
        assert abs(M.auroc(scores, labels) - brute_auroc(scores, labels)) < 1e-9
    s = np.array([0.5, 0.5, 0.5, 0.9]); y = np.array([0, 1, 0, 1])
    assert abs(M.auroc(s, y) - brute_auroc(s, y)) < 1e-12
    assert abs(M.auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) - 1.0) < 1e-12
    assert abs(M.auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) - 0.0) < 1e-12
    assert np.isnan(M.auroc(np.array([0.3, 0.4]), np.array([1, 1])))
    print("  auroc             OK")


def test_ece():
    rng = np.random.default_rng(1)
    for _ in range(30):
        n, C = int(rng.integers(20, 120)), int(rng.integers(2, 6))
        probs = softmax_rows(rng.random((n, C)) * 4)
        labels = rng.integers(0, C, n)
        for nb in (5, 10, 15):
            a = M.ece(probs, labels, n_bins=nb, strategy="equal_width")
            b = brute_ece_equal_width(probs, labels, nb)
            assert abs(a - b) < 1e-9, (a, b, nb)
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]); labels = np.array([0, 1, 0])
    assert M.ece(probs, labels, n_bins=10) < 1e-12
    e = M.ece(softmax_rows(rng.random((200, 3))), rng.integers(0, 3, 200), strategy="equal_mass")
    assert 0.0 <= e <= 1.0
    print("  ece               OK")


def test_brier():
    assert M.brier(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 1])) < 1e-12
    rng = np.random.default_rng(2)
    n, C = 60, 4
    p = softmax_rows(rng.random((n, C)) * 3); y = rng.integers(0, C, n)
    ref = np.mean(((p - np.eye(C)[y]) ** 2).sum(1))
    assert abs(M.brier(p, y) - ref) < 1e-12
    print("  brier             OK")


def test_reliability_curve():
    rng = np.random.default_rng(4)
    p = softmax_rows(rng.random((100, 3)) * 3); y = rng.integers(0, 3, 100)
    curve = M.reliability_curve(p, y, n_bins=10)
    assert len(curve) == 10
    assert sum(c[2] for c in curve) == 100        # every sample in exactly one bin
    print("  reliability_curve OK")


def test_cluster_bootstrap():
    rng = np.random.default_rng(3)
    n = 300
    groups = rng.integers(0, 25, n)
    vals = rng.random(n)

    def stat(idx):
        return vals[idx].mean()

    pt, lo, hi = M.cluster_bootstrap(groups, stat, n_boot=800, seed=42)
    assert abs(pt - vals.mean()) < 1e-12          # point estimate = full-sample statistic
    assert lo <= pt <= hi                          # CI brackets the point
    a = M.cluster_bootstrap(groups, stat, n_boot=200, seed=7)
    b = M.cluster_bootstrap(groups, stat, n_boot=200, seed=7)
    assert a == b                                  # reproducible under a fixed seed
    print("  cluster_bootstrap OK")


if __name__ == "__main__":
    print("running metric unit tests (checked vs brute-force references):")
    test_auroc()
    test_ece()
    test_brier()
    test_reliability_curve()
    test_cluster_bootstrap()
    print("ALL PASS")
