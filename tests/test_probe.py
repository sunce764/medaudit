"""Tests for the shortcut probe, on synthetic features with a KNOWN ground truth.

Three regimes, each with a designed-in answer:
  A. attribute genuinely encoded (and independent of class) -> decodable overall
     AND within every class -> verdict SHORTCUT ENCODED.
  B. attribute is pure noise vs the features -> near chance -> NOT DECODABLE.
  C. features encode only the CLASS, and the attribute is merely class-collinear
     -> decodable overall, but NOT within a fixed class -> verdict AMBIGUOUS.
     This is the case the within-class probe exists to catch.
"""
import numpy as np

from medaudit.audits import probe
from medaudit.audits.probe import _group_kfold


def _make(n_groups, class_fn, attr_fn, feat_fn, rows_per_group=3, seed=0):
    """Build (features, attr, labels, groups). Class/attr are per-group;
    features are per-row = feat_fn(class, attr) + row noise."""
    rng = np.random.default_rng(seed)
    F, A, Y, G = [], [], [], []
    for g in range(n_groups):
        c = class_fn(rng)
        a = attr_fn(rng, c)
        for _ in range(rows_per_group):
            F.append(feat_fn(rng, c, a))
            A.append(a)
            Y.append(c)
            G.append(g)
    return np.array(F), np.array(A), np.array(Y), np.array(G)


# --------------------------------------------------------------------------- #
def test_group_kfold_no_leakage():
    groups = np.repeat(np.arange(50), 3)
    fold = _group_kfold(groups, n_splits=5, seed=42)
    # every group maps to exactly one fold
    for g in np.unique(groups):
        assert len(np.unique(fold[groups == g])) == 1
    assert set(np.unique(fold)) <= set(range(5))
    print("  group_kfold_no_leakage  OK")


def test_encoded_shortcut():
    # attr independent of class; one feature dim carries attr, another carries class
    def cls(rng):  return int(rng.random() < 0.5)
    def att(rng, c): return int(rng.random() < 0.5)          # independent of class
    def feat(rng, c, a):
        return np.array([4.0 * a, 4.0 * c]) + rng.normal(0, 1, size=2)
    F, A, Y, G = _make(220, cls, att, feat, seed=1)

    rep = probe.probe_report(F, A, Y, G, attr_name="mode", min_per_class=40)
    assert rep["overall"]["auroc"] > 0.85, rep["overall"]
    # decodable within EACH class too (attr varies within class, feature carries it)
    for cname, r in rep["within_class"].items():
        assert "skipped" not in r, (cname, r)
        assert r["auroc"] > 0.75, (cname, r)
    assert rep["verdict"] == "SHORTCUT ENCODED", rep["verdict"]
    print("  encoded_shortcut        OK  (overall AUROC "
          f"{rep['overall']['auroc']:.3f})")


def test_not_encoded():
    def cls(rng):  return int(rng.random() < 0.5)
    def att(rng, c): return int(rng.random() < 0.5)
    def feat(rng, c, a):
        return np.array([4.0 * c]) + rng.normal(0, 1, size=1)  # only class, no attr
    F, A, Y, G = _make(220, cls, att, feat, seed=2)

    rep = probe.probe_report(F, A, Y, G, attr_name="mode", min_per_class=40)
    # probe should not clear the 0.60 bar -> NOT DECODABLE
    assert rep["overall"]["ci"][0] < 0.60, rep["overall"]
    assert rep["verdict"] == "NOT DECODABLE", rep
    print("  not_encoded             OK  (overall AUROC "
          f"{rep['overall']['auroc']:.3f})")


def test_class_collinear_is_ambiguous():
    # attr correlated with class, but features encode ONLY the class.
    # overall probe reads the attr<-class collinearity; within-class it vanishes.
    def cls(rng):  return int(rng.random() < 0.5)
    def att(rng, c):
        p = 0.85 if c == 1 else 0.15
        return int(rng.random() < p)
    def feat(rng, c, a):
        return np.array([6.0 * c]) + rng.normal(0, 1, size=1)  # class only
    F, A, Y, G = _make(300, cls, att, feat, seed=3)

    rep = probe.probe_report(F, A, Y, G, attr_name="mode", min_per_class=40)
    assert rep["overall"]["auroc"] > 0.65, rep["overall"]      # decodable overall
    # within each fixed class the attribute is NOT decodable (features hold no attr info)
    for cname, r in rep["within_class"].items():
        if "skipped" in r:
            continue
        assert r["ci"][0] < 0.62, (cname, r)
    assert rep["verdict"] == "AMBIGUOUS", rep
    print("  class_collinear         OK  (overall "
          f"{rep['overall']['auroc']:.3f}, within-class near chance)")


# --------------------------------------------------------------------------- #
# robustness of the verdict machinery itself
# --------------------------------------------------------------------------- #
def test_l2_grid_and_partition_repeats_reported():
    """A grid of ridge penalties is selected by nested inner-CV, and the report
    exposes the spread across fold partitions so a lucky split cannot hide."""
    def cls(rng):  return int(rng.random() < 0.5)
    def att(rng, c): return int(rng.random() < 0.5)
    def feat(rng, c, a):
        return np.array([4.0 * a, 4.0 * c]) + rng.normal(0, 1, 2)
    F, A, Y, G = _make(200, cls, att, feat, seed=11)

    r = probe.linear_probe_auroc(F, A, G)              # default: L2_GRID, N_REPEATS
    assert r["n_repeats"] == probe.N_REPEATS, r["n_repeats"]
    lo, hi = r["partition_spread"]
    assert lo <= r["auroc"] <= hi, (lo, r["auroc"], hi)
    assert all(l in probe.L2_GRID for l in r["l2_chosen"]), r["l2_chosen"]
    # a single scalar l2 must still work (back-compatible)
    r1 = probe.linear_probe_auroc(F, A, G, l2=1.0)
    assert r1["auroc"] > 0.85, r1
    print(f"  l2_grid_and_repeats     OK  (chose l2={r['l2_chosen']}, "
          f"spread {lo:.3f}-{hi:.3f})")


def test_mixed_verdict_requires_all_classes():
    """Mode encoded in ONE class only must NOT be promoted to SHORTCUT ENCODED —
    with several classes probed, one hit is what chance produces."""
    def cls(rng):  return int(rng.random() < 0.5)
    def att(rng, c): return int(rng.random() < 0.5)
    def feat(rng, c, a):
        # dim0 carries mode ONLY for the malignant class (c == 1)
        return np.array([5.0 * a * c, 4.0 * c]) + rng.normal(0, 1, 2)
    F, A, Y, G = _make(220, cls, att, feat, seed=12)

    rep = probe.probe_report(F, A, Y, G, attr_name="mode",
                             class_names=["benign", "malignant"], min_per_class=40)
    assert rep["verdict"] == "MIXED", (rep["verdict"], rep["detail"])
    assert "not others" in rep["detail"]
    print("  mixed_not_promoted      OK  (1-of-2 classes -> MIXED, not ENCODED)")


if __name__ == "__main__":
    print("running probe tests:")
    test_group_kfold_no_leakage()
    test_encoded_shortcut()
    test_not_encoded()
    test_class_collinear_is_ambiguous()
    test_l2_grid_and_partition_repeats_reported()
    test_mixed_verdict_requires_all_classes()
    print("ALL PASS")
