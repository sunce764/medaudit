"""Tests for the leakage audit, on synthetic embeddings with a known answer.

  A. clean random embeddings, group-clean split -> CLEAN.
  B. a test embedding copied from a train embedding (+tiny noise) -> the pair is
     found -> NEAR-DUPLICATES.
  C. one group placed in two splits -> GROUP LEAKAGE (and it takes precedence).
"""
import numpy as np

from medaudit.audits import leakage


def _split_by_group(n_groups, rows_per_group, seed=0):
    """Clean group-disjoint train/val/test labels + a group id per row."""
    rng = np.random.default_rng(seed)
    groups, splits = [], []
    for g in range(n_groups):
        s = "train" if g % 5 < 3 else ("val" if g % 5 == 3 else "test")
        for _ in range(rows_per_group):
            groups.append(g)
            splits.append(s)
    return np.array(groups), np.array(splits), rng


def test_clean():
    groups, splits, rng = _split_by_group(60, 3, seed=1)
    feats = rng.normal(0, 1, size=(len(groups), 64))     # ~orthogonal, no dups
    rep = leakage.leakage_report(feats, splits, groups, threshold=0.90)
    assert rep["n_group_leak"] == 0, rep["group_leak"]
    assert rep["n_near_dup"] == 0, rep["near_dup"][:3]
    assert rep["verdict"] == "CLEAN", rep
    print("  clean            OK")


def test_near_duplicate_found():
    groups, splits, rng = _split_by_group(60, 3, seed=2)
    feats = rng.normal(0, 1, size=(len(groups), 64))
    tr = np.where(splits == "train")[0]
    te = np.where(splits == "test")[0]
    # plant 3 near-duplicates: copy a train row into a test row with tiny noise
    planted = []
    for k in range(3):
        i, j = tr[k], te[k]
        feats[j] = feats[i] + rng.normal(0, 1e-3, size=64)
        planted.append((i, j))
    rep = leakage.leakage_report(feats, splits, groups, threshold=0.90)
    assert rep["verdict"] == "NEAR-DUPLICATES", rep["verdict"]
    assert rep["n_near_dup"] >= 3, rep["n_near_dup"]
    found = {tuple(sorted((i, j))) for i, j, _ in rep["near_dup"]}
    for i, j in planted:
        assert tuple(sorted((i, j))) in found, (i, j, "not caught")
    assert rep["near_dup"][0][2] > 0.99                  # cosine of a true dup
    print(f"  near_duplicate   OK  (found {rep['n_near_dup']}, "
          f"worst cosine {rep['near_dup'][0][2]:.4f})")


def test_group_leak_takes_precedence():
    groups, splits, rng = _split_by_group(60, 3, seed=3)
    feats = rng.normal(0, 1, size=(len(groups), 64))
    # force group 0 (train) to also appear in test
    idx = np.where(groups == 0)[0]
    splits[idx[0]] = "test"
    rep = leakage.leakage_report(feats, splits, groups, threshold=0.90)
    assert rep["n_group_leak"] >= 1, rep
    assert 0 in rep["group_leak"], rep["group_leak"]
    assert rep["verdict"] == "GROUP LEAKAGE", rep
    print("  group_leak       OK")


if __name__ == "__main__":
    print("running leakage tests:")
    test_clean()
    test_near_duplicate_found()
    test_group_leak_takes_precedence()
    print("ALL PASS")
