"""Tests for group-aware stratified splitting: the leakage guarantee, stratification,
and determinism."""
import numpy as np

from medaudit.splits import group_stratified_split, assert_no_group_leakage


def _toy(n_groups=200, rows_per_group=3, pos_rate=0.3, seed=0):
    rng = np.random.default_rng(seed)
    groups, strata = [], []
    for g in range(n_groups):
        is_pos = int(rng.random() < pos_rate)
        for _ in range(rows_per_group):
            groups.append(g)
            strata.append(is_pos)
    return np.array(groups), np.array(strata)


def test_no_leakage():
    groups, strata = _toy()
    sp = group_stratified_split(groups, strata, seed=42)
    assert_no_group_leakage(groups, sp)                       # would raise on leak
    # every row assigned exactly once
    total = sum(len(v) for v in sp.values())
    assert total == len(groups)
    allidx = np.concatenate(list(sp.values()))
    assert len(np.unique(allidx)) == len(groups)
    print("  no_leakage        OK")


def test_stratified():
    groups, strata = _toy(pos_rate=0.3)
    sp = group_stratified_split(groups, strata, seed=42)
    # positive-row fraction should be close across folds
    fracs = [strata[idx].mean() for idx in sp.values()]
    assert max(fracs) - min(fracs) < 0.08, fracs
    print("  stratified        OK")


def test_deterministic():
    groups, strata = _toy()
    a = group_stratified_split(groups, strata, seed=42)
    b = group_stratified_split(groups, strata, seed=42)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    print("  deterministic     OK")


if __name__ == "__main__":
    print("running split tests:")
    test_no_leakage()
    test_stratified()
    test_deterministic()
    print("ALL PASS")
