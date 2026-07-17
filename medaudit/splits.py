"""Group-aware, stratified train/val/test split — the first line of defence
against identity leakage (the same patient's images landing in both train and test).

A group (patient/subject/case) is assigned as a whole to exactly one fold, and
folds are stratified by a per-group label so class balance is preserved.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

__all__ = ["group_stratified_split", "assert_no_group_leakage"]


def group_stratified_split(groups, strata, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split rows by group, stratified by a per-row stratum label.

    Args:
      groups: (N,) group id per row (patient/subject/case).
      strata: (N,) per-row comparable stratum (e.g. 1 if the row is a lesion, else 0).
              A group's stratum is the max over its rows, so a group counts as
              positive if any of its rows is.
      val_ratio, test_ratio: group-level fractions.
      seed: RNG seed (deterministic).

    Returns:
      dict {"train": idx, "val": idx, "test": idx} of row-index arrays.
    """
    groups = np.asarray(groups)
    strata = np.asarray(strata)
    grp_rows = defaultdict(list)
    for i, g in enumerate(groups):
        grp_rows[g].append(i)
    grp_stratum = {g: max(strata[rows]) for g, rows in grp_rows.items()}

    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for s in sorted(set(grp_stratum.values())):
        gs = sorted(g for g in grp_rows if grp_stratum[g] == s)
        gs = list(rng.permutation(gs))
        n = len(gs)
        n_te = int(round(n * test_ratio))
        n_va = int(round(n * val_ratio))
        te += gs[:n_te]
        va += gs[n_te:n_te + n_va]
        tr += gs[n_te + n_va:]

    def collect(gset):
        s = set(gset)
        return np.array([i for i, g in enumerate(groups) if g in s], dtype=int)

    splits = {"train": collect(tr), "val": collect(va), "test": collect(te)}
    assert_no_group_leakage(groups, splits)
    return splits


def assert_no_group_leakage(groups, splits):
    """Raise if any group appears in more than one fold."""
    groups = np.asarray(groups)
    sets = {k: set(groups[idx]) for k, idx in splits.items()}
    keys = list(sets)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            inter = sets[keys[i]] & sets[keys[j]]
            if inter:
                raise AssertionError(
                    f"group leakage between {keys[i]!r} and {keys[j]!r}: "
                    f"{list(inter)[:3]}{'…' if len(inter) > 3 else ''}")
    return True
