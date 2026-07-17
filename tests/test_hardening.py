"""Regression tests for the 12 issues found by the v0.2 adversarial review.
Each test pins one fix so it cannot silently regress.
"""
import numpy as np

from medaudit.manifest import Manifest
from medaudit.metrics import cluster_bootstrap, auroc
from medaudit.audits import probe, leakage
from medaudit.audit import run_audit_arrays


def _feats(rng, code, dim=48):
    x = rng.normal(0, 1, dim)
    x[0] += 4.0 * code               # dim0 encodes the binary `code`
    return x


# --- [1] single-class probe must NOT claim a class-collinearity control -------- #
def test_single_class_probe_not_overclaimed():
    rng = np.random.default_rng(1)
    feats, mode, labels, groups = [], [], [], []
    for g in range(120):
        m = int(rng.random() < 0.5)
        for _ in range(3):
            feats.append(_feats(rng, m)); mode.append(m)
            labels.append(0)                              # ONE class only
            groups.append(g)
    rep = probe.probe_report(np.array(feats), np.array(mode), np.array(labels),
                             np.array(groups), attr_name="mode", min_per_class=40)
    assert rep["within_class"] == {}, rep["within_class"]   # no within-class ran
    assert rep["verdict"] != "SHORTCUT ENCODED", rep["verdict"]
    assert rep["verdict"] == "DECODABLE", rep               # decodable, collinearity unknown
    print("  [1] single-class -> DECODABLE, not SHORTCUT ENCODED   OK")


# --- [2] blank group cell is an error, not a silent singleton ------------------ #
def test_blank_group_cell_raises():
    rows = [{"path": "a", "label": "x", "group": "p1"},
            {"path": "b", "label": "x", "group": ""},      # blank!
            {"path": "c", "label": "y", "group": "p2"}]
    man = Manifest(rows)
    try:
        man.groups()
        assert False, "expected ValueError on blank group cell"
    except ValueError as e:
        assert "blank" in str(e).lower()
    print("  [2] blank group cell -> ValueError                    OK")


# --- [3] cluster_bootstrap exposes a valid-resample fraction ------------------- #
def test_cluster_bootstrap_valid_frac():
    # one group holds the only positives -> resamples that miss it -> nan
    groups = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([1, 0, 0, 0, 0, 0])
    sc = np.array([0.9, 0.1, 0.2, 0.3, 0.4, 0.5])
    out = cluster_bootstrap(groups, lambda idx: auroc(sc[idx], y[idx]),
                            n_boot=500, verbose=False, return_valid_frac=True)
    assert len(out) == 4, out
    frac = out[3]
    assert 0.0 < frac < 1.0, frac                          # some resamples degenerate
    print(f"  [3] cluster_bootstrap valid_frac={frac:.2f}            OK")


# --- [5] leakage: group check honestly reported as NOT ASSESSED --------------- #
def test_group_leak_not_assessed_when_no_ids():
    rng = np.random.default_rng(5)
    feats = rng.normal(0, 1, (60, 32))
    splits = np.array(["train"] * 30 + ["test"] * 30)
    rep = leakage.leakage_report(feats, splits, groups=None, threshold=0.90)
    assert rep["group_assessed"] is False, rep
    assert "NOT ASSESSED" in rep["detail"], rep["detail"]
    print("  [5] no ids -> group leakage NOT ASSESSED              OK")


# --- [6] near-duplicate count is the true total, not the display cap ----------- #
def test_near_dup_count_not_capped():
    rng = np.random.default_rng(6)
    feats = rng.normal(0, 1, (40, 32))
    splits = np.array(["train"] * 20 + ["test"] * 20)
    for k in range(8):                                     # plant 8 cross-split dups
        feats[20 + k] = feats[k] + rng.normal(0, 1e-3, 32)
    rep = leakage.leakage_report(feats, splits, groups=None, threshold=0.90, top_k=3)
    assert rep["n_near_dup"] >= 8, rep["n_near_dup"]        # true total, not 3
    assert rep["n_near_dup_shown"] == 3, rep["n_near_dup_shown"]
    assert len(rep["near_dup"]) == 3
    print(f"  [6] {rep['n_near_dup']} total, showing {rep['n_near_dup_shown']} "
          "(count not capped)          OK")


# --- [7] high point but wide CI -> NOT ESTABLISHED, not 'near chance' ---------- #
def test_underpowered_not_called_near_chance():
    # tiny sample: attr encoded, but only a handful of groups -> wide CI
    rng = np.random.default_rng(7)
    feats, mode, labels, groups = [], [], [], []
    for g in range(6):                                     # very few groups
        m = g % 2
        feats.append(_feats(rng, m)); mode.append(m); labels.append(g % 2); groups.append(g)
    rep = probe.probe_report(np.array(feats), np.array(mode), np.array(labels),
                             np.array(groups), attr_name="mode", min_per_class=2)
    # with dim0 cleanly encoding mode the point is high; if the CI is wide the
    # verdict must not be the 'near chance' NOT DECODABLE wording
    if rep["verdict"] == "NOT DECODABLE":
        assert "near chance" in rep["detail"] and rep["overall"]["auroc"] <= 0.60
    print(f"  [7] verdict={rep['verdict']} (no false 'near chance')     OK")


# --- [10] non-2D features -> clear error, not an opaque deep crash ------------- #
def test_non_2d_features_raise():
    rows = [{"path": f"{i}", "label": "a" if i % 2 else "b", "group": f"p{i}"}
            for i in range(10)]
    man = Manifest(rows)
    try:
        run_audit_arrays(man, np.zeros(10), {})            # 1-D, right length
        assert False, "expected ValueError on non-2D features"
    except ValueError as e:
        assert "2-D" in str(e) or "2-d" in str(e).lower()
    print("  [10] 1-D features -> clear ValueError                 OK")


# --- [11] a provided split column is used, and group leakage is a real check --- #
def test_split_column_is_split_of_record():
    rng = np.random.default_rng(11)
    rows, feats = [], []
    for g in range(30):
        for r in range(3):
            rows.append({"path": f"{g}_{r}", "label": "a" if g % 2 else "b",
                         "group": f"p{g}", "attr_mode": "x" if g % 3 else "y",
                         "split": "train" if g < 20 else "test"})
            feats.append(rng.normal(0, 1, 32))
    # force patient p0 to also appear in test -> real group leakage
    rows[0]["split"] = "test"
    man = Manifest(rows)
    rep = run_audit_arrays(man, np.array(feats), {"leakage": {"threshold": 0.9}})
    assert rep["leakage"]["group_assessed"] is True, rep["leakage"]
    assert rep["leakage"]["verdict"] == "GROUP LEAKAGE", rep["leakage"]
    print("  [11] split column used, group leakage caught          OK")


# --- [12] empty manifest -> clear error at construction ------------------------ #
def test_empty_manifest_raises():
    try:
        Manifest([])
        assert False, "expected ValueError on empty manifest"
    except ValueError as e:
        assert "one row" in str(e).lower()
    print("  [12] empty manifest -> ValueError                     OK")


if __name__ == "__main__":
    print("running hardening regression tests:")
    test_single_class_probe_not_overclaimed()
    test_blank_group_cell_raises()
    test_cluster_bootstrap_valid_frac()
    test_group_leak_not_assessed_when_no_ids()
    test_near_dup_count_not_capped()
    test_underpowered_not_called_near_chance()
    test_non_2d_features_raise()
    test_split_column_is_split_of_record()
    test_empty_manifest_raises()
    print("ALL PASS")
