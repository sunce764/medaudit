"""Integration test for the audit orchestrator: manifest + features + config in,
one report out. Synthetic features with a designed-in answer (mode encoded,
split clean) so both audits have a known verdict, and a file round-trip that
exercises run_audit() + the CLI path.
"""
import csv
import json
import os
import tempfile

import numpy as np

from medaudit.manifest import Manifest, load_manifest
from medaudit.audit import run_audit_arrays, run_audit, render_report


def _synth(n_groups=140, rows_per_group=3, dim=64, seed=0):
    """Rows + row-aligned features. dim0 encodes mode, dim1 encodes class,
    the rest is noise (so distinct rows are not near-duplicates)."""
    rng = np.random.default_rng(seed)
    rows, feats = [], []
    for g in range(n_groups):
        c = int(rng.random() < 0.5)
        m = int(rng.random() < 0.5)                 # mode, independent of class
        for r in range(rows_per_group):
            x = rng.normal(0, 1, size=dim)
            x[0] += 4.0 * m
            x[1] += 4.0 * c
            feats.append(x)
            rows.append({"path": f"img_{g}_{r}.png",
                         "label": "malignant" if c else "benign",
                         "group": f"case{g}",
                         "attr_mode": "blue" if m else "white"})
    return rows, np.array(feats)


def test_run_audit_arrays():
    rows, feats = _synth(seed=1)
    man = Manifest(rows)
    rep = run_audit_arrays(man, feats,
                           {"probe": {"attributes": ["mode"], "min_per_class": 40},
                            "leakage": {"threshold": 0.90}})

    assert rep["n_rows"] == len(rows)
    assert sum(rep["split_sizes"].values()) == len(rows)
    # mode is encoded -> shortcut probe should catch it
    assert rep["probe"]["mode"]["verdict"] == "SHORTCUT ENCODED", rep["probe"]["mode"]
    # clean group split + varied features -> no leakage
    assert rep["leakage"]["verdict"] == "CLEAN", rep["leakage"]
    # report renders without error
    text = render_report(rep)
    assert "SHORTCUT PROBE" in text and "LEAKAGE" in text
    print("  run_audit_arrays  OK  (mode -> SHORTCUT ENCODED, split CLEAN)")


def test_default_attributes_probes_all_attr_cols():
    rows, feats = _synth(seed=2)
    man = Manifest(rows)
    rep = run_audit_arrays(man, feats, {})        # no probe.attributes -> use all attr_*
    assert "mode" in rep["probe"], rep["probe"].keys()
    print("  default_attributes OK")


def test_file_roundtrip():
    rows, feats = _synth(seed=3)
    d = tempfile.mkdtemp(prefix="medaudit_audit_")
    csv_path = os.path.join(d, "data.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.save(os.path.join(d, "features.npy"), feats)
    cfg = {"manifest": "data.csv", "features": "features.npy",
           "probe": {"attributes": ["mode"]}, "leakage": {"threshold": 0.90}}
    cfg_path = os.path.join(d, "audit.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)

    # sanity: manifest loads and summarises
    man = load_manifest(csv_path)
    assert len(man) == len(rows)

    rep = run_audit(cfg_path)
    assert rep["probe"]["mode"]["verdict"] == "SHORTCUT ENCODED", rep["probe"]["mode"]
    assert rep["leakage"]["verdict"] == "CLEAN", rep["leakage"]
    print("  file_roundtrip    OK  (loaded CSV+npy from disk, ran end-to-end)")


if __name__ == "__main__":
    print("running audit integration tests:")
    test_run_audit_arrays()
    test_default_attributes_probes_all_attr_cols()
    test_file_roundtrip()
    print("ALL PASS")
