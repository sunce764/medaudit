"""Reproduce every number shown in `from-pixels-to-patients.md`.

    python tutorial/make_demo.py

Builds a synthetic cohort (no patient data — none is redistributable) whose
structure mirrors the real audit in miniature, then prints the three outputs the
tutorial walks through:

  A. an attribute that is only class-collinear      -> AMBIGUOUS
  B. an attribute genuinely encoded in the features -> SHORTCUT ENCODED
  C. the full `medaudit audit` report

The point of A vs B is that they share almost the same headline AUROC and get
opposite verdicts — the within-class probe is what separates them.

Writes the manifest/features/config to tutorial/demo/ so you can also run the CLI:

    medaudit audit --config tutorial/demo/audit.json
"""
import csv
import json
import os

import numpy as np

from medaudit.audits import probe
from medaudit.manifest import Manifest
from medaudit.audit import run_audit_arrays, render_report

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "demo")
CLASSES = ["benign", "malignant"]


def _cohort(n_groups, attr_fn, feat_fn, seed, rows_per_group=3):
    """A patient-structured cohort: class and attribute are per-patient,
    features are per-image (so grouping actually matters)."""
    rng = np.random.default_rng(seed)
    feats, attr, labels, groups = [], [], [], []
    for g in range(n_groups):
        c = int(rng.random() < 0.5)
        a = attr_fn(rng, c)
        for _ in range(rows_per_group):
            feats.append(feat_fn(rng, c, a))
            attr.append(a)
            labels.append(c)
            groups.append(g)
    return np.array(feats), np.array(attr), np.array(labels), np.array(groups)


def case_a_class_collinear():
    """Mode correlates with class, but the features encode ONLY the class.
    The overall probe looks alarming; the within-class probe exonerates it."""
    f, a, y, g = _cohort(
        300,
        attr_fn=lambda rng, c: int(rng.random() < (0.85 if c else 0.15)),
        feat_fn=lambda rng, c, a: np.array([6.0 * c]) + rng.normal(0, 1, 1),
        seed=3)
    return probe.probe_report(f, a, y, g, attr_name="mode",
                              class_names=CLASSES, min_per_class=40)


def case_b_encoded():
    """Mode is independent of class AND written into the features: a real,
    encoded shortcut that survives holding the class fixed."""
    f, a, y, g = _cohort(
        220,
        attr_fn=lambda rng, c: int(rng.random() < 0.5),
        feat_fn=lambda rng, c, a: np.array([4.0 * a, 4.0 * c]) + rng.normal(0, 1, 2),
        seed=1)
    return probe.probe_report(f, a, y, g, attr_name="mode",
                              class_names=CLASSES, min_per_class=40)


def case_c_full_audit():
    """The realistic case: mode is BOTH correlated with class (as in clinical
    practice, where blue light is chosen for suspicious tissue) AND encoded."""
    f, a, y, g = _cohort(
        160,
        attr_fn=lambda rng, c: int(rng.random() < (0.75 if c else 0.25)),
        feat_fn=lambda rng, c, a: (np.concatenate([[4.0 * a, 4.0 * c], np.zeros(62)])
                                   + rng.normal(0, 1, 64)),
        seed=7)
    rows = [{"path": f"img_{g[i]}_{i}.png", "label": CLASSES[y[i]],
             "group": f"patient{g[i]}",
             "attr_mode": "blue_light" if a[i] else "white_light"}
            for i in range(len(y))]

    os.makedirs(DEMO, exist_ok=True)
    with open(os.path.join(DEMO, "data.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.save(os.path.join(DEMO, "features.npy"), f)
    with open(os.path.join(DEMO, "audit.json"), "w", encoding="utf-8") as fh:
        json.dump({"manifest": "data.csv", "features": "features.npy",
                   "probe": {"attributes": ["mode"]},
                   "leakage": {"threshold": 0.90}}, fh, indent=2)

    return run_audit_arrays(Manifest(rows), f,
                            {"probe": {"attributes": ["mode"]},
                             "leakage": {"threshold": 0.90}})


if __name__ == "__main__":
    print("=" * 70)
    print("A. attribute is only CLASS-COLLINEAR  (expect: AMBIGUOUS)")
    print("=" * 70)
    print(probe.format_report(case_a_class_collinear()))

    print("\n" + "=" * 70)
    print("B. attribute is GENUINELY ENCODED  (expect: SHORTCUT ENCODED)")
    print("=" * 70)
    print(probe.format_report(case_b_encoded()))

    print("\n" + "=" * 70)
    print("C. the full audit report")
    print("=" * 70)
    print(render_report(case_c_full_audit()))
    print(f"\ndemo written to {DEMO}/ — now try:\n"
          f"    medaudit audit --config tutorial/demo/audit.json")
