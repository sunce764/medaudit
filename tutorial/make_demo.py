"""Reproduce every number shown in `from-pixels-to-patients.md`.

    pip install -e . && python tutorial/make_demo.py

Builds a synthetic cohort (no patient data — none of it is redistributable) whose
structure mirrors the real audit in miniature, then prints the outputs the
tutorial walks through.

The minimal pair (A vs B) is deliberately CONTROLLED — it differs in exactly one
variable, `beta`, which is how strongly the features encode imaging mode:

    A  beta = 0.0   features encode the CLASS only. Mode is merely correlated
                    with the class (85/15), so the overall probe still scores
                    high — off the collinearity alone.
    B  beta = 1.2   same 85/15 collinearity, same class signal, but the features
                    ALSO encode mode.

Everything else — the collinearity, the class signal strength, the noise, the
patient structure, the sample size — is identical. That matters: a tutorial that
tells you to control your variables has to control its own. (An earlier draft of
this demo changed the collinearity between A and B as well, which made the
comparison meaningless.)

Both cases produce a high, alarming-looking headline (~0.84 vs ~0.91). Only one
is a real shortcut. Separating them is the within-class probe's entire job.

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

# Shared across the minimal pair — these are the CONTROLLED variables.
P_MODE_GIVEN_CLASS = (0.15, 0.85)   # P(blue light | benign), P(blue light | malignant)
ALPHA = 6.0                         # how strongly the features encode the class
N_GROUPS = 300                      # patients
ROWS_PER_GROUP = 3                  # images per patient


def _cohort(beta, seed, n_groups=N_GROUPS):
    """A patient-structured cohort. `beta` — how strongly the features encode
    mode — is the ONLY thing that varies between case A and case B."""
    rng = np.random.default_rng(seed)
    feats, attr, labels, groups = [], [], [], []
    for g in range(n_groups):
        c = int(rng.random() < 0.5)                       # class, per patient
        a = int(rng.random() < P_MODE_GIVEN_CLASS[c])     # mode, correlated with class
        for _ in range(ROWS_PER_GROUP):
            feats.append(np.array([ALPHA * c, beta * a]) + rng.normal(0, 1, 2))
            attr.append(a)
            labels.append(c)
            groups.append(g)
    return np.array(feats), np.array(attr), np.array(labels), np.array(groups)


def case_a_class_collinear():
    """beta = 0: the features never encode mode. The high headline is entirely
    the class-collinearity showing through."""
    f, a, y, g = _cohort(beta=0.0, seed=3)
    return probe.probe_report(f, a, y, g, attr_name="mode",
                              class_names=CLASSES, min_per_class=40)


def case_b_encoded():
    """beta = 1.2: identical setup, but the features also encode mode — a real
    shortcut, and one that survives holding the class fixed."""
    f, a, y, g = _cohort(beta=1.2, seed=3)
    return probe.probe_report(f, a, y, g, attr_name="mode",
                              class_names=CLASSES, min_per_class=40)


def _write_demo(rows, feats, name, config_name):
    """Write one self-contained demo: manifest, its OWN features file, and a config.

    Each demo gets its own features filename on purpose. An earlier version wrote
    every demo to a shared features.npy, so the last one silently clobbered the
    others and left audit.json pointing at a feature matrix from a different
    cohort — a row-count mismatch waiting to happen for anyone who ran the CLI.
    """
    os.makedirs(DEMO, exist_ok=True)
    with open(os.path.join(DEMO, f"{name}.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.save(os.path.join(DEMO, f"{name}_features.npy"), feats)
    with open(os.path.join(DEMO, f"{config_name}.json"), "w", encoding="utf-8") as fh:
        json.dump({"manifest": f"{name}.csv", "features": f"{name}_features.npy",
                   "probe": {"attributes": ["mode"]},
                   "leakage": {"threshold": 0.90}}, fh, indent=2)


def case_c_full_audit():
    """The realistic case, at 64 dimensions: mode is both correlated with the
    class (as in clinical practice) and encoded in the features."""
    rng = np.random.default_rng(7)
    feats, rows = [], []
    for g in range(160):
        c = int(rng.random() < 0.5)
        a = int(rng.random() < P_MODE_GIVEN_CLASS[c])
        for r in range(ROWS_PER_GROUP):
            x = rng.normal(0, 1, 64)
            x[0] += 4.0 * a
            x[1] += 4.0 * c
            feats.append(x)
            rows.append({"path": f"img_{g}_{r}.png", "label": CLASSES[c],
                         "group": f"patient{g}",
                         "attr_mode": "blue_light" if a else "white_light"})
    feats = np.array(feats)
    _write_demo(rows, feats, "data", "audit")
    return run_audit_arrays(Manifest(rows), feats,
                            {"probe": {"attributes": ["mode"]},
                             "leakage": {"threshold": 0.90}})


def case_d_leaked_split():
    """A split of record WITH a planted leak, so you can watch the group-leakage
    check actually fire. A check you never see fail is a check you won't trust."""
    rng = np.random.default_rng(11)
    feats, rows = [], []
    for g in range(60):
        c = int(rng.random() < 0.5)
        for r in range(ROWS_PER_GROUP):
            feats.append(rng.normal(0, 1, 32))
            rows.append({"path": f"img_{g}_{r}.png", "label": CLASSES[c],
                         "group": f"patient{g}",
                         "attr_mode": "blue_light" if c else "white_light",
                         "split": "train" if g < 45 else "test"})
    rows[0]["split"] = "test"          # patient0 now spans train AND test
    feats = np.array(feats)
    _write_demo(rows, feats, "leaked", "leaked")
    return run_audit_arrays(Manifest(rows), feats,
                            {"probe": {"attributes": ["mode"]},
                             "leakage": {"threshold": 0.90}})


if __name__ == "__main__":
    print("=" * 72)
    print("A. features encode the CLASS only  (beta=0.0)   expect: AMBIGUOUS")
    print("=" * 72)
    print(probe.format_report(case_a_class_collinear()))

    print("\n" + "=" * 72)
    print("B. same cohort, features ALSO encode MODE  (beta=1.2)")
    print("   the ONLY change from A. expect: SHORTCUT ENCODED")
    print("=" * 72)
    print(probe.format_report(case_b_encoded()))

    print("\n" + "=" * 72)
    print("C. the full audit report (64-dim features, generated split)")
    print("=" * 72)
    print(render_report(case_c_full_audit()))

    print("\n" + "=" * 72)
    print("D. a split of record with a planted leak — watch the check fire")
    print("=" * 72)
    print(render_report(case_d_leaked_split()))

    print(f"\ndemo written to {DEMO}/ — now try the CLI:\n"
          f"    medaudit audit --config tutorial/demo/audit.json\n"
          f"    medaudit audit --config tutorial/demo/leaked.json")
