# medaudit

**A reliability audit toolkit for medical-image classifiers.**

Medical-imaging models routinely clear their headline metrics while quietly
relying on an *acquisition shortcut* (a scanner, a stain, an imaging mode), on
*data leakage* (near-duplicate frames split across train and test), or on
*miscalibration* that collapses under a prevalence shift between hospitals.
None of these are visible in accuracy or AUROC on a naive split.

`medaudit` audits a **frozen** model — you supply features, it retrains nothing —
and answers the questions a careful clinical-ML reviewer asks. It is deliberately
small, hand-written, and non-overclaiming: every verdict also states what it does
*not* establish.

## What actually runs today (v0.2)

`medaudit audit --config audit.json` runs **two** audits and prints one report:

- **Shortcut probe** — is an acquisition/metadata attribute (mode, scanner, site)
  linearly decodable from the model's frozen features, **overall** and **within
  each fixed class**? The within-class control is the point: it separates
  "the features encode this attribute" from "this attribute merely correlates
  with the label".
- **Leakage audit** — cross-split near-duplicates by embedding cosine similarity,
  plus an exact group-leakage check when you supply a split of record.

Both report group-cluster bootstrap confidence intervals (resampling whole
patients, not rows) and a plain-language verdict.

**Calibration and prevalence are not yet wired into the report.** Their metric
primitives are implemented and tested (`medaudit.metrics`: ECE, Brier, reliability
curve, AUROC, cluster bootstrap) — compose them yourself for now. Roadmap in
[DESIGN.md](DESIGN.md).

Metrics are **hand-written in pure numpy** (no sklearn) and checked against
independent brute-force references, so every number in a report is auditable.

## Install

```bash
git clone https://github.com/sunce764/medaudit && cd medaudit
pip install -e .            # numpy only — no torch, no sklearn
```

## Quickstart

```bash
python tutorial/make_demo.py                        # builds a synthetic cohort, prints the report
medaudit audit --config tutorial/demo/audit.json    # the same audit via the CLI
```

Then point it at your own model:

```python
from medaudit.audit import run_audit
report = run_audit("audit.json")     # your manifest CSV + the features.npy you extracted
```

```python
from medaudit import metrics
auc = metrics.auroc(scores, labels)                              # tie-safe, no sklearn
point, lo, hi = metrics.cluster_bootstrap(patient_id, stat_fn)   # patient-clustered CI
```

You supply `features.npy` (your model's penultimate-layer activations, row-aligned
to the manifest). Extracting them is out of scope by design — it keeps the audit
dependency-free and runnable on a laptop.

## Status

| Component | State |
|---|---|
| `metrics` — ECE, Brier, reliability, AUROC, cluster bootstrap | **done + unit-tested** |
| `splits` — group-aware stratified split, leakage assertions | **done + unit-tested** |
| `manifest` — `(path,label,group,attr_*)` CSV abstraction | **done + unit-tested** |
| `audits.probe` — shortcut probe (overall + within-class) | **done + unit-tested** |
| `audits.leakage` — embedding near-duplicates + group leakage | **done + unit-tested** |
| `audit` + `cli` — config-driven report | **done + unit-tested** |
| `audits.calibration` · `audits.prevalence` · `audits.external` | **not implemented** (primitives live in `metrics`) |
| `models` (feature extraction) · HTML report | **not implemented** |

Run the tests: `for t in tests/test_*.py; do PYTHONPATH=. python "$t"; done`

## Tutorial

[tutorial/from-pixels-to-patients.md](tutorial/from-pixels-to-patients.md) — a
hands-on reliability audit walkthrough, including the war stories from a real
audit and the findings that did not survive scrutiny.

## License

MIT — see [LICENSE](LICENSE).
