# medaudit

**A reliability audit toolkit for medical-image classifiers.**

Medical-imaging models routinely clear their headline metrics while quietly
relying on *acquisition shortcuts* (a scanner, a stain, an imaging mode), on
*data leakage* (near-duplicate frames split across train and test), or on
*miscalibration* that collapses under a prevalence shift between hospitals.
These failures are invisible to accuracy and AUROC on a naive split.

`medaudit` runs the checks a careful clinical-ML reviewer would demand — from
one config — and produces a single self-contained audit report:

- **Shortcut probe** — can an acquisition/metadata attribute (mode, scanner,
  site) be linearly decoded from the model's features? How much beyond the class?
- **Leakage audit** — near-duplicate detection (perceptual-hash + embedding)
  within a dataset and across train/external sets; group-aware split checks.
- **Calibration** — ECE / Brier / reliability, overall and stratified.
- **Prevalence decomposition** — how much of an internal↔external gap is a
  base-rate artifact (recoverable by prior correction) vs a genuine failure.
- **External validation** — a harness for cross-center, threshold-honest evaluation.

Metrics are **hand-written in pure numpy** (no sklearn) and **unit-tested against
independent brute-force references** — so every number in a report is auditable.

## Install

```bash
pip install -e .            # core (numpy only)
pip install -e ".[audit]"   # + torch/torchvision/pillow/matplotlib for model-based audits
```

## Quickstart

```bash
medaudit version
medaudit audit --config audit.yaml --out report.html
```

```python
from medaudit import metrics
auc = metrics.auroc(scores, labels)                       # tie-safe, no sklearn
point, lo, hi = metrics.cluster_bootstrap(patient_id, stat_fn)  # honest CIs
```

## Status

| Component | State |
|---|---|
| Metrics core (ECE, Brier, AUROC, reliability, cluster bootstrap) | **done + unit-tested** |
| Group-aware split · manifest abstraction | in progress |
| Audits: shortcut probe · leakage · calibration · prevalence · external | migrating from reference pipeline |
| HTML report generator · CLI config | planned |

See [DESIGN.md](DESIGN.md) for the architecture and roadmap.

## License

MIT.
