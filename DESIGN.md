# medaudit — architecture & roadmap

## Design goals

1. **Generic, not dataset-specific.** Works on any image classifier + a manifest
   of `(path, label, group, attr_*)`. No hard-coded task assumptions.
2. **Auditable numbers.** Every metric is hand-written and unit-tested against an
   independent reference; no silent library behaviour.
3. **One command, one report.** A reviewer runs `medaudit audit --config …` and
   gets a readable report that flags shortcut / leakage / calibration / prevalence
   risks with confidence intervals.
4. **Runs on a laptop.** CPU-friendly; a single GPU (MPS/CUDA/Kaggle) only for
   the model forward passes.

## Pipeline

```
 manifest.csv ─┐
               ├─▶ split (group-aware) ─▶ model features/logits ─┐
 model  ───────┘                                                 ├─▶ audits ─▶ report.html
 config.yaml ──────────────────────────────────────────────────┘
                                   │
        ┌──────────────┬───────────┴────────────┬──────────────┬─────────────┐
   shortcut probe    leakage              calibration     prevalence     external
   (attr decodable?) (near-dup + split)   (ECE/Brier)     decomposition   (cross-center)
```

## Module map

| Module | Purpose | State |
|---|---|---|
| `medaudit.metrics` | ECE, Brier, reliability, AUROC, cluster bootstrap | **done + tested** |
| `medaudit.manifest` | load `(path,label,group,attr_*)` CSV; validate | **done + tested** |
| `medaudit.splits` | group-aware stratified split; leakage assertions | **done + tested** |
| `medaudit.models` | wrap a user model / ImageNet feature extractor | planned |
| `medaudit.audits.probe` | linear probe: attribute decodability from features (+ within-class) | **done + tested** |
| `medaudit.audits.leakage` | embedding near-duplicate scan + group-leakage check | **done + tested** |
| `medaudit.audits.calibration` | overall + stratified calibration report | planned |
| `medaudit.audits.prevalence` | Saerens-EM prior correction; base-rate decomposition | planned |
| `medaudit.audits.external` | cross-center, threshold-honest evaluation harness | planned |
| `medaudit.report` | assemble a self-contained HTML report | planned |
| `medaudit.cli` | `medaudit audit / version` | skeleton done |

## Provenance (reference pipeline → product)

Each module is a generalisation of a step that was first built, and validated
end-to-end, in a research pipeline auditing an endoscopic-image classifier. The
research code was single-purpose; the product makes each step dataset-agnostic and
config-driven.

| Reference step | Product module |
|---|---|
| hand-written metrics + brute-force unit tests | `medaudit.metrics` + `tests/test_metrics.py` |
| patient-grouped stratified split | `medaudit.splits` + `medaudit.manifest` |
| linear + within-class attribute probe | `medaudit.audits.probe` |
| near-duplicate scan (hash, then embedding) | `medaudit.audits.leakage` |
| calibration audit (ECE/Brier + cluster CI) | `medaudit.audits.calibration` |
| prevalence / base-rate decomposition | `medaudit.audits.prevalence` |
| cross-centre external evaluation harness | `medaudit.audits.external` |
| reliability-diagram figures | `medaudit.report` (figures) |
| backbone / feature extraction | `medaudit.models` |

## Roadmap

- **v0.1** — metrics core + package/CLI skeleton + tests *(done)*
- **v0.2** — manifest + group-aware split + shortcut probe + leakage audit (the two
  checks with the highest catch-rate) *(modules done + tested, 15 unit tests green;
  remaining: wire them behind `medaudit audit --config`)*
- **v0.3** — calibration + prevalence decomposition + external harness
- **v0.4** — HTML report generator; end-to-end `medaudit audit` on a demo dataset
- **v1.0** — documented, packaged, example gallery; register + open-source
