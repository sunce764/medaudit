# From Pixels to Patients

**Learn to explain what a medical-image classifier's results show — and what they don't.**

A hands-on reliability tutorial for the **MICCAI Educational Challenge 2026**,
by Chao Sheng, with **MedAudit** as its companion Python toolkit.

**[Read the tutorial →](tutorial/from-pixels-to-patients.md)** ·
**[Run and interpret the examples →](tutorial/RUNNING.md)**

## Who this is for

Students and researchers entering medical imaging who know basic Python and
classification metrics. You need a laptop and Python 3.9+ to run the examples;
no medical images, model weights, GPU, or dataset access is needed.

You will learn to:

- compare an overall shortcut probe with within-class controls;
- check a recorded split for patient overlap and inspect near-duplicate flags;
- distinguish discrimination, calibration, and prevalence-dependent PPV;
- explain why five findings in the original audit were withdrawn.

The runnable examples are **synthetic**. The tutorial also discusses an earlier
cystoscopy audit; its historical aggregate results cannot be reproduced from this
repository and are not clinical validation. Start with tutorial §2 for the A/B
example featured in the promotional video, then §6 for the self-corrections.

## Quickstart

Run these commands in a terminal (macOS/Linux):

```bash
git clone https://github.com/sunce764/medaudit.git
cd medaudit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python tutorial/make_demo.py
medaudit audit --config tutorial/demo/audit.json --out tutorial/demo/report.txt
medaudit audit --config tutorial/demo/leaked.json --out tutorial/demo/leaked-report.txt
python tutorial/worked_examples.py
```

On Windows PowerShell, create the environment with `py -m venv .venv` and activate
it with `.venv\Scripts\Activate.ps1`; the remaining commands are the same.
The core installation needs only NumPy. Installation downloads dependencies;
the examples then run locally without downloading data. Figure generation is
optional and needs `python -m pip install -e '.[figures]'`.

### What you should see

| Example | Expected output | How to read it |
|---|---|---|
| A: class signal only | Overall AUROC ≈0.841; `AMBIGUOUS` | Overall decodability can arise from correlation with the diagnostic label. |
| B: same cohort, added mode signal | Overall AUROC ≈0.905; `SHORTCUT ENCODED` | Within-class evidence supports encoded mode information; it does **not** prove the diagnostic classifier used it. |
| C: full audit | Probe and leakage text report | The generated split cannot establish that your original model's training/test split was clean. |
| D: planted patient overlap | `GROUP LEAKAGE` | The detector must flag the deliberately contaminated split. |
| Calibration / prevalence | ECE, Brier, reliability bins; PPV ≈0.154 at 1% prevalence | Worked examples of existing metrics and arithmetic, separate from the automated audit. |

`make_demo.py` prints A–D and creates synthetic CSV/NumPy/config files under
`tutorial/demo/`. The two CLI commands repeat C and D, respectively, and save
separate text reports there. These generated files are ignored by Git.
**A leakage verdict is report content: the CLI still exits successfully.** It
is not an automated deployment gate. See the [reading guide](tutorial/RUNNING.md)
for warnings, exact conventions, and short exercises.

## What MedAudit automates (v0.2.0)

`medaudit audit` runs **two** checks on row-aligned frozen features:

| Check | Implemented behavior | Limit |
|---|---|---|
| Shortcut probe | Fits separate linear attribute probes overall and within each class; group-aware folds and group-cluster bootstrap intervals | Decodability is not evidence that the original classifier relies on that attribute. |
| Leakage audit | Embedding cosine-similarity flags across splits; exact group overlap when a split is supplied | Similarity needs inspection; an automatically generated split does not assess historical leakage. |

The original classifier stays frozen; the separate probes are fitted during the
audit. **Calibration and prevalence are worked examples, not additional CLI
checks.** `medaudit.metrics` supplies ECE, Brier, reliability curves, AUROC and
cluster bootstrap. Automatic calibration/prevalence/external audit modules,
feature extraction, and HTML report generation are not implemented.
[DESIGN.md](DESIGN.md) includes the broader roadmap.

For your own features, see [input format and configuration](tutorial/RUNNING.md#use-your-own-features).
Keep images, manifests, features, weights, and reports containing private details
local; the repository's ignore rules help prevent accidental inclusion.

## Verify the installation

```bash
python -m pip install -e '.[dev]'
python -m pytest tests -q
python -m pip check
```

The tests cover metrics, splits, manifests, probes, leakage and report assembly.
Passing synthetic tests does not establish performance on medical data.
See the [local verification record](docs/READER-VERIFICATION.md) for the tested
versions and scope; the package's supported dependency ranges are not a lockfile.

## Attribution, AI assistance, and license

Author: Chao Sheng. The
[AI assistance disclosure](tutorial/from-pixels-to-patients.md#ai-assistance)
describes how Claude Code and OpenAI Codex contributed to the original work
and this reader-facing revision.

To cite this educational resource, use: Chao Sheng. *From Pixels to Patients:
a hands-on reliability audit of a medical-image classifier*. MedAudit repository,
2026. Include the commit you used and an access date; no paper DOI is claimed.
The tutorial's [references](tutorial/from-pixels-to-patients.md#references) credit
the underlying methods.

Code and accompanying documentation are available under the [MIT license](LICENSE).
The tutorial charts are generated from synthetic examples/arithmetic by
[make_figures.py](tutorial/make_figures.py). No patient data is distributed.
