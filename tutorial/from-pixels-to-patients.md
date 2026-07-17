# From Pixels to Patients: a hands-on reliability audit of a medical-image classifier

*The checks a clinical reviewer runs before trusting your AUROC.*

**MICCAI Educational Challenge 2026 submission.** Author: Chao Sheng
(Biomedical Engineering, Guangdong Medical University). Accompanying code:
the `medaudit` toolkit (open-source, MIT).

> **Audience & goal.** For Masters/PhD readers who can train a classifier but
> have not yet had one fail *silently*. By the end you will be able to run four
> reliability audits — shortcut, leakage, calibration, prevalence — on your own
> model, from one command, and read the results the way a reviewer would. Every
> code block runs on synthetic data shipped with the toolkit; the war stories
> come from a real audit of a classifier trained on a 2025 public cystoscopy
> dataset.
>
> **About those war-story numbers.** They are **single-seed** results from one
> audit. They are here to show you *what the audit does and how to read it* — not
> as settled claims about that dataset; a single seed cannot separate a real
> effect from checkpoint-selection luck. Flagging that is not throat-clearing:
> it is the first instance of the discipline this whole tutorial is about, and
> §6 is entirely about what happened when we forgot it.

---

## 0. Would you deploy this?

You train a classifier on cystoscopy images to flag malignant tissue against
everything else. On an internal held-out split it reaches **AUROC 0.796**. Not
spectacular — but then you get access to *another hospital's* data, and it holds:
**0.808**. Discrimination transfers across centres. That is the result everyone
hopes for and most papers do not get.

Your supervisor asks the only question that matters: *would you put this in a
clinic?*

The honest answer is: **you still don't know.** In the real audit those numbers
came from, the model had all but memorised **which light source the endoscopist
had switched on** — and a clinician switches to blue light precisely *when they
already suspect cancer*. Transferring discrimination did not mean the model had
learned the biology.

An AUROC tells you the model separates the classes *on data drawn like your test
split*. It does not tell you *what* it separates them by, whether your test split
was independent of training, whether a "0.9" prediction is really 90% likely, or
what the number means when the disease is rare.

Those four unknowns are four audits. This tutorial walks through each one, with
runnable code, and ends with the audit's most uncomfortable and most valuable
output: **the findings that did not survive scrutiny.**

A high AUROC is where the work starts, not where it ends.

---

## 1. Set-up: the manifest and one command

`medaudit` audits a *frozen* model — you do not retrain anything. You give it a
**manifest**: a CSV with one row per image.

```
path,label,group,attr_mode
img_0001.png,malignant,patient_07,blue_light
img_0002.png,benign,patient_07,white_light
...
```

- `label` — the diagnostic target.
- `group` — patient/case id. Everything downstream is *group-aware*: no patient
  spans train and test, and confidence intervals resample whole patients, not
  rows (images from one patient are not independent).
- `attr_*` — any acquisition/metadata attribute you can record: imaging mode,
  scanner, site, stain. **These are the audit's raw material** — the things a
  model might latch onto instead of the biology.

You also give it **features**: the penultimate-layer activations of your model
(or any embedding) as an `(N, D)` array, row-aligned to the manifest. Auditing
frozen features means the whole toolkit runs on a laptop — no GPU, no retraining.

```python
from medaudit.audit import run_audit          # or: medaudit audit --config audit.json
report = run_audit("audit.json")
```

The rest of this tutorial is what that report contains, one section at a time.
The numbers you'll see come from the synthetic demo shipped in the repo, built
to reproduce — in miniature — what a real audit turned up.

---

## 2. Audit I — the shortcut probe ★

**The war story.** Before any model saw a pixel, the dataset already had a
problem. Suspicious lesions had, in routine practice, been imaged more often
under **blue light** — a contrast mode a clinician reaches for *when they already
suspect pathology*; ordinary tissue was mostly **white light**. Imaging mode and
lesion status were entangled at an **odds ratio of 33.5×** (Cramér's V = 0.462,
χ² ≈ 1725). The shortcut was sitting in the data, waiting to be learned.

A model can get a high AUROC by learning "blue light → malignant." That is a
**shortcut** [1]: a feature that is predictive in your data but is not the
disease, and that breaks the moment the correlation does — a new hospital that
blue-lights everything, or a screening cohort where mode is chosen by protocol,
not suspicion. This is not an exotic failure. Models that appeared to detect
COVID-19 from chest radiographs were substantially reading laterality markers and
source-specific artefacts rather than lung pathology [2]; and aggregate metrics
routinely hide subgroups where the model fails outright [3].

**How to detect it: the linear probe.** Freeze the features. Train a simple
linear classifier to predict the *attribute* (imaging mode) from those features,
evaluated group-aware out-of-fold. If mode is linearly decodable, the model's
representation *encodes* it — it is available to be used as a shortcut.

```python
from medaudit.audits import probe

rep = probe.probe_report(features, attr_codes=mode, labels=diagnosis,
                         groups=patient_id, attr_name="mode")
print(probe.format_report(rep))
```

On the real data the probe read imaging mode out of the frozen features at
**AUROC 0.994** (95% CI 0.986–0.999). The representation had all but memorised
the acquisition mode.

And here is the part that should worry you most: we then tried to *remove* it.
Adversarial de-biasing (a gradient-reversal head, GRL) moved the probe from
**0.994 to 0.969**; Group-DRO [7] reached **0.947** — with all three confidence
intervals overlapping, and every lower bound still ≥ 0.89, i.e. **nowhere near
the 0.5 you'd get if the mode were truly gone.** Standard mitigation did not pull
the shortcut out of the representation. (If you do reach for a mitigation, note
the practical difference: Group-DRO [7] needs group labels throughout training
and careful regularisation, whereas last-layer retraining [8] needs them only for
a small reweighting set — much cheaper to try first.)

Detecting a shortcut is much easier than deleting one — which is exactly why you
should detect it *before* you build on top of it.

**The trap in the probe — and how to escape it.** A high overall probe score is
*ambiguous*. If mode is correlated with the diagnosis (it was, at OR 33×), a probe
can score high just by reading the *class* and exploiting the correlation — that
is not the same as the features genuinely encoding mode. So `medaudit` runs a
second probe **within each fixed class**: among malignant-only images, can you
still decode mode? Holding the diagnosis constant removes the class-collinearity
explanation.

This distinction is the whole game, and it is worth seeing it fail safely on
synthetic data. In the demo we construct features that encode *only the class*,
with the attribute merely correlated with it:

```
shortcut probe · attribute = 'mode'  (decodable if CI lower bound > 0.60)
  overall        AUROC 0.821  (95% CI 0.774–0.868, n=900)
  within benign       AUROC 0.463  (95% CI 0.376–0.547, n=456)
  within malignant    AUROC 0.458  (95% CI 0.392–0.526, n=444)
  -> AMBIGUOUS: decodable overall but not within any single fixed class — the
     overall signal may be driven by class-collinearity rather than a genuinely
     encoded attribute; gather more per-class samples
```

Overall 0.82 *looks* like an encoded shortcut. The within-class probes sitting on
0.5 reveal it is not — the features never carried mode, only class. Report the
overall number alone and you cry wolf. Now the genuinely-encoded case:

```
shortcut probe · attribute = 'mode'  (decodable if CI lower bound > 0.60)
  overall        AUROC 0.996  (95% CI 0.993–0.999, n=660)
  within benign       AUROC 0.998  (95% CI 0.995–1.000, n=360)
  within malignant    AUROC 0.994  (95% CI 0.987–0.998, n=300)
  -> SHORTCUT ENCODED: … remains decodable within fixed classes (benign,
     malignant) — encoded beyond class-collinearity. It is therefore available to
     the model as a potential shortcut; IF the decision head relies on it, expect
     degradation when its link to the label shifts
```

Same headline number, opposite verdict. **The within-class probe is what turns a
suggestive correlation into evidence.**

Notice the hedge in that last verdict — *"available to the model … IF the decision
head relies on it."* A probe shows the attribute is **encoded in the features**;
that is not the same as proving the classifier **uses** it. Stating the weaker,
true claim is the difference between an audit and a scare story.

**A caution that matters.** A linear probe near chance does *not* prove the model
is clean — it only rules out a *linearly* decodable shortcut. Non-linear
shortcuts exist. An audit narrows the space of failure modes; it never certifies
their absence.

---

## 3. Audit II — leakage

**The war story.** Endoscopy is video. Nearby frames of the same lesion are
near-identical. If two near-duplicate frames land one in train and one in test,
the model can *memorise* rather than generalise, and your test score is inflated
by an amount you cannot see in the AUROC.

Two kinds of leakage, both silent:

1. **Group leakage** — the same patient in two splits. Cheap to check exactly
   from ids; `medaudit` does it and refuses to trust a split that fails.
2. **Near-duplicate leakage** — visually near-identical images with *no shared
   id*. There is no key to join on; you must compare the images themselves.

**The instrument matters.** The intuitive tool is a perceptual hash (dHash).
It is the wrong one here. In a homogeneous domain — every frame is pink mucosa —
a perceptual hash sees everything as similar and **over-flags**. On the real data
a dHash scan reported **hundreds of cross-split "duplicates"** that were, on
visual inspection, simply *different images of the same kind of tissue.* We
verified by eye that a hash-distance-zero pair was two genuinely different frames.

The right instrument is **cosine similarity in a learned embedding space**
(an ImageNet backbone, or the model's own features): it keeps true
near-duplicates separable from merely same-domain images.

```python
from medaudit.audits import leakage

rep = leakage.leakage_report(features, split_labels, groups=patient_id,
                             threshold=0.90)
print(leakage.format_report(rep))
```

`medaudit` reports the worst cross-split pairs and their similarity — it **does
not silently delete anything.** You look at the top pairs and decide; the
threshold is something you calibrate by eye for your domain, not a magic default.
The lesson generalises past endoscopy: *choose the duplicate-detector that
matches your domain's homogeneity, and always eyeball what it flags.*

---

## 4. Audit III — calibration

AUROC only cares about the *ranking* of scores [9]. It is completely blind to
whether a "0.9" means 90%. A model can rank perfectly and still be systematically
over-confident — and modern networks generally *are* [4] — which, at the bedside,
is the difference between a probability a clinician can act on and a number that
merely sorts.

`medaudit` uses hand-written, unit-tested calibration metrics (no `sklearn`, so
every number is auditable against its definition):

```python
from medaudit import metrics

print("ECE :", metrics.ece(probs, labels, strategy="equal_mass"))
print("Brier:", metrics.brier(probs, labels))
for conf, acc, n in metrics.reliability_curve(probs, labels):
    ...   # plot acc vs conf; the diagonal is perfect calibration
```

Two habits worth teaching:

- **Report Brier alongside ECE.** ECE is a biased estimator of calibration error
  and its value depends on how you bin [11, 12]; equal-mass (quantile) bins are
  more robust than equal-width, but ECE alone can mislead. Brier [5] is a proper
  scoring rule — pair them.
- **Calibrate on data the model was not selected on.** Picking a temperature on
  the same split you report calibration for is the calibration version of
  training on the test set.

---

## 5. Audit IV — prevalence, and the metric that matters clinically

Here is the audit that most often explains "it worked in the lab, it failed in
the clinic."

**AUROC is prevalence-invariant.** By construction it does not change when the
disease gets rarer. That is a feature for comparing models — and a trap for
predicting clinical behaviour, because the clinic *feels* prevalence directly.

At a screening prevalence of ~1%, the quantity a clinician lives with is
**positive predictive value (PPV)** [10]: of the cases the model flags, how many
are truly positive? You do not need a citation for why it collapses — just the
arithmetic:

```
              sens · prev
PPV = ─────────────────────────────────
      sens · prev + (1 − spec)(1 − prev)
```

Hold sensitivity at 0.90 and specificity at a strong 0.95. At 50% prevalence,
PPV = **0.95**. At 10% it is already **0.67**. At **1% prevalence the same model
gives PPV = 0.15** — roughly **85% of its flags are false alarms**, and about six
and a half flags for every true positive found. Not one number describing the
model changed. Prevalence did all of it. This is not a thought experiment: screening challenges
now score exactly this. Barrett's oesophagus neoplasia detection at EndoVis/MICCAI
2026, for instance, ranks entries by *PPV at 90% recall, evaluated under a
simulated ~1% prevalence*. Leaderboard numbers there look crushingly low next to
the AUROCs in the same papers — which is the point of the metric, because the low
number is the one the clinic actually experiences.

Two things to do:

- **Report the operating point, not just the curve.** Fix a clinically relevant
  recall; report PPV *at the prevalence you'll deploy at*, not your test split's.
  If the deployment base rate differs from training, the classifier's outputs can
  be corrected for the new prior rather than merely re-thresholded [6].
- **Give it an honest confidence interval.** Use the **cluster bootstrap** —
  resample whole patients, not rows [14, 13]. A per-image bootstrap understates
  uncertainty because one patient's frames move together; it is the same
  independence assumption that group-aware splitting protects in §1, showing up
  again in your error bars.

```python
from medaudit.metrics import cluster_bootstrap, auroc
point, lo, hi = cluster_bootstrap(patient_id, lambda idx: auroc(scores[idx], y[idx]))
print(f"AUROC {point:.3f}  (95% CI {lo:.3f}–{hi:.3f})")
```

When a resampled patient-subgroup has only one class, the statistic is undefined;
`medaudit` drops those resamples and *warns you* that the subgroup is
underpowered — because a silently-narrow CI is worse than an honestly-wide one.

---

## 6. The honest part: findings that died ★

The most valuable output of the real audit was not a green check. It was a series
of headlines we **retracted ourselves.** Four, each with a transferable lesson.

**1. We compared apples to oranges.** Our first alarming result: malignant
sensitivity fell from **0.586 internally to 0.378 externally**. Except the
internal number was computed by `argmax` over five classes and the external one
by thresholding *P > 0.5* — two different decision rules. Half the "collapse" was
a unit error. *Lesson: before you explain a gap, check that both sides were
measured the same way.*

**2. "The external collapse proves the shortcut fails on new data."** It doesn't
— because there was no collapse in *discrimination*. Re-measured under one rule
on a matched domain, internal white-light malignant AUROC was **0.796** and
external **0.808**. The ranking ability transferred almost perfectly. What
actually degraded was the **operating point**: with the same P > 0.5 rule,
sensitivity was 0.570 internally and 0.378 externally — a calibration mismatch,
not a discrimination failure. *Lesson: "it got worse externally" is not a
diagnosis. Separate discrimination from calibration before assigning blame — and
be suspicious when the culprit happens to be your favourite hypothesis.*

**3. "The two biases cancel."** We had a satisfying story where the shortcut bias
and a prevalence shift offset each other. It died twice over. First, a
**metric-category error**: we were measuring the effect with **AUROC**, which is
*prevalence-invariant by construction* — a metric that literally cannot see a
prevalence effect cannot be used to argue about one. Second, re-measured with a
prevalence-*sensitive* metric, the effect was **≈ 0**. There was never a signal to
explain. (The prevalence shift was real and large — malignant 9% internally vs
65% externally — which is precisely why it, not the shortcut, deserves the credit
for the calibration mismatch in #2.)

**4. Hundreds of duplicates that never existed** — the dHash false alarm from
Audit II.

**Why put this in a tutorial?** Because *self-skepticism is the method, not a
disclaimer at the end.* Every one of those four was killed by a discipline that is
cheap to apply and boring to describe: compare like with like; pick a metric that
can see your claim; get an honest CI; match the instrument to the domain. That is
the same discipline that catches someone else's over-claim in review. An audit you
cannot fail is not an audit.

**Why put this in a tutorial?** Because *self-skepticism is the method, not a
disclaimer at the end.* The discipline that kills your own favourite result — the
right metric for the claim, an honest CI, an instrument matched to the domain — is
the same discipline that would have caught someone else's over-claim in review.
An audit you cannot fail is not an audit. Reporting the retraction is the most
honest thing this project did, and it is the transferable skill.

---

## 7. Run it yourself

```bash
git clone <repo-url> && cd medaudit
pip install -e .               # numpy-only core; torch optional, for feature extraction
medaudit audit --config audit.json
```

`audit.json` points at your manifest and a precomputed `features.npy`; the report
prints the four audits with confidence intervals and a plain-language verdict for
each. Point it at your own model's features and metadata and you have, in one
command, the checks this tutorial walked through.

**What an audit does and does not give you.** It narrows failure modes — a
linear probe that clears chance is real evidence of an encoded shortcut; one that
doesn't is *not* a clean bill of health (non-linear shortcuts survive). Embedding
duplicate-detection misses what the embedding can't see. Calibration on a held-out
split doesn't guarantee calibration in a new hospital. The audit's job is to move
you from "AUROC 0.92, ship it" to a defensible, itemised account of *what you
checked, what you found, and what you still don't know* — which is exactly the
account a clinical reviewer, and a patient, deserve.

From pixels to patients, the last mile is not a higher number. It is the audit
that tells you whether the number means what you hoped.

---

## Code & reproducibility

All code: the `medaudit` repository (MIT). Every metric is unit-tested against an
independent brute-force reference (27 tests, `tests/`).

**Every number printed in this tutorial regenerates with one command:**

```bash
python tutorial/make_demo.py          # prints §2's two probe reports and §1's full audit
```

No patient data is redistributed and none is needed to follow along: the runnable
examples are synthetic by design, built to mirror the real audit's structure in
miniature. The war stories quote aggregate results from an audit of a classifier
trained on a 2025 public cystoscopy dataset — no images, no rows, no raw data.
(The repository's `.gitignore` refuses images, arrays and weights by default. If
you fork this to audit your own model, keep it that way: the code is what ships.)

---

## References

[1] R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and
F. A. Wichmann. Shortcut learning in deep neural networks. *Nature Machine
Intelligence*, 2(11):665–673, 2020. doi:10.1038/s42256-020-00257-z

[2] A. J. DeGrave, J. D. Janizek, and S.-I. Lee. AI for radiographic COVID-19
detection selects shortcuts over signal. *Nature Machine Intelligence*,
3(7):610–619, 2021. doi:10.1038/s42256-021-00338-7

[3] L. Oakden-Rayner, J. Dunnmon, G. Carneiro, and C. Ré. Hidden stratification
causes clinically meaningful failures in machine learning for medical imaging. In
*Proceedings of the ACM Conference on Health, Inference, and Learning (CHIL)*,
pages 151–159, 2020. doi:10.1145/3368555.3384468

[4] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern
neural networks. In *Proceedings of the 34th International Conference on Machine
Learning (ICML)*, PMLR 70:1321–1330, 2017.

[5] G. W. Brier. Verification of forecasts expressed in terms of probability.
*Monthly Weather Review*, 78(1):1–3, 1950.
doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2

[6] M. Saerens, P. Latinne, and C. Decaestecker. Adjusting the outputs of a
classifier to new a priori probabilities: A simple procedure. *Neural
Computation*, 14(1):21–41, 2002. doi:10.1162/089976602753284446

[7] S. Sagawa, P. W. Koh, T. B. Hashimoto, and P. Liang. Distributionally robust
neural networks for group shifts: On the importance of regularization for
worst-case generalization. arXiv:1911.08731, 2019. Published at the 8th
International Conference on Learning Representations (ICLR), 2020.

[8] P. Kirichenko, P. Izmailov, and A. G. Wilson. Last layer re-training is
sufficient for robustness to spurious correlations. In *The Eleventh
International Conference on Learning Representations (ICLR)*, 2023.

[9] M. H. Zweig and G. Campbell. Receiver-operating characteristic (ROC) plots: A
fundamental evaluation tool in clinical medicine. *Clinical Chemistry*,
39(4):561–577, 1993. doi:10.1093/clinchem/39.4.561

[10] D. G. Altman and J. M. Bland. Statistics Notes: Diagnostic tests 2:
Predictive values. *BMJ*, 309(6947):102, 1994. doi:10.1136/bmj.309.6947.102

[11] J. Nixon, M. W. Dusenberry, L. Zhang, G. Jerfel, and D. Tran. Measuring
calibration in deep learning. In *CVPR Workshops*, pages 38–41, 2019.

[12] R. Roelofs, N. Cain, J. Shlens, and M. C. Mozer. Mitigating bias in
calibration error estimation. In *Proceedings of the 25th International
Conference on Artificial Intelligence and Statistics (AISTATS)*, PMLR
151:4036–4054, 2022.

[13] C. A. Field and A. H. Welsh. Bootstrapping clustered data. *Journal of the
Royal Statistical Society: Series B*, 69(3):369–390, 2007.
doi:10.1111/j.1467-9868.2007.00593.x

[14] A. C. Davison and D. V. Hinkley. *Bootstrap Methods and their Application*.
Cambridge University Press, 1997. ISBN 0-521-57471-4

---
