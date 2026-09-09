"""Small synthetic worked examples for tutorial sections 4 and 5.

Run from the repository root after installation: python tutorial/worked_examples.py
These compose existing metrics; they do not add checks to `medaudit audit`.
"""
import numpy as np

from medaudit import metrics


def main():
    # Fixed teaching inputs, not patient predictions; rows are [P(class 0), P(class 1)].
    positive = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    probs = np.column_stack([1 - positive, positive])
    print("Synthetic calibration example (8 rows; not a clinical estimate)")
    print(f"AUROC: {metrics.auroc(positive, labels):.3f}")
    print(f"Top-label ECE (4 equal-mass bins): {metrics.ece(probs, labels, n_bins=4, strategy='equal_mass'):.3f}")
    print(f"Multiclass Brier (sum over both classes): {metrics.brier(probs, labels):.3f}")
    print("Reliability bins (4 equal-width bins): confidence, accuracy, count")
    for conf, acc, count in metrics.reliability_curve(probs, labels, n_bins=4):
        print(f"  {conf:.3f}, {acc:.3f}, {count}")
    print("Empty bins have nan means; ECE and this curve use different binning.")

    # Hold sensitivity/specificity fixed and vary prevalence to illustrate Bayes arithmetic.
    sensitivity, specificity = 0.90, 0.95
    print("\nPrevalence arithmetic (fixed sensitivity=0.90, specificity=0.95)")
    for prevalence in (0.50, 0.10, 0.01):
        ppv = sensitivity * prevalence / (
            sensitivity * prevalence + (1 - specificity) * (1 - prevalence))
        print(f"  prevalence={prevalence:.0%}: PPV={ppv:.3f}")
    print("Fixed operating characteristics are an assumption, not a transfer guarantee.")


if __name__ == '__main__':
    main()
