# Clean retraining: measured model comparison

## Locked selection

- Selected experiment: `resnet18_clahe`
- Current local inference checkpoint (explicitly configured): `models/clean_v3/selected_matched98/best_model.pt`
- Checkpoint SHA-256: `da5c58d36865cc01d8df1dce5c358dac2a06b0b0b4981115bb7e35d806e2be9b`
- Operating threshold: `0.88200253`
- Selection used validation specificity at recall >=98%; F1 breaks ties.
- The previous deployed model's documented validation recall floor was 98%.
- The threshold was fixed before evaluating the selected model on the legacy test.
- Validation operating point: recall 98.09%,
  specificity 98.89%, F1 98.82%.

## Same 624-image legacy test

The previous model uses its deployed 0.70 threshold. New metrics use the new
validation-selected threshold. Neither threshold is optimized on these test labels.
Changes are percentage points; intervals are 95% patient-proxy cluster bootstrap
intervals (1,000 resamples), not evidence of clinical reliability.

| Metric | Previous | Clean selected | Change (pp) | New 95% interval |
|---|---:|---:|---:|---:|
| Accuracy | 88.30% | 89.58% | +1.28 | 86.93%–92.01% |
| Precision | 84.68% | 86.03% | +1.35 | 82.19%–89.43% |
| Recall | 99.23% | 99.49% | +0.26 | 98.62%–100.00% |
| Specificity | 70.09% | 73.08% | +2.99 | 66.67%–78.39% |
| F1 | 91.38% | 92.27% | +0.89 | 90.03%–94.21% |
| ROC-AUC | 95.99% | 97.92% | +1.93 | 96.76%–98.85% |

- False positives: **70 → 63**.
- False negatives: **3 → 2**.

![Same-test confusion matrices](figures/clean_v3_matched98_results_comparison.png)

## Validation comparison used for selection

These validation results must not be compared directly with the test results above.

| Experiment | Accuracy | Precision | Recall | F1 | Specificity |
|---|---:|---:|---:|---:|---:|
| xrv_clahe_finetuned | 97.48% | 98.38% | 98.09% | 98.24% | 95.93% |
| resnet18_clahe | 98.32% | 99.55% | 98.09% | 98.82% | 98.89% |

## Leakage controls and limits

- Train: 3802 images; validation: 951;
  unchanged original test: 624.
- Excluded development images: 479; raw files retained.
- No shared filename-derived patient groups, byte hashes, decoded pixel hashes,
  or connected duplicate groups between any pair of splits.
- Detected cross-split near-duplicate pairs: 0.
- Identities are conservative filename proxies. Actual patient-disjointness cannot
  be established beyond the supplied metadata. Similarity screening is not exhaustive.
- The legacy test was inspected by older project versions. This is a same-dataset
  benchmark comparison, not untouched external validation or clinical validation.
- Validation threshold selection has its own estimation uncertainty; the 98% recall
  target does not guarantee the same recall on a new population.

## Reproducibility

See [the experiment protocol](clean_training_protocol.md). Full audit, excluded
manifest and original source hashes are in `data/processed/clean_v1`. Each experiment
contains its configuration, training history, validation predictions, and checkpoint.
The selected directory contains the locked selection, test predictions, bootstrap
intervals, and the machine-readable comparison. Previous artifacts are retained.

Educational/research model; not a medical diagnostic device.
