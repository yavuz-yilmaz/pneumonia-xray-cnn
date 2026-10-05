# NIH MIMIC final-block adaptation — completed 2026-10-02

The ten-epoch float32 experiment completed. It did **not** establish the
requested improvement across accuracy, precision, recall and F1. The deployed
checkpoint remains unchanged. This is exploratory development on NIH validation
after earlier NIH/OpenI test observation; it is not an untouched external test.

## Recipe and selection

The [registered recipe](nih_transfer_protocol.md) updates DenseNet121's final
block and classifier from the MIMIC-only frozen probe. Earlier features and all
BatchNorm running statistics remain frozen. Highest validation average precision
selects the checkpoint; epoch 9 won. Training used only the existing audited NIH
training/validation split. No NIH test or OpenI scoring occurred in this run.

## Same-cohort validation results

All rows use the same 14,212 NIH validation images, including 155 report-derived
pneumonia positives. Negative labels include other diseases. The adapted row
uses the recipe's recall-oriented operating point; it is not a successful joint
threshold. These percentages cannot be compared directly to the old pediatric
benchmark as a before/after improvement.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Original classifier | 49.26% | 1.47% | 69.03% | 2.88% | 0.6032 | 0.014763 |
| Current deployed classifier | 62.84% | 1.44% | 49.03% | 2.80% | 0.6136 | 0.015817 |
| Initial frozen MIMIC probe | 25.91% | 1.31% | 90.32% | 2.59% | 0.6471 | 0.021111 |
| Adapted final block | 24.87% | 1.30% | 90.32% | 2.56% | 0.6653 | 0.022926 |

The adapted classifier produced 140 true positives, 15 false negatives, **10,663
false positives**, and 3,394 true negatives. Its small average-precision gain
over the frozen probe does not establish the requested four-metric improvement.

An independent review enumerated all **14,197 distinct validation thresholds**.
**None** met all accuracy/precision/recall/F1 floors of the two saved original
and current validation baselines. No floors were relaxed. `joint_threshold` and
`joint_validation` remain null; no replacement was deployed.

## Failure preservation and verification

The initial AMP attempt overflowed at epoch one and was deliberately stopped.
Its checkpoint retained all 727 initial tensors. On two actual training images,
the diagnosis reproduced nonfinite AMP logits/loss/gradients while float32 was
finite. Failed artifacts and the original source/protocol snapshots are retained
in `reports/metrics/nih_mimic_tail_v1_failure`.

The corrected trainer rejects nonfinite logits, loss and gradient norm. Before
launch, 22 focused tests and Ruff passed, including a tiny end-to-end training
run guarding against NIH test/OpenI access and a forced nonfinite failure.
The final independent artifact review confirmed 727 finite tensors, 98 changes
only in allowed parameters, unchanged BatchNorm buffers, selection/label order,
confusion matrices, joint-threshold failure and no source/input hash drift.
Post-run original-image rehashing also passed for all 65,662 training and 14,212
validation images, without opening test images. That check does not rehash the
derived training cache.

Evidence: `models/nih_mimic_tail_v2/result.json`,
`reports/metrics/nih_mimic_tail_v2_final_review/artifact_review.json`, and
`post_run_development_image_hashes.json` in the same review directory.
Checkpoint SHA256:
`2173b40e58cffeffe06f1725825605115e0478b519742b37b9c36c05635b9ba8`.
Runtime versions are recorded locally; NumPy 2.5.2 exceeded the project's
declared `<2.3` constraint. No dependency/environment change was made.

The broader improvement goal remains unmet. Further work requires a tangible
data/label improvement and a defensible evaluation cohort, rather than treating
another completed training run as success.
