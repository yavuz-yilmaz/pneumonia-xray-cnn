# Fourth experiment and independent RSNA transfer results

## Pediatric validation selection

The incumbent CLAHE ResNet18 won the prespecified validation comparison. Neither new candidate was evaluated on the legacy test; the incumbent retains its previously reported test result. Default deployment is unchanged.

| Candidate | Validation recall | Validation specificity | Validation F1 |
|---|---:|---:|---:|
| resnet18_clahe | 98.09% | 98.89% | 98.82% |
| resnet18_acquisition | 98.38% | 97.41% | 98.67% |
| efficientnet_v2_s_acquisition | 98.09% | 97.04% | 98.45% |

## External transfer: 1,000 distinct NIH patients

This is a balanced adult lung-opacity versus normal cohort, not pediatric pneumonia ground truth. Both thresholds were fixed before external scoring. No external threshold optimization or model selection was performed.

| Metric | Original baseline | Selected clean model | Clean model 95% interval |
|---|---:|---:|---:|
| accuracy | 75.80% | 83.80% | 81.30%–86.00% |
| precision | 71.72% | 84.92% | 81.72%–87.79% |
| recall | 85.20% | 82.20% | 78.91%–85.63% |
| specificity | 66.40% | 85.40% | 82.06%–88.54% |
| f1 | 77.88% | 83.54% | 80.89%–85.98% |
| roc_auc | 84.81% | 90.88% | 88.99%–92.51% |

False positives decreased **168 → 73**, but false negatives increased **74 → 89**. Accuracy improved by 8 percentage points, while recall fell by 3 percentage points. Thus the requested improvement in every metric is **not established**.

The external evaluation is now observed. Any future experiments responding to these results must disclose this exposure and cannot describe the same cohort as newly untouched. Its patient identities remain permanently reserved from training and calibration. Further model development should rely on development-only grouped validation.

A separate RSNA development inventory, excluding all 1,000 reserved patients, contains 11,233 patients. Taking one hash-ranked acquisition per patient yields 5,807 normal, 1,007 lung-opacity and 4,419 other-abnormal cases. These source labels must not be silently converted into pediatric pneumonia labels.

See [external protocol and audit](rsna_external_protocol.md) and [training protocol](clean_training_protocol.md). Full score arrays are retained in `reports/metrics/rsna_external_v1`; aggregate metrics are also published in `rsna_external_results.json`.
