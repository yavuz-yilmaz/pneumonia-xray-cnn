> **Historical record:** This document preserves the original ResNet18 model's analysis. For current results, see the [short project report](project_report.md) and [main comparison](clean_v3_matched98_results.md).

# Test Set Evaluation Summary

## Test Results

- Model: `resnet18`
- Checkpoint: `models\best_model.pt`
- Test sample count: `624`
- Accuracy: `0.8574`
- Precision: `0.8168`
- Recall: `0.9949`
- F1-score: `0.8971`
- ROC-AUC: `0.9599`

## Decision Threshold Optimization

The default `0.50` classification threshold was reviewed on the validation set.
Among thresholds maintaining recall above `0.98`, `0.70` gave the highest
specificity and became the default for the historical inference/API workflow.

| Metric | Threshold 0.50 | Threshold 0.70 |
|---|---:|---:|
| Accuracy | 0.8574 | **0.8830** |
| Precision | 0.8168 | 0.8468 |
| Recall | 0.9949 | **0.9923** |
| F1-score | 0.8971 | **0.9138** |
| Specificity | 0.6282 | **0.7009** |
| False positive | 87 | **70** |
| False negative | 2 | 3 |

Threshold `0.70` was selected on validation and then checked once on the test
set in the original experiment. It reduced false positives while maintaining
pneumonia recall at approximately `99%`.

## Strongest Metrics

- `recall`: `0.9949`
- `roc_auc`: `0.9599`

## Weaknesses

- `precision`: `0.8168`
- `accuracy`: `0.8574`

## False Positive / False Negative Interpretation

- False positive count: `87`. The model classified a NORMAL image as PNEUMONIA.
- False negative count: `2`. The model classified a PNEUMONIA image as NORMAL.
- Misclassified samples: `reports\figures\misclassified_examples.png`

## Why Does Recall Matter in a Medical Context?

Recall measures the proportion of pneumonia-positive samples detected. Because PNEUMONIA is the positive class in this project, low recall means some images with pneumonia findings are missed and classified as NORMAL. This error can be more serious clinically; however, this project is intended only for educational and academic work.

## Generated Outputs

- Metrics: `reports\metrics\test_metrics.json`
- Confusion matrix: `reports\figures\confusion_matrix.png`
- ROC curve: `reports\figures\roc_curve.png`
- Precision-recall curve: `reports\figures\precision_recall_curve.png`
- Misclassified samples: `reports\figures\misclassified_examples.png`

## Warning

This project is for educational purposes; its outputs must not be used for medical diagnosis.
