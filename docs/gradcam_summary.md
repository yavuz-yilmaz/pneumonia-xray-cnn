> **Historical record:** This document preserves the original model's analysis. For current results, see the [short project report](project_report.md) and [main comparison](clean_v3_matched98_results.md).

# Grad-CAM Explainability Summary

## What Is Grad-CAM?

Grad-CAM is an explainability method that approximates which regions in a CNN's
final convolutional feature maps contribute most to a particular class decision.
The heatmap highlights regions with a higher contribution to the model's class
score using warm colors.

## How Was It Used in This Project?

- Explained model: `resnet18`
- Runtime device: `cuda`
- Target layer: the selected architecture's final convolutional feature layer
- Explained class: the model's predicted class
- Figure format: original X-ray and Grad-CAM heatmap overlay

Selected samples:

- Correctly classified NORMAL: `4`
- Correctly classified PNEUMONIA: `4`
- Misclassified samples: `6`

## Which Regions Does the Model Appear to Focus On?

The overlays can help examine whether activations associated with the model's
decision concentrate within lung regions. For PNEUMONIA predictions, check whether
the highlighted regions remain over the lungs. Heatmaps concentrated on image
borders, text, or areas outside the lungs may indicate that the model has learned
cues with no clinical relevance.

## Limitations

Grad-CAM is not a causal explanation; it provides an approximate visual
interpretation based on the model's final convolutional features. The heatmap
must not be treated as high-resolution pathology localization. This project is
for educational purposes; predictions and explanations must not be used for
medical diagnosis.

## Generated Figures

- `reports\figures\gradcam_correct_normal_01.png`
- `reports\figures\gradcam_correct_normal_02.png`
- `reports\figures\gradcam_correct_normal_03.png`
- `reports\figures\gradcam_correct_normal_04.png`
- `reports\figures\gradcam_correct_pneumonia_01.png`
- `reports\figures\gradcam_correct_pneumonia_02.png`
- `reports\figures\gradcam_correct_pneumonia_03.png`
- `reports\figures\gradcam_correct_pneumonia_04.png`
- `reports\figures\gradcam_misclassified_01.png`
- `reports\figures\gradcam_misclassified_02.png`
- `reports\figures\gradcam_misclassified_03.png`
- `reports\figures\gradcam_misclassified_04.png`
- `reports\figures\gradcam_misclassified_05.png`
- `reports\figures\gradcam_misclassified_06.png`

## Note

Figures were generated for 6 misclassified samples.

## Warning

This project is for educational purposes; its outputs must not be used for medical diagnosis.
