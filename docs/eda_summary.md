# EDA Summary

This document was automatically generated with `python -m src.eda.run_eda --config configs/config.yaml`.

## Dataset Summary

- Readable image count: 5856
- Unreadable image count: 0

## Class Distribution

| Split | NORMAL | PNEUMONIA | Total |
| --- | ---: | ---: | ---: |
| train | 1341 | 3875 | 5216 |
| val | 8 | 8 | 16 |
| test | 234 | 390 | 624 |

## Numerical Summary

- Majority/minority class ratio: 2.699
- Mean width: 1327.881 pixels
- Mean height: 970.689 pixels
- Mean aspect ratio: 1.443
- Mean pixel intensity: 122.786

## Potential Issues

- The class distribution appears imbalanced; consider class weights or a sampler during training.
- Aspect ratios vary widely; monitor distortion introduced by resizing.
- No unreadable images were found during EDA.

## Modeling Recommendations

- To address the imbalance between PNEUMONIA and NORMAL, try weighted loss or WeightedRandomSampler.
- When resizing images to a fixed size, use the same normalization for training and inference.
- Avoid aggressive augmentations that could alter medically meaningful features.
- During model selection, report recall, precision, F1-score, and the confusion matrix alongside accuracy.

## Generated Figures

- `reports/figures/eda_class_distribution.png`
- `reports/figures/eda_sample_grid_NORMAL.png`
- `reports/figures/eda_sample_grid_PNEUMONIA.png`
- `reports/figures/eda_image_size_distribution.png`
- `reports/figures/eda_aspect_ratio_distribution.png`
- `reports/figures/eda_pixel_intensity_histogram.png`
- `reports/figures/eda_average_pixel_maps.png`
