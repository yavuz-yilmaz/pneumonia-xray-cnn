# Presentation outline

Use the [short report](project_report.md) for current results.

1. **Purpose:** A NORMAL/PNEUMONIA classification research project; not a clinical diagnostic tool.
2. **Main workflow:** data preparation → group/duplicate audit → training → validation selection → inference/API.
3. **Data:** 5,856 raw Kermany images; 3,802 training, 951 validation, 624 test; 479 excluded development images.
4. **Leakage controls:** filename groups, byte/pixel hashes, and near duplicates; limited by unavailable verified patient identities.
5. **Main model:** ResNet18, ImageNet initialization, 224px CLAHE, registered 98% validation recall condition.
6. **Same-test results:** accuracy 88.30% → 89.58%, F1 91.38% → 92.27%; FP 70 → 63, FN 3 → 2.
7. **Unsuccessful experiments:** NIH/MIMIC/RSNA candidates and the limitation of repeated test exposure.
8. **Source shift:** SSMU candidate F1 86.36% on its own test; 2.82% on OpenI with 1,505 false positives.
9. **Demo:** JSON prediction with your own checkpoint, `/model-info`, FastAPI, and Streamlit. Model probabilities are not clinical risk.
10. **Conclusion:** Traceable implementation and honest evaluation; future modeling needs verified patient/label information and a fresh external test.
