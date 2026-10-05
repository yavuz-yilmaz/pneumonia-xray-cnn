# SSMU one-time proxy-group test — registered before test scoring

The fixed frozen-feature development run completed without test access. Its
validation AP rule selected C=0.01. The joint validation threshold is
0.7265916466712952. At that threshold, accuracy/precision/recall/F1 are
79.70% / 100% / 63.51% / 77.69%. Original precision is already 100%, so this
validation result does not show a substantial increase in every requested metric.

## Frozen comparison

- Selected input: `models/ssmu_probe_v1/c0p01/best_model.pt`, SHA256
  `a0b921548b65125a43c8d86c8e634bc16edcdc822f18900988e404bd3ce43624`.
  Export the same tensors with the joint threshold above before inference.
- Compare only that model, the pinned original and pinned current models.
  Keep their saved thresholds (0.7 and 0.8820025324821472 respectively), input
  preprocessing, and weights unchanged. No other C candidate receives test scores.
- Locked test: `data/processed/ssmu_frontal_v1/test_manifest.csv`, SHA256
  `55745a8a15709488fd74868cc394f7a1b73ce40972b3fedade9b763482c9ec59`.
  It contains 131 images, 72 source-pneumonia and 59 source-normal labels,
  in 71 conservative proxy/duplicate groups. Age is unverified.
- Before inference, verify training/source inputs, original/current hashes,
  validation-label order, AP selection, exported threshold, unchanged MIMIC
  feature tensors, and zero recorded split overlap. Register evaluator hashes,
  all three checkpoint hashes and the test manifest hash.
- Score the locked test once with serving preprocessing in float32. Preserve
  aligned probabilities, groups, thresholds, confusion matrices and metrics.
  No threshold fitting, checkpoint replacement or additional recipe selection
  follows test observation in this protocol.
- Use 1,000 paired bootstrap repetitions resampling the same conservative
  groups for all models; report 95% intervals for metrics and paired differences.
  These are proxy-group intervals, not verified patient-level intervals.

Report all outcomes, including ties, failures and uncertainty. A zero known
overlap audit does not prove unknown patient identities independent. Source
folder diagnoses remain unverified. Results are source-specific and do not
establish an improvement on the legacy pediatric benchmark or clinical use.
No automatic deployment/publication or full-goal completion follows this test.
