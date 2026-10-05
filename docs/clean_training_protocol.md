# Clean retraining protocol

## Status

The first clean comparison completed. ResNet18 completed 12 epochs with early stopping
(best checkpoint: epoch 6, validation cross entropy 0.04556, runtime 1053 seconds).
EfficientNet-B0 also completed 12 epochs (best checkpoint: epoch 6, runtime 1154 seconds).
ResNet18 won the prespecified validation comparison. Its legacy-test accuracy was
87.34%, precision 83.30%, recall 99.74%, and F1 90.78%. This does **not** improve the
previous deployed model's accuracy/precision/F1; it is not promoted to deployment.
See [the measured comparison](clean_model_results.md).

## Second experiment: external chest-X-ray representations

Following the unsuccessful first experiment, `clean_v2` evaluates a new hypothesis:
features pretrained on large multi-source chest-X-ray datasets may generalize better
than ImageNet features fine-tuned on this small pediatric collection. This experiment
is recorded separately; the legacy test is already observed and is not described as
an untouched test. No legacy test image is used for fitting the new model or choosing
its operating threshold.

The frozen TorchXRayVision `densenet121-res224-all` backbone is described by its
provider as trained on NIH, PadChest, CheXpert, MIMIC-CXR, Google/NIH, OpenI and RSNA.
The `kaggle` token in the weight filename refers to the RSNA Pneumonia Challenge,
not the Kermany pediatric dataset used here. This provenance is provider-reported;
we do not have the original pretraining image manifest for an independent overlap audit.

- Source: https://github.com/mlmed/torchxrayvision
- Provider metadata: https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/models.py
- Download SHA-256: `56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899`
- Training-only loader: `torchxrayvision==1.5.4`; exported inference needs torchvision.
- Input: 224x224 grayscale, normalized to [-1024, 1024] as in XRV.
- Fit feature standardization on training images only.
- Compare L2 logistic heads with C in {0.01, 0.1, 1, 10}, selected on validation.
- Keep the same data split and validation recall floor of 99%.
- Export the fitted scaler into the linear classifier, preserving a standard model checkpoint.

```bash
python -m pip install -e ".[xray]"
python -m src.training.train_xray_probe
```

The second experiment completed. C=0.01 won its validation comparison, but legacy-test
accuracy was 83.97%, precision 79.84%, recall 99.49%, and F1 88.58%. It is not promoted.
See [the second measured comparison](clean_v2_results.md).

## Third experiment: contrast standardization and fine-tuning

The first two failures motivate a separately recorded `clean_v3` experiment. Its
hypothesis is that acquisition contrast variation and insufficient adaptation of
the external representation contribute to the development/test gap. This is a
hypothesis, not an established causal explanation of the failures.

- Deterministically resize to 224px then apply grayscale CLAHE (clipLimit=2,
  8x8 tile grid), retaining the original labels and group-disjoint manifests.
- Compare fully fine-tuned XRV DenseNet121 and ImageNet ResNet18 on these same
  prepared images. No test-based parameter sweep is performed.
- DenseNet: batch size 16, head LR 0.0003, backbone LR 0.00006; ResNet: batch 24,
  head LR 0.0005, backbone LR 0.0001. Both allow 16 epochs with two head-warmup
  epochs and early stopping after five non-improving validation losses.
- Retain the same validation threshold and candidate selection rule.
- Lossless prepared-image caching changes I/O cost, not model inputs. Tests assert
  exact equality between cached evaluation tensors and deployed preprocessing.
- Record all third-experiment choices before its legacy-test evaluation. Prior
  legacy-test exposures remain disclosed; this is not a new untouched test.

```bash
python -m src.training.train_clean --model xrv_densenet121 --preprocessing clahe --cache-images --batch-size 16 --epochs 16 --lr 0.0003 --patience 5 --output models/clean_v3/xrv_clahe_finetuned
python -m src.training.train_clean --model resnet18 --preprocessing clahe --cache-images --epochs 16 --lr 0.0005 --patience 5 --output models/clean_v3/resnet18_clahe
```

### Matched operating-policy correction, registered before v3 test evaluation

The historical baseline's documented operating point maximized validation specificity
subject to recall above **98%** (`docs/evaluation_summary.md`, lines 16–18). The first
clean experiments instead imposed **99%**, a stricter sensitivity requirement. Those
results should not be treated as a comparison at the same operating policy.

At `2026-09-15 21:22:09 UTC`, before any v3 test report existed, the primary v3 policy
was fixed to the historical **98% validation recall floor**. The originally scheduled
99% policy remains a reported sensitivity analysis. Both policies select their model
and threshold using validation only; the primary policy is not chosen according to
which test result is better. The full registration is in
`configs/evaluation_policy_v3.json`. Prior benchmark exposures remain disclosed.

The third experiment completed. ResNet18 was selected by validation under both
policies. The primary 98% policy selected threshold 0.88200253. On the 624-image
legacy benchmark it reached accuracy 89.58%, precision 86.03%, recall 99.49%,
F1 92.27%, and ROC-AUC 97.92%. All five point estimates exceed the original
deployed baseline, but the improvement in accuracy/F1 is modest, not a demonstrated
large or externally validated improvement. The primary checkpoint is now configured
for local inference; the original `models/best_model.pt` is retained unchanged.

See [primary results](clean_v3_matched98_results.md) and the
[99% sensitivity analysis](clean_v3_results.md). The latter has lower accuracy and
F1 than the old model; it is not the deployed policy. Confidence intervals on a
finite benchmark do not establish a statistically significant paired improvement.

```bash
python -m src.evaluation.evaluate_clean select --experiments models/clean_v3/xrv_clahe_finetuned models/clean_v3/resnet18_clahe --minimum-recall 0.98 --output models/clean_v3/selected_matched98
python -m src.evaluation.evaluate_clean test --output models/clean_v3/selected_matched98
python -m src.evaluation.report_clean --selected models/clean_v3/selected_matched98 --destination docs/clean_v3_matched98_results.md
```

The `report_clean` command is a historical comparison utility, not a fresh-clone
reproduction step. It requires retained local artifacts that are not distributed
in this repository: `reports/metrics/legacy_v0_test_metrics.json`,
`data/processed/test_manifest.csv`, `data/processed/clean_v1/test_manifest.csv`,
and the selected run's `selection.json`, `test_metrics.json` and
`test_predictions.npz`. The two manifests and saved predictions must describe
the same ordered test images. Without these artifacts, use the selected run's
own evaluation outputs; the main README training/evaluation workflow does not
require this historical report generator.

The selection and test commands reproduce the workflow in a fresh output directory. Existing locked
selections and final test reports deliberately refuse overwriting. Further model
development should use grouped cross-validation and a separately sourced, documented
external evaluation cohort. Repeated tuning against this legacy test cannot provide
an unbiased claim of generalization.

## Fourth experiment: capacity and acquisition augmentation

`configs/evaluation_policy_v4.json` records the next comparison before its test
evaluation. Two additional runs use the same audited splits and CLAHE: ResNet18
and torchvision ImageNet-pretrained EfficientNetV2-S. Both use 90–100% area crops,
mild affine transforms, brightness/contrast variation, random autocontrast and
occasional image softening. These augmentations apply only during training.

The incumbent ResNet18 remains a validation candidate. Model selection still uses
specificity at validation recall >=98%, then F1. Only the locked winner is eligible
for a new legacy-test evaluation; if the incumbent wins, its existing result is
retained. This experiment remains exploratory because the legacy benchmark has
already been observed. Run metadata now also records hashes of the training,
model-definition and preprocessing source files.

```bash
python -m src.training.train_clean --model resnet18 --preprocessing clahe --augmentation acquisition --cache-images --epochs 20 --patience 6 --minimum-recall 0.98 --output models/clean_v4/resnet18_acquisition
python -m src.training.train_clean --model efficientnet_v2_s --preprocessing clahe --augmentation acquisition --cache-images --batch-size 8 --lr 0.0003 --epochs 20 --patience 6 --minimum-recall 0.98 --output models/clean_v4/efficientnet_v2_s_acquisition
```

## Dataset integrity details

The fourth experiment completed: the incumbent CLAHE ResNet18 won the validation
comparison, so neither new candidate was evaluated on the legacy test. Independent
RSNA transfer evaluation improved accuracy, precision and F1, but decreased recall.
See [v4 and external results](clean_v4_results.md). The full user objective of a
large improvement across all requested metrics is not yet established.

The original images and previous checkpoint are retained. New artifacts use
`data/processed/clean_v1` and `models/clean_v1`.

Patient identifiers are not supplied in a separately verified patient table.
We therefore use conservative filename proxies:

- All `personN` images belong to one group, including bacteria and virus filenames.
- Repeated `IM-N` acquisitions belong together.
- `NORMAL2-IM-N` has its own namespace.
- Unknown filename conventions fail rather than silently splitting acquisitions.

We join these groups with byte-identical and decoded-pixel-identical images.
Near-duplicate screening joins pairs with 64-bit perceptual hash Hamming distance
at most 6 and normalized 64x64 thumbnail correlation at least 0.995. This rule is
a screening method, not proof that every related acquisition can be identified.

The original 624-image test set is preserved. Development images connected to a
test group are excluded, and redundant decoded images are removed from development
manifests. Raw files are never deleted. The remaining development groups are split
with the fixed first fold of `StratifiedGroupKFold(5, shuffle=True, random_state=20260915)`.
The seed/fold was set before inspecting model performance.

| Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Train | 1078 | 2724 | 3802 |
| Validation | 270 | 681 | 951 |
| Original test | 234 | 390 | 624 |
| Excluded development images | 1 | 478 | 479 |

All three split pairs have zero shared filename proxies, file hashes, decoded
pixel hashes, connected components, and detected near-duplicate pairs. Original
test images can still be correlated within test; confidence intervals use cluster
bootstrap over the conservative groups.

Audit artifacts include the source inventory, duplicate pairs, excluded manifest,
split counts and SHA-256 digests of the generated manifests.

A sensitivity review also searched cross-split pairs with pHash distance <=10
and thumbnail correlation >=0.96. Three pairs were found (correlation
0.9632–0.9643); side-by-side inspection showed differences in acquisition/anatomical
structure rather than an obvious re-encoding or resizing of the same image.
This does not verify patient identity. The review artifacts are
`reports/metrics/near_duplicate_sensitivity.json` and
`reports/figures/near_duplicate_candidates.png`.

## Prespecified experiment design

1. Train ImageNet-initialized ResNet18 and EfficientNet-B0 on the same clean split.
2. Warm up the classification head for two epochs with frozen backbone BatchNorm
   statistics, then fine-tune all layers with a lower backbone learning rate.
3. Use mild affine and brightness/contrast augmentation, no horizontal flipping.
4. Use unweighted cross entropy initially. Tune neither weights nor architecture
   against the legacy test results.
5. Save the checkpoint with minimum unweighted validation cross entropy.
6. Select the operating threshold on validation to maximize specificity subject
   to recall >= 0.99. Recompute in full precision with the serving preprocessing.
7. Compare completed candidates by validation specificity at that recall floor,
   using F1 to break ties. Persist the selected checkpoint hash before test access.
8. Evaluate only the locked winner on the original test and report accuracy,
   precision, recall, specificity, F1, ROC-AUC, PR-AUC and cluster-bootstrap intervals.

The original test is a **legacy benchmark**, not a genuinely untouched new external
test: previous project versions have already inspected it. Its results cannot
establish clinical validity or cross-hospital generalization. Filename proxies also
cannot establish verified patient-disjointness beyond the available metadata.

## Reproduction

```bash
python -m src.data.clean_split
python -m src.training.train_clean --output models/clean_v1/resnet18_unweighted --model resnet18
python -m src.training.train_clean --output models/clean_v1/efficientnet_b0_unweighted --model efficientnet_b0
python -m src.evaluation.evaluate_clean select --experiments models/clean_v1/resnet18_unweighted models/clean_v1/efficientnet_b0_unweighted
python -m src.evaluation.evaluate_clean test
```

Experiment directories cannot be silently overwritten. The selected model stores
its operating threshold and preprocessing metadata for inference.

## External data considered

- Existing source: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
- [RSNA Pneumonia Detection Challenge](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pneumonia-detection-challenge-2018)
  describes NIH images relabeled for pneumonia identification/localization and
  provides source-image mappings and attribution requirements. It is a candidate
  for a separately designed external validation study. It is not mixed into this
  experiment without reconciling labels, source populations and identity mappings.
- The Mendeley source page returned an access challenge during retrieval; no claim
  of patient-identity verification is based on inaccessible content.

Adding more images does not by itself resolve source confounding or label mismatch.
The first controlled comparison therefore uses the cleaned existing dataset.
