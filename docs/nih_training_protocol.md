# NIH pneumonia experiment — prospective protocol

## Purpose and current status

The user permits adult-only development. The target remains the NIH
**report-derived Pneumonia label**. Adjudicated RSNA Lung Opacity is a distinct
target and must not be substituted or relabeled as pneumonia. This experiment
does not establish clinical diagnosis from radiographs.

Official updated NIH metadata is downloaded and inventoried. The image archive
download has started. Image integrity, duplicate auditing, final cohort selection,
training and evaluation are pending. No new model results exist under this protocol.
The existing deployed checkpoint remains unchanged.

## Source and cohort contract

- Use `Data_Entry_2017_v2020.csv` with SHA256
  `c69a6daca3549af707ca9cacbdf9f9a7b6a9188e8c61157df653520bb72d8eb1`.
- Use original NIH patient IDs, checked against image filename prefixes. Every
  acquisition belonging to one patient must stay in one evaluation partition.
- Include images aged 18–120 according to updated NIH metadata. Record exclusions;
  do not trust the older RSNA DICOM age field over this metadata.
- Positive means Pneumonia appears among the report labels. Negatives include
  both No Finding and other pathology labels. The negative class must be named
  **no reported pneumonia**, not NORMAL. Negative reports can contain undetected
  or uncertain pneumonia; this limitation must accompany results.
- Verify listed archive sizes, gzip integrity, image decoding and metadata
  coverage. Record source and decoded-pixel hashes and near-duplicate screening
  using the existing documented pHash/correlation rule. Connect duplicate images
  and original patient IDs into groups before allocating partitions.
- Public NIH PNGs are 1024×1024. The metadata's OriginalImage dimensions describe
  the original acquisition; retain both dimensions without incorrectly treating
  their difference as corruption. Source image previews confirmed this distinction.
- NIH and RSNA share images and patients. They do not constitute independent
  institutional evaluation sources.

## Prior exposure and new partition requirements

The inventories record 4,021 patients from previous RSNA experiments. The
1,000 external patients remain reserved from all training and calibration.
The old 3,021 development patients were used for feature pretraining and
validation; their prior roles must be preserved if they are reused. Previously
trained patients may not enter validation or a new untouched test; previously
validated patients may not enter a new untouched test.

Only patients absent from all recorded RSNA cohorts can enter a new final test.
There are 531 adult report-pneumonia patients absent from these cohorts before
image duplicate screening; this is an eligibility count, not a final test size.
Any duplicate component touching a reserved or previously exposed patient must
inherit that restriction. Conflicts between historical roles must be excluded
and reported rather than resolved by moving images into a fresh holdout.
Exact decoded images with conflicting report-pneumonia labels are excluded with
their connected patient group. Components touching previously observed pediatric
source images are excluded as well. These conservative exclusions must be recorded.

For eligible new groups, fix seed `20261001` and use reproducible patient-group
stratification for approximately 70% training, 15% validation and 15% test, using
whether the group contains any positive acquisition as the stratification label.
Persist image membership, exclusions, source hashes and overlap audit before
training. Do not overwrite finalized partitions to improve class counts or scores.

Validation and test must retain the selected source cohort's class prevalence.
Training may use class weighting or balanced sampling. Do not balance the test
and then interpret its precision as precision in the original cohort.

## Baseline, candidates and evaluation

1. Evaluate the original and current deployed pediatric classifiers on the same
   new test cases only after the candidate selection is locked. These are transfer
   baselines; their old pediatric scores are not directly comparable with NIH scores.
2. Train ImageNet-initialized ResNet18 and EfficientNetV2-S as adult candidates,
   using existing training conventions for augmentation, AMP, early stopping and
   immutable run directories. Explicitly record their task labels and preprocessing.
   Use 16,000 balanced training samples per epoch with replacement; retain natural
   validation/test prevalence. Do not combine balanced sampling with weighted loss.
   RSNA-transferred and XRV NIH-pretrained initialization are outside these initial
   candidates. Use CLAHE at 224px for ResNet18 and resize at 320px for EfficientNetV2-S.
3. Select architecture and checkpoint by validation PR-AUC, with validation loss
   as a deterministic tie-breaker. Select the decision threshold on validation by
   maximum F1 subject to recall >=90%. If this constraint cannot be met, report that
   fact; do not lower it after observing test results.
4. Lock checkpoint hashes, thresholds and baseline versions before final test
   scoring. Score the final test once. Further experiments after inspecting it
   require a new genuinely eligible test cohort or must be described as exploratory.
5. Report accuracy, precision, recall, F1, specificity, ROC-AUC, PR-AUC, confusion
   counts, prevalence and patient-cluster bootstrap confidence intervals. Use
   paired patient resampling for baseline differences. A point-estimate increase
   alone does not establish a substantial improvement across all requested metrics.

Success remains unproven until patient-disjoint evaluation shows the requested
improvements on a fixed, meaningful task. Access to cleaner adjudicated pneumonia
labels from VinDr-CXR or another institution remains valuable if NIH label quality
limits performance. Passing acquisition tests is not evidence of model improvement.

## Reproducing the current acquisition step

The official NIH image folder lists the metadata and publisher's downloader.
The reviewed public URLs and archive sizes are saved in
`configs/nih_download_manifest.json`; no patient records are included. Download
the official metadata before running the inventory, then use this manifest
without executing publisher code:

```powershell
python -m src.data.inspect_nih
python -m src.data.download_nih --manifest configs/nih_download_manifest.json
python -m src.data.extract_nih --manifest configs/nih_download_manifest.json --watch
python -m src.data.prepare_nih_split --manifest configs/nih_download_manifest.json
```

After all source archives and the global split audit have passed, the registered
candidate commands are:

```powershell
python -m src.training.train_clean --task nih_report_pneumonia --manifests data/processed/nih_pneumonia_v1 --output models/nih_v1/resnet18_clahe --model resnet18 --preprocessing clahe --cache-images --size 224 --batch-size 24 --epochs 22 --minimum-recall 0.90 --balanced-sampling --epoch-samples 16000
python -m src.training.train_clean --task nih_report_pneumonia --manifests data/processed/nih_pneumonia_v1 --output models/nih_v1/efficientnet_v2_s --model efficientnet_v2_s --preprocessing resize --cache-images --size 320 --batch-size 8 --epochs 22 --minimum-recall 0.90 --balanced-sampling --epoch-samples 16000
python -m src.evaluation.evaluate_nih select --experiments models/nih_v1/resnet18_clahe models/nih_v1/efficientnet_v2_s
python -m src.evaluation.evaluate_nih test
```

These commands are registered plans, not completed runs. NIH checkpoint labels
remain distinct from the existing deployed API contract; do not point that API at
a NIH checkpoint and describe its negative probability as a normal probability.

The NIH evaluator reads both original-task and NIH-task checkpoints with explicit
metadata and uses the NIH report labels for the comparison. It pins each baseline's
existing threshold, model hashes, test manifest hash and evaluation implementation
before scoring. A test-start marker prevents accidental repeat selection/evaluation;
an interrupted final scoring run requires review rather than silently starting a
second test. Paired patient/duplicate-cluster confidence intervals include PR-AUC
and selected-minus-baseline differences.

The local acquisition state, patient inventory, archives and image files remain Git-ignored.
Downloaded data is not relicensed by this repository's software license.
