# DINOv3 external source check — 2026-10-07

The frozen DINOv3 checkpoint produced **54 false alarms among 78 previously
unused Montgomery normal chest X-rays**. Specificity was **30.77%**
(image-level Wilson 95% interval: **21.63–41.71%**); false-positive rate was
**69.23%** (58.29–78.37%). This source check demonstrates poor transfer to
these normal images. It does not establish performance on pneumonia cases.

The Kermany validation false-positive rate was 6/270 = 2.22% at the same
decision threshold. These are different populations; the comparison is
descriptive, not a matched clinical trial.

A descriptive readback of the saved predictions found that 43 of the 54
false positives had pneumonia scores above 0.9, including 18 above 0.99.
Their median score was 0.9761. These scores are not calibrated clinical
probabilities. This check does not demonstrate that changing the threshold
would repair performance while retaining sensitivity. No threshold selection
or additional inference was performed; the audit is saved as
`reports/metrics/modal_dinov3_external_v1/failure_analysis.json`.

| Source | Acquisition/audit | Scored | Interpretation |
|---|---|---:|---|
| Montgomery County, official NLM originals | All 80 normal frontal PNGs; two conservative near-overlap exclusions | 78 normal images | 24 true negatives, 54 false positives; specificity only |
| SSMU 2026 part 1 | All 50 frontal PNGs matched previously acquired SSMU images byte-for-byte | 0 | Not an unseen project cohort; no external sensitivity result |
| RSUA validated archive | Official cloud download returned HTTP 403 | 0 | No performance result |

The requested evaluation on multiple independent pneumonia-labelled datasets
is therefore **incomplete**. No independent pneumonia recall, overall binary
accuracy, F1 or AUC is reported. Positive and negative examples from unrelated
sites are not pooled into an artificial binary evaluation.

## Frozen model and method

- Training job: `1b73b1facc83fda1`; best validation checkpoint: epoch 4.
- Checkpoint SHA-256:
  `7b67f3404c97de505e38eeedbb062d066980044108e59e0c45e2b1ed7d8ff1cd`.
- Fixed decision threshold: `0.6127732396125793`.
- RGB, resize to 512×512, saved ImageNet normalization, evaluation mode and
  bfloat16 inference; no augmentation, retraining or threshold search.
- External protocol ID: `e68748b9776e1628`. The source/label selection and
  the Montgomery amendment were recorded before any predictions. The amendment
  followed the RSUA delivery failure and added a separate normal-control check.
- Public source metadata and Kermany/all-splits + prior-SSMU fingerprints were
  transferred. Large images and the 1.21 GB checkpoint remained in the cloud.
- Byte/decoded-pixel matches and pHash Hamming distance ≤6 were excluded before
  scoring. The pHash rule is conservative and does not confirm shared patients
  or acquisitions. The two excluded Montgomery cases were
  `MCUCXR_0062_0.png` and `MCUCXR_0099_0.png`.
- All 78 retained original Montgomery PNGs decoded in native 8-bit `L` mode,
  with values within 0–255. The declared 12-bit conversion fallback was unused.

## Sources and limitations

Montgomery images were acquired through the Montgomery County TB Control
Program in Maryland, USA. The [NLM source page](https://lhncbc.nlm.nih.gov/CHRB/CHRB-resources.html)
identifies 80 normal and 58 TB-consistent abnormal images. Its
[readme](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet/NLM-MontgomeryCXRSet-ReadMe.pdf)
defines `_0.png` as normal; only those original `CXR_png` files were used.
TB-positive files, masks and annotated/resized alternatives were excluded.
Dataset citation: Jaeger et al., *Two public chest X-ray datasets for
computer-aided screening of pulmonary diseases*, Quant Imaging Med Surg. 2014;
DOI [10.3978/j.issn.2223-4292.2014.11.20](https://doi.org/10.3978/j.issn.2223-4292.2014.11.20).

[SSMU 2026 part 1](https://zenodo.org/records/18241541), DOI
10.5281/zenodo.18241541, is CC BY 4.0. Although the release was new to the
planned evaluation, all 50 selected files have exactly the same SHA-256 as
images already present in `data/processed/ssmu_images_v1/images.json`.
An independent hash join verified every match. The release date does not
establish image or patient novelty.

[RSUA](https://data.mendeley.com/datasets/2jg8vfdmpm/1), DOI
10.17632/2jg8vfdmpm.1, is CC BY 4.0. Its validated archive reports 53 pneumonia
and 32 normal images after selection based on segmentation acceptability.
The pinned publisher checksum was recorded, but the archive could not be
acquired by the Modal worker. No score is attributed to this source.

This is a small retrospective normal-only check, not clinical validation.
Patient independence and age-specific results are unverified; the confidence
interval assumes independent images. Population, acquisition and label shifts
may explain the failure, but this experiment does not identify their individual
causal contributions. Overlap with the foundation model's pretraining corpus
was not audited. Kermany-only validation is insufficient evidence for deployment.

## Budget, evidence and validation

The external stages reserved **$0.3860**, preserving all prior ledger entries
and the separate $4 reserve. Cumulative reservations are **$23.9713** against
the $24 stage ceiling, leaving **$0.0287**. Reservations are conservative
estimates, not invoices or an invoice cap. No further GPU work was launched;
all three coordinator/external functions had zero running inputs after completion.

CPU preparation took 31.14 seconds; the L4 stage, including acquisition,
screening and inference, took 107.84 seconds. Three additional small cloud
evidence files totalled 225,074 downloaded bytes. No large model/image payload
was downloaded to the local computer.

Individual predictions, frozen inference manifest, source metadata and overlap
matches are under `reports/metrics/modal_dinov3_external_v1/`. The
[machine-readable summary](dinov3_external_results.json) pins their hashes.
Counts and Wilson intervals were independently recomputed from the predictions.
Ten focused external tests and Ruff passed; the existing 26 Modal/training
tests also passed during this work.

The detailed evidence bundle, source images and model checkpoint are not tracked
in Git. The checked-in report preserves aggregate outcomes and evidence hashes;
reproducing case-level predictions requires the source data and authorized
DINOv3 weights or the original checkpoint from the Modal Volume. Dataset
downloads remain subject to their publishers' terms.

`scripts/modal_train_dinov3.py` now exposes `external-prepare` and
`external-test`. They use the same deployed single-writer budget coordinator,
fail closed on incomplete/invalid cohorts, and reuse completed results without
rerunning inference. Required metadata and the reviewed pre-score selection
are preserved in the evidence directory. The completed selection cannot be
changed to obtain a new score from the same evaluation ID.

RSUA scoring remains disabled until its archive layout can be reviewed and
original radiographs distinguished from masks and line overlays.
