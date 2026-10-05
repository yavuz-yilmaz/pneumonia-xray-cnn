# SSMU frontal development experiment — registered 2026-10-02

## Purpose and limits

Evaluate whether a newly acquired normal/pneumonia source supports a better
classifier than the original and current models on the **same source cohort**.
This does not replace the full improvement goal with a source-specific success.
The archive lacks verified patient IDs, ages and an annotation methodology.
Filename groups are conservative proxies; patient independence and clinical
ground truth remain unverified. Do not call this experiment clinically validated
or fully leakage-free. No automatic deployment or publication is authorized.

The CC BY 4.0 publisher archive passed MD5, SHA256, gzip CRC and all 1,837 PNG
decodes. Full NIH/legacy/OpenI screening recorded no cross-source match under
its conservative rule. Six 16-bit lateral images had clipped L fingerprints;
they are excluded from model inputs. All 934 frontal filename-hint images are
original 8-bit L. PA/lat names are source hints, not independent view verification.

## Fixed data rule, before model scores

- Use only `dataset/images/{pneumonia,norma}/N_pa.png`, source L mode.
  Interpret `norma` as NORMAL and `pneumonia` as PNEUMONIA, preserving source
  directory provenance. This is not relabeling NIH report negatives as normal.
- Across the **entire archive**, join equal numeric stems regardless of class
  or view. Also join every recorded exact/near duplicate pair, including lateral
  pairs. This deliberately over-groups possibly different people rather than
  assuming that the two numeric namespaces are independent.
- Remove redundant frontal exact/near matches by keeping the lexicographically
  first member of each image-match component. All other frontal images remain.
- Use shuffled StratifiedGroupKFold with seven folds, seed 20261002. First fold
  is reserved test, second validation, remaining five training. Choose no folds
  based on metrics. Preserve all lateral/redundant images as excluded metadata.
- Audit byte, standardized-pixel, numeric-proxy and connected-component overlap,
  and all recorded near pairs. Persist immutable manifests and source hashes.
  These checks address documented overlap; they cannot verify unknown identities.

## Fixed first model experiment

Use the separately hash-verified MIMIC-only DenseNet121 feature artifact.
No SSMU test images enter feature extraction, fitting, scaling, checkpoint or
threshold selection. Extract float32 features from training and validation only,
using the existing 224px resize and XRV normalization. Fit a StandardScaler on
training only, then four L2 logistic heads, C = 0.001, 0.01, 0.1, 1.0. Select
highest validation average precision, lowest C on ties. Fold scaling into the
binary PyTorch head and preserve all candidate predictions and checkpoints.

Evaluate the original and current checkpoints only on validation, at their
existing saved thresholds and serving preprocessing. After AP selection, choose
the candidate threshold that meets all four validation metric floors of both
baselines; maximize F1, then accuracy, then threshold. If none exists, retain
failure without relaxing floors or searching additional C values on this run.
Keep a separate descriptive F1 threshold with recall >=90%.

No test inference occurs until the selected candidate and operating threshold
are frozen, and the validation joint requirement is feasible. Register any
one-time comparison separately before test inference. Same-cohort improvement
must be shown across all four metrics against both baselines, with uncertainty
and identity limitations disclosed. It cannot establish a better result on the
previous cohort or satisfy the full goal by itself.
