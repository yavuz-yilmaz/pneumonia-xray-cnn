# SSMU model results — completed 2026-10-02

The new frozen-feature classifier has **higher test point estimates in all four
requested metrics than both existing models on the same SSMU cohort**. Accuracy,
recall and F1 gains have positive paired uncertainty intervals. Precision's
interval includes zero, and patient identities remain unverified. This is useful
source-specific improvement, not proof that the full leakage-free/generalization
goal has been achieved. The deployed model remains unchanged.

## Data and separation

The CC BY 4.0 source archive contains 1,837 decoded PNGs. Its publisher checksum
and gzip CRC passed; the conservative NIH/legacy/OpenI screen found no match.
Use only original 8-bit frontal filename-hint images. Remove five redundant
frontal images; exclude all lateral images, including the six 16-bit files whose
earlier L fingerprints were clipped.

Group equal numeric stems across **both classes and both views**, then join all
eleven recorded exact/near duplicate pairs, including lateral pairs. This avoids
assuming independent numbering namespaces. A fixed seven-fold rule reserves
665 train / 133 validation / 131 test images. Known byte, pixel, numeric-proxy,
connected-component and recorded near-match split overlap is zero.

These are conservative **filename proxies**, not verified patient identifiers.
The archive contains no patient/age/annotation-method table. Other visits under
different numbers could remain undetected. Source folders supply normal/pneumonia
labels; clinical adjudication and adult-only coverage are not established.

Subsequent inspection of all 929 selected files found no PNG information keys
or EXIF tags after source-byte verification. Six training-only visual samples
did not show apparent identifying text; this does not exclude text in other
images. Embedded metadata therefore did not establish patient grouping. An
[anonymous grouping/label clarification draft](ssmu_identity_questions.md)
is prepared and remains unsent.

## Registered training and one-time test

The [development protocol](ssmu_development_protocol.md) uses separately verified
MIMIC-only DenseNet121 features. Training-only StandardScaler and four fixed L2
logistic heads were fitted; validation AP selected C=0.01. All feature tensors
remain unchanged. A threshold satisfying both original/current validation metric
floors was selected without test access: **0.7265916466712952**.

The [test protocol](ssmu_test_protocol.md) then froze that model and threshold,
the test manifest, both baseline checkpoints and evaluator source hashes before
inference. It compared only the selected model and the two fixed baselines,
using serving preprocessing and float32 predictions. No test-driven threshold
or candidate changes occurred. The cohort has **131 images, 72 positives and
71 conservative proxy groups**. It is now observed and must remain reserved.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Original classifier | 64.89% | 90.63% | 40.28% | 55.77% | 0.8206 | 0.8518 |
| Current deployed classifier | 48.85% | 69.23% | 12.50% | 21.18% | 0.8249 | 0.7952 |
| New SSMU classifier | **86.26%** | **95.00%** | **79.17%** | **86.36%** | **0.9353** | **0.9560** |

The new classifier has 57 true positives, 15 false negatives, three false
positives and 56 true negatives. Original has 29/43/3/56 respectively; current
has 9/63/4/55. Against current, the point gains are +37.40 accuracy, +25.77
precision, +66.67 recall and +65.19 F1 **percentage points**.

Baseline thresholds are their existing saved operating points (0.7 original,
0.8820025324821472 current), whereas the new model's threshold was calibrated
on SSMU validation. Thus improvement includes source-specific head fitting and
calibration; it is not an architecture-only comparison. Positive ROC-AUC/AP
difference intervals also support better ranking on this source. These scores
must not be compared directly with the old pediatric benchmark's scores as a
before/after improvement; all rows above use the same new source cohort.

## Uncertainty

One thousand paired bootstrap repetitions resampled the same proxy/duplicate
groups for every model. These are **proxy-group**, not verified patient-level,
intervals. Selected-minus-baseline 95% intervals, in percentage points:

| Metric | Versus original | Versus current |
|---|---:|---:|
| Accuracy | +12.86 to +30.47 | +30.15 to +45.19 |
| Precision | **−6.78 to +17.74** | **−0.55 to +55.99** |
| Recall | +26.03 to +53.42 | +56.15 to +77.79 |
| F1 | +18.84 to +44.89 | +53.67 to +77.17 |

Do not claim statistically established improvement in precision: its paired
intervals include zero. The selected recall interval is 69.44–87.68%; 15/72
source-pneumonia images were missed. Small sample size and source-specific
selection limit the practical claim.

## Artifacts and verification

The usable checkpoint is
`models/ssmu_probe_v1/selected_joint/best_model.pt`, SHA256
`6e8202040581f8e8412fea206f66e1eb748471e0161fff3d4cc56396b237fbc6`.
It uses existing NORMAL/PNEUMONIA mapping and serving preprocessing. To use it
explicitly with an image, run:

```powershell
python scripts/predict_image.py --model models/ssmu_probe_v1/selected_joint/best_model.pt --image path/to/frontal_xray.png
```

Twenty-five focused tests passed in two batches: conservative cross-class and
lateral-match grouping, actual logistic fit/export, training-only scaling,
test-manifest-free fitting, infeasible threshold retention, one-time evaluator
guards, source drift rejection and existing paired-evaluation/threshold checks.
Ruff passed. Before test scoring, saved validation metrics/selection and unchanged
MIMIC feature tensors were verified. After scoring, an independent readback
confirmed all three prediction/label/group/member orders, checkpoint/source
hashes, frozen export tensor equality, thresholds, confusion matrices and all
four primary metrics without new image inference.

Local evidence: `models/ssmu_probe_v1/{protocol,result}.json`,
`models/ssmu_probe_v1/selected_joint/selection.json`, and
`reports/metrics/ssmu_probe_v1/{registration,test_metrics,artifact_review}.json`.
Raw images, weights and private evidence remain ignored by Git. No commits,
provider messages, public release or automatic deployment were performed.

Remaining work for the full goal: stronger identity/label provenance, a larger
defensible evaluation source for precision and generalization, and verification
that a replacement supports the intended application beyond this source.

## Exploratory OpenI check — completed 2026-10-02

The frozen SSMU checkpoint and its unchanged threshold were checked against
2,892 existing OpenI cases (32 positive), averaging unique frontal-image scores
per case. The [separate protocol](ssmu_openi_check_protocol.md) and input/source
hashes were recorded before inference. Original/current probabilities were
reused after alignment, hash, aggregation and metric verification. Only the
SSMU model ran new image inference. OpenI had already been observed for five
earlier models before SSMU development, so this is **exploratory**, not an
untouched final test or an amendment to the earlier studies. No OpenI tuning
occurred.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|
| Original | 64.28% | 1.36% | 43.75% | 2.64% | 0.5724 | 0.0169 |
| Current | 78.49% | 1.32% | 25.00% | 2.51% | 0.6031 | 0.0154 |
| Frozen SSMU | **47.61%** | **1.44%** | **68.75%** | **2.82%** | **0.6276** | **0.0204** |

SSMU produced 22 true positives, 10 false negatives, **1,505 false positives**
and 1,355 true negatives. It does not improve all four metrics: accuracy falls
substantially, and precision/F1 remain very low. These outcomes do not support
replacing the deployed model for general use. OpenI targets are report-indexed
pneumonia; a negative is not confirmed normal lungs, so this differs from the
SSMU source-folder task. The source shift, small positive count and previously
observed cohort limit interpretation; none erases the observed failure.

Paired case bootstrap accuracy differences are negative against both baselines:
95% intervals are −18.74 to −14.63 percentage points against original and
−32.92 to −28.77 against current. Precision and F1 difference intervals include
zero against both. Recall improves against current (interval +22.86 to +65.39
points); the interval against original touches zero. Ranking improvements also
have intervals spanning zero. Small point gains in precision/F1 therefore do
not establish better performance.

Twenty-five focused tests (new exploratory evaluator and existing OpenI
evaluator) passed. Ruff passed after import/format repairs. The completed
registration, predictions, paired case bootstrap intervals and result are in
`reports/metrics/ssmu_openi_check_v1/`. Model and threshold remain unchanged,
and these outcomes must not drive subsequent tuning on either reserved source.
Independent saved-output readback verified model/source hashes, all case/label
orders, image-to-case aggregation and confusion-derived primary metrics for
each model without new image inference. Its passed record is `artifact_review.json`.
