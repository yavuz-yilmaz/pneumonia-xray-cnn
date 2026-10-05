# NIH / RSNA target review — 2026-10-02

This metadata-only audit checks whether existing development data can supply
cleaner pneumonia targets without altering the task or using observed tests.
No image inference, test-manifest access, relabeling or partition change occurred.

## Source composition

The pinned NIH metadata and existing train/validation manifests passed hashes,
image linkage, report-target equality and original patient-ID checks. The
partitions contain 20,142 and 4,402 distinct NIH patients respectively, with
zero patient overlap. Patient counts differ from duplicate-connected group
counts; the latter are 20,140 and 4,397.

| NIH source target | Train | Validation |
|---|---:|---:|
| Report pneumonia | 700 | 155 |
| No Finding report | 36,894 | 7,897 |
| Other finding without reported pneumonia | 28,068 | 6,160 |
| Total | 65,662 | 14,212 |

Removing other-finding negatives would remove 34,228 development images and
change the population. Such a healthy-versus-pneumonia experiment could not
establish improved pneumonia discrimination from other diseases. A No Finding
report also does not independently establish clinical normality.

## NIH pneumonia versus final RSNA labels

The official unique image/SOP mapping and final `Calculated` annotations passed
hash and linkage checks. Do not use initial RSNA labels as final labels.

| Final RSNA target on NIH report-pneumonia images | Train | Validation |
|---|---:|---:|
| Lung Opacity | 166 | 27 |
| Normal | 86 | 22 |
| No Lung Opacity / Not Normal | 173 | 47 |
| Not in final RSNA labels | 275 | 59 |
| Total | 700 | 155 |

Of the 96 validation pneumonia images with final RSNA labels, 27 have opacity.
Conversely, 608 validation images without NIH-reported pneumonia have RSNA
opacity. **These are different annotation targets, not an error count.** The
audit cannot determine which diagnosis is correct or explain model failures.
Pooling opacity as pneumonia would replace the target rather than clean it.

## Consequence for the requested model

The available data do not supply an automatic, defensible relabeling rule.
NIH's observed final test and OpenI's observed cohort cannot become fresh tests
through filtering or a new split. Preserve patient roles and all exclusions.
SSMU improves source-specific point metrics but lacks verified patient grouping
and fails the exploratory OpenI check. The full requested combination of
verified separation and substantially better metrics remains unproven.

Useful next evidence is an anonymous grouping rule/table from SSMU, or an
accessible new pneumonia cohort with documented patient grouping. The
[SSMU provider draft](ssmu_identity_questions.md) is ready and unsent. Credentialed
PhysioNet sources and nonacademic PadChest permission remain unavailable under
the user's current access. See the [access inventory](dataset_access_inventory.md)
and [candidate review](additional_dataset_candidates_20261002.md).

Local evidence: `reports/data_access_20261002/nih_rsna_targets_v1.json` and
`nih_rsna_targets_v1_review.json`. An independent pandas join reproduced NIH
counts, target equality, patient counts, patient separation and cross-table
totals. Model outputs and serving settings were unchanged.
