# Documentation index

The main usage workflow is in the [repository README](../README.md).
Start with the first three guides below; the other files preserve completed experiments.

## Current guides

- [Short project report](project_report.md): main results, leakage controls, and failed external transfer.
- [Data and weight acquisition / licensing](data_and_weights.md).
- [Fresh installation verification](reproduction_check.md).
- [Presentation outline](presentation_outline.md).

## Main model and development experiments

| Document | Scope |
|---|---|
| [Main v3 / 98% recall comparison](clean_v3_matched98_results.md) | Current local ResNet18/CLAHE model; same 624 test images as the original |
| [Experiment overview](experiment_ledger.md) | Successful/unsuccessful candidates and the limitation of repeated test exposure |
| [Group audit and v1–v4 protocol](clean_training_protocol.md) | Initial clean preparation, candidate recipes, and registered threshold policies |
| [v1](clean_model_results.md), [v1 JSON](clean_model_summary.json), [v2](clean_v2_results.md) | Candidates preceding the main model |
| [v3 / 99% recall](clean_v3_results.md) | Sensitivity analysis under a different recall condition; not the main threshold |
| [v4](clean_v4_results.md), [v5](clean_v5_results.md), [v6](clean_v6_results.md), [v7](clean_v7_results.md) | Later candidates that did not replace the main model |
| [v5 protocol](clean_v5_protocol.md), [RSNA pretraining protocol](rsna_pretraining_protocol.md) | Ensemble and opacity pretraining experiments |

## Additional sources: NIH / MIMIC / OpenI

- [Access inventory](dataset_access_inventory.md): the opening status summary is current;
  timestamped pending items farther down are historical snapshots.
- [Additional dataset candidates](additional_dataset_candidates_20261002.md),
  [Epic duplicate/label review](epic_candidate_review.md),
  [earlier pediatric source research](pediatric_external_data.md).
  The last document does not impose an age restriction on the project.
- NIH: [training protocol](nih_training_protocol.md), [results](nih_v1_results.md),
  [NIH/RSNA target review](nih_rsna_target_review.md).
- MIMIC: [frozen probe](nih_mimic_probe_protocol.md),
  [source head](nih_mimic_source_head_protocol.md),
  [joint metric calibration](nih_joint_metrics_protocol.md),
  [fine-tuning protocol](nih_transfer_protocol.md),
  [fine-tuning results](nih_transfer_results.md).
- OpenI: [comparison protocol](openi_external_protocol.md),
  [completed MIMIC/OpenI results](mimic_openi_results.md).
- RSNA: [external check protocol](rsna_external_protocol.md),
  [results JSON](rsna_external_results.json).

## SSMU source model and failed transfer

- [Development protocol](ssmu_development_protocol.md).
- [Source test protocol](ssmu_test_protocol.md).
- [Model and OpenI results](ssmu_model_results.md).
- [Exploratory OpenI protocol](ssmu_openi_check_protocol.md).
- [Draft patient-group / label clarification questions](ssmu_identity_questions.md);
  this is an unsent draft.

## Analyses preserved from the original version

- [Raw-data EDA summary](eda_summary.md).
- [Original model test/threshold analysis](evaluation_summary.md).
- [Original model Grad-CAM analysis](gradcam_summary.md).
- [Original project report (English PDF)](../Project_Report.pdf).

Historical scores, thresholds, and old pending records do not describe the current model.
Protocol names and paths are retained to preserve registered experiment references.
Large datasets, weights, and detailed local outputs remain outside the repository.
