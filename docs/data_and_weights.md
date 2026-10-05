# Data, pretrained weights, and licenses

The main workflow trains ResNet18/CLAHE on Kermany data. NIH, SSMU, and OpenI
are not required to run the main demo. **This page does not initiate downloads.**
Access/experiment records are dated October 2, 2026; Kermany, NIH, and RSNA
source descriptions were rechecked through their publisher pages on October 5, 2026.

## Main dataset: Kermany Chest X-Ray Images (Pneumonia)

Source: Daniel Kermany, Kang Zhang, Michael Goldbaum,
*Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*,
Mendeley Data, **version 2**, DOI **10.17632/rscbjbr9sj.2**.

- [Original Mendeley record](https://data.mendeley.com/datasets/rscbjbr9sj/2)
- [Kaggle distribution with the train/val/test directory layout used here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [Publisher-stated CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/)

The Mendeley record contains both OCT and chest X-rays; this project uses the
**chest_xray** subset. Kaggle access may require an account; the project does
not request a Kaggle API key. Acquire the data manually and extract it into
`data/raw/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}`. Avoid including
`__MACOSX` or an extra `chest_xray` directory level in the data root.
The reference distribution contains 5,856 images; retain the original filenames.

For CC BY 4.0, credit the authors/source, link the license, and describe changes.
This project resizes images, applies CLAHE, and generates new group-audited
manifests. Also cite Kermany et al.,
*Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*,
Cell (2018), [DOI 10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010).
Raw images are not distributed in this repository.

## ResNet18 pretrained weights

The main training code uses `torchvision` **ResNet18_Weights.DEFAULT / IMAGENET1K_V1**.
The [official weight record](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
identifies `resnet18-f37072fd.pth`: **46,830,571 bytes (about 45 MiB)**.
It is downloaded automatically on the first training run if absent from the local Torch cache.

- Default cache: `~/.cache/torch/hub/checkpoints/`.
- With `TORCH_HOME` set: `<TORCH_HOME>/hub/checkpoints/`.
- Place an existing file there to avoid downloading it again.
- Inference loads all weights from the trained checkpoint; it does not separately
  download ImageNet weights.

Review the [torchvision source license](https://github.com/pytorch/vision/blob/main/LICENSE)
and the publisher's [model licensing notes](https://docs.pytorch.org/vision/stable/models.html)
separately. Third-party model/data terms are not replaced by the project's MIT
license. Installing PyTorch packages, especially CUDA builds, can require much
larger downloads than the weight file. Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
for GPU setup; prefer an existing environment and cache on a limited connection.

## The project's trained checkpoint

The public repository contains no ready-made `best_model.pt` or published model
download link. The [main training/selection workflow](../README.md) produces
`models/clean_runs/selected/best_model.pt`.

Historical local main model:

```text
models/clean_v3/selected_matched98/best_model.pt
SHA256: da5c58d36865cc01d8df1dce5c358dac2a06b0b0b4981115bb7e35d806e2be9b
```

In PowerShell, verify it with `Get-FileHash -Algorithm SHA256 <checkpoint-path>`.
This hash identifies the historical file; a newly trained checkpoint is not
expected to have the same hash. The checkpoint stores input size, normalization,
preprocessing, threshold, and training protocol. Load only PyTorch checkpoints
you produced yourself or obtained from a trusted source.

## Other experiment sources

This table describes acquisition and attribution. Large archives are outside
the main workflow; detailed acceptance, auditing, and access records are in the
[inventory](dataset_access_inventory.md).

| Source | Acquisition / attribution | License and use limitations | Status in this project |
|---|---|---|---|
| NIH ChestX-ray14 | [Official NIH listing](https://nihcc.app.box.com/v/ChestXray-NIHCC), [access and citation terms](https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest) | The publisher asks for NIH Clinical Center credit, a download-page link, and a Wang et al., CVPR 2017 citation; no additional CC BY designation was assigned here. The Google Cloud alternative uses Requester Pays. | 112,120 images, about 45.08 GB of archives acquired; training/experiments completed. |
| RSNA 2018 | [Official RSNA release and terms](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018); Shih et al., Radiology: AI (2019), DOI 10.1148/ryai.2019180041 | The official description permits research/education and commercial/noncommercial use with attribution. Opacity labels are not clinical pneumonia diagnoses; the source overlaps with NIH. | Separate opacity experiments; not presented as adult-only or independent of NIH. |
| SSMU 2021 | [Version v1 / DOI 10.5281/zenodo.5732746](https://doi.org/10.5281/zenodo.5732746); Udodov, Kurazhov, Zorkaltsev, Zavadovskaya | Recorded publisher metadata states CC BY 4.0. Patient identities and label provenance are unverified. | A 4,185,026,327-byte archive and 1,837 images acquired; source model and external check completed. |
| OpenI / Indiana | [NLM OpenI](https://openi.nlm.nih.gov/), [source publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC5009925/); Demner-Fushman et al. (2016) | Official XML records reviewed state CC BY-NC-ND 4.0. Permission to evaluate is not interpreted as permission to redistribute or share modified images. | 7,470 images / about 1.36 GB of archives acquired; report-coded external evaluation completed. Images are absent from the repository. |
| MIMIC-only pretrained features | [TorchXRayVision author repository](https://github.com/mlmed/torchxrayvision), [official MIMIC-CXR-JPG record](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) | Access to publicly available author weights does not grant access/approval for raw MIMIC data; weight and data terms are separate. | Author-provided MIMIC-only weights used; raw MIMIC images were not downloaded. `.[xray]` is needed only for those experiments. |

PadChest-pneumonia was not acquired because publisher approval required for a
personal portfolio was unavailable. This project lacks the authorization needed
for raw VinDr/MIMIC/BRAX access. BDCXR had conflicting license texts; Epic had
duplicate/label issues. Neither was used as model data.
See the [candidate reviews](additional_dataset_candidates_20261002.md)
and [Epic review](epic_candidate_review.md).

## Publication scope

The [MIT license](../LICENSE) covers project code. It does not replace source
dataset licenses, and trained weights are not automatically published under MIT.
Raw data, processed manifests, checkpoints, and private research evidence are
excluded through `.gitignore`. Public documentation contains aggregate results
and source citations.
