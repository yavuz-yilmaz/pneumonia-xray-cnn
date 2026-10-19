# DINOv3 on Modal — $28 experiment

This completed research workflow trained **DINOv3 ViT-L/16 (~300M parameters)**
with a binary head. It keeps the main ResNet18 checkpoint, configuration, API
and demo unchanged. No notebook or large local dataset/model upload is required.
Running the cloud commands below can incur charges; the budget ledger is an
estimate and is not a provider invoice cap.

## Budget and scope

`configs/modal_dinov3.json` registers one 512px, seed-42 recipe: two head-warmup
epochs, then adaptation of the last four transformer blocks and final norm,
up to 16 epochs, early stopping on full-validation loss, and a validation-only
threshold maximizing specificity with recall >=98%. Training/pilot do not score
test data. The separately frozen [external source check](dinov3_external_results.md)
records poor Montgomery specificity, excluded SSMU overlap and unavailable RSUA data.

Using Modal's **2026-10-06** published base rates, the initial worst-case time
reservations are approximately **$14.51**: CPU preparation (1 hour), GPU pilot
(3 minutes), GPU training (4 hours 40 minutes), startup/idle allowances, CPU/RAM and a
coordinator allowance. **$4 is held separately** for builds and costs outside
these estimates. The ledger refuses dispatch above $24 cumulative reservations;
failed and interrupted stages retain their full reservation. There are no
automatic retries. This is deliberately conservative, not actual billing.
The $14.51 plan assumes `all` in one coordinator invocation; separate `pilot`
and `train` invocations reserve another $0.50, for approximately **$15.01**.

**Set the Modal workspace budget to $28 before running jobs.** That caps total
monthly usage before credits. Also set a $28 spend limit if you want a separate
cap on out-of-pocket charges after credits. The code's
ledger cannot cap the provider invoice, other applications in the workspace,
rate changes, regional premiums, or costs beyond its resource estimates. Use
base pricing (no explicit region or non-preemptible surcharge). Use stable Modal
app and Volume names across runs; do not delete/reset the ledger. The defaults
are `pneumonia-dinov3` and `pneumonia-dinov3-data`. Override them with
`MODAL_DINOV3_APP_NAME` and `MODAL_DINOV3_VOLUME_NAME` if you need separate
resources. Review the workspace's existing usage and credit balance yourself.

Sources: [pricing](https://modal.com/pricing),
[GPU options](https://modal.com/docs/guide/gpu),
[workspace budgets](https://modal.com/docs/guide/budgets).

## Local checks — no authentication, network or charges

Run from the repository root:

```powershell
python scripts/plan_modal_dinov3.py --check
```

Only the existing `train_manifest.csv`, `val_manifest.csv` and
`split_audit.json` are serialized (~1.1 MB in the current workspace). Neither
raw images, test CSV, trained weights, `.venv`, credentials nor the repository's
large artifacts are uploaded. Only selected source code and one small config
are mounted. Missing or modified audited manifests stop the process.

## Account and model access

1. Modal CLI **1.4.2** is the supported client. If missing, install only
   `python -m pip install modal==1.4.2` locally; all ML dependencies install on
   Modal. `pip install -e ".[cloud]"` is optional and can download more locally.
2. Authenticate yourself with `python -m modal setup` if needed. GPU use requires
   a valid payment method. Do not put tokens in Git, commands or chat.
3. Accept Meta's DINOv3 access/license conditions on
   [the pinned model's page](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m).
   Wait for access approval. Create a Hugging Face read token from that same
   approved account, permitted to download this gated model. For fine-grained
   tokens, enable access to the public gated repositories you can access.
4. In Modal's **Secrets** dashboard, create `pneumonia-dinov3-access` containing
   `HF_TOKEN`. If Kaggle's public route fails, add `KAGGLE_USERNAME` and
   `KAGGLE_KEY` there. Never paste credentials into project files.

The 2026-10-06 cloud preparation downloaded the ZIP and passed the train/val
image checks, then stopped because the supplied Hugging Face account lacked
DINOv3 access. A later download can still fail due to source availability,
access restrictions or changed files.
Exact image SHA-256 verification against the current audited manifests is
mandatory; the code will not silently train on a different distribution.

## Deploy once, then use the deployed coordinator

```powershell
python -m modal deploy scripts/modal_train_dinov3.py
python -m modal run scripts/modal_train_dinov3.py --action access
```

Continue only when the access check prints `"access_granted": true`. A
`gated_model` result means the approved account and token permissions need to
be checked; update `HF_TOKEN` in the existing Modal Secret after approval. This
HEAD-only check downloads no dataset or weights, uses no GPU and does not make
coordinator/preparation reservations. Its small bounded CPU charge falls within
the separate reserve; it is not a free cloud operation. Preparation also checks
access before hashing/extracting the archive and emits a short sanitized error.
Failed preparation attempts still retain their original reservations.

```powershell
python -m modal run --detach scripts/modal_train_dinov3.py --action pilot
```

`pilot` first downloads/prepares on CPU if needed, then measures six training
batches after three warmup batches, plus six validation batches after two
warmup batches, on **A100-80GB**. It reports
time and peak allocated VRAM, not a partial-cohort model quality score. Data and
revision-pinned safetensors are kept in the configured Modal Volume.
Only matching train/val archive members are extracted; test images are not.
The Kaggle ZIP mirrors images under `chest_xray/` and
`chest_xray/chest_xray/`. Every copy must match the source manifest's SHA-256;
byte-identical copies yield one extracted image. Conflicting copies still fail
preparation, and the 30 MB per-member limit remains enforced. If an earlier
attempt failed here, redeploy the fixed source before retrying `pilot`. A ZIP
already saved in the Volume is reused, and prior budget reservations are kept.

The first GPU pilot reached its 20-minute limit without a stage log. A diagnostic
retry confirmed a wait inside `verify_manifests` opening individual images from
the Volume and was cancelled. CPU preparation now bundles the audited train/val
bytes from the ZIP into `development.tar`. GPU containers stream this bundle and
the pinned weight files to their own temporary disk, checking preparation hashes,
then perform the original per-image audit locally. Only paths are rebased; labels,
groups and image bytes remain the same. Checkpoints still persist to the Volume.
No dataset or weights are transferred to the user's computer. A cloud pilot
then completed in about 85 seconds with 3.27 GiB peak allocated GPU memory.

The launcher saves `pilot_progress.json` at each stage and prints a stack trace
every two minutes during a pilot. Data workers start with `spawn` and wait at most
120 seconds for a batch. The forecast includes staging, setup and batch loading.
Timing schema 3 measures consecutive batches after both image workers have warmed
up, allowing the prefetch queue to drain at actual training speed. It adds warmup
overhead once per epoch and keeps diagnostic commits outside measured batches.
Earlier reports are recalibrated through
the same budgeted coordinator; their reservations are retained.
Training's timeout was reduced from six hours to four hours forty minutes to retain
room for the final calibration pilot within the $28 budget; the 16-epoch recipe is
unchanged and admission still requires the forecast to fit. The timing-schema-3
cloud pilot completed successfully: setup 29.50 seconds, conservative epoch
estimate 221.14 seconds, total 16-epoch forecast 3,567.79 seconds (about 59.5 minutes),
and peak allocated GPU memory 3.27 GiB. This short sample is a throughput estimate;
the completed training results are recorded below.
Before training, the cumulative ledger reserved $9.9282. The separate training
invocation reserved another $13.6571 including its coordinator, bringing the
cumulative total to $23.5853 and keeping the separate $4 reserve intact. These
reservations are not the actual invoice.

Inspect the pilot and ledger:

```powershell
python -m modal run scripts/modal_train_dinov3.py --action status
```

On Windows, the launcher uses the operating system's DNS resolver for Modal's
HTTP uploads. This avoids `Could not contact DNS servers` errors from optional
`aiodns`/c-ares installations when Windows itself can resolve the upload host;
it changes neither network settings nor installed packages. An error inside
`coordinator.spawn` while uploading its arguments occurs before the coordinator
is dispatched, so that invocation has not yet started preparation or the pilot.
Build/storage charges and any other running jobs must still be checked in Modal.

Run the registered training after reviewing the pilot:

```powershell
python -m modal run --detach scripts/modal_train_dinov3.py --action train
```

`--action all` combines preparation, pilot and training. The coordinator refuses
training if the pilot forecasts the remaining epochs cannot fit in the training
window. It also refuses any stage whose full reservation exceeds the remaining
allowance. Always use the deployed coordinator; directly invoking internal
functions would bypass the ledger. Do not redeploy during a running experiment.

The detached call survives local terminal closure. At each complete epoch,
model, optimizer, history and protocol are saved atomically and committed to
the Volume. An interrupted epoch is replayed from the previous complete epoch
with its deterministic epoch seed. The model never resumes from a partial
epoch. Protocol/source/manifest changes invalidate resume. Time-limited results
explicitly report incomplete training; an incomplete validation pass cannot
select a checkpoint. Reservations do not automatically renew: another training
attempt may be refused even if the previous attempt finished early.

Matching seeds do not guarantee bitwise identical GPU results: SDPA kernels can
be nondeterministic. Model selection, full-cohort validation and budget behavior
are tested locally. The cloud pilot and subsequent training completed; the saved
validation predictions reproduce the selected epoch's recorded metrics.

If a resume needs a shorter reservation, reduce `train_timeout_seconds` in the
config and redeploy **after the old job ends**. This changes the time window,
not the registered recipe. Changing epochs/size/LR creates a distinct recipe.
Keep the same ledger, and inspect remaining allowance before any new dispatch.

## Outputs and inference

The 2026-10-06 run `1b73b1facc83fda1` finished with `early_stopped` after nine
epochs and 1,175.60 seconds of training-function time (about 19.6 minutes). The
lowest validation loss was 0.04289475 at epoch 4; five subsequent epochs did not
improve it. The saved checkpoint therefore represents epoch 4, even though some
later epochs had higher threshold-dependent accuracy.

On the 951-image validation split, the selected checkpoint achieved:

| Metric | Value |
| --- | --- |
| Accuracy | 98.32% |
| Precision | 99.11% |
| Recall | 98.53% |
| Specificity | 97.78% |
| F1 | 98.82% |
| ROC AUC | 0.99879 |

At the validation-selected threshold 0.61277324, there were 264 true negatives,
6 false positives, 10 false negatives and 671 true positives. No test scoring
occurred. Source hashes, early-stopping consistency and metrics recomputed from
the saved predictions were checked. Small reports and a summary are stored in
`reports/metrics/modal_dinov3/1b73b1facc83fda1/` locally (22,433 bytes downloaded).
The 1.21 GB best checkpoint and 1.62 GB resume checkpoint remain in the Volume;
neither was downloaded. The GPU function has stopped.

Status returns only small reports, including the last saved pilot stage.
Full artifacts stay under `/runs/<job_id>/`:
`protocol.json`, `history.json`, `pilot.json`, `pilot_progress.json`, `result.json`,
`validation_predictions.npz`, `best_model.pt` and resume-only `last.pt`.
Do not download these large checkpoints on the current limited connection.
`best_model.pt` includes the backbone configuration and all model tensors, so
later inference requires no pretrained-weight download.

When bandwidth permits, download `best_model.pt` from the Volume dashboard into
an ignored `models/` directory. With the optional Transformers dependency:

```powershell
python scripts/predict_dinov3.py --model models/modal_dinov3/best_model.pt --image path/to/image.jpeg
```

This dedicated inference command supports the new checkpoint format. The main
API/demo's historical checkpoint loader does not accept it; deployment is a
separate step after validating the candidate. The DINOv3 license applies to its
weights/derivatives separately from the project code's MIT license.

Scores from this workflow are development results, not an independent final
test or evidence of clinical reliability. Existing historical test exposure
and filename-derived patient-group limitations remain disclosed. A gain on
validation does not establish generalization to other hospitals or labels.
