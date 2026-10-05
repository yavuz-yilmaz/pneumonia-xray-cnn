# Fresh installation and main workflow verification

Verification date: **October 5, 2026**. Result: **passed**.
The [machine-readable record](reproduction_check.json) contains package versions,
commands, and SHA256 hashes of the source files tested in that run.
This records the installation before subsequent language edits; those edits
change source hashes without changing the historical verification record.

## Environment and scope

- **Python 3.12.10**, Windows, RTX 3060 Laptop GPU; torch **2.11.0+cu128**,
  torchvision **0.26.0+cu128**, NumPy **2.2.6**, Pillow **11.3.0**.
- A fresh `venv` was created without access to system packages.
- Code/config/tests/documentation were copied to a new source directory.
  Initially, it contained no data, models, reports, `.git`, or other virtual environment.
- The project was installed editable with `.[dev]`. Imports were verified to
  resolve to this copy, whose source hashes matched the working tree at that time.
- Packages were installed **only from the local wheel cache** using
  `--no-index --find-links`. ResNet18 ImageNet weights also came from the
  existing Torch cache. External DNS/network access was blocked during runtime
  checks; the local API was allowed.

This run downloaded no new datasets, packages, or weights. Small publisher
pages were checked while preparing the licensing guide. A reader with a fresh
clone and no local cache must acquire packages, data, and initial weights separately.

## Checks performed

| Check | Result |
|---|---|
| `python -m pip check` | No dependency conflicts |
| `scripts/check_dataset_ready.py` and config-driven `standardize_dataset` | Synthetic raw data prepared; 32 train / 8 val / 8 test, zero known group/duplicate intersections |
| `train_clean` / ResNet18 / CLAHE / 224px | 3 epochs; two warmup epochs and one fine-tuning epoch completed |
| `evaluate_clean select` / recall condition 0.98 | Model and threshold selected on validation and saved to the checkpoint |
| Test manifest unavailable during training/selection | Both steps completed successfully |
| `evaluate_clean test` | Command/output check on only 8 synthetic test images |
| `scripts/predict_image.py --model ... --image ...` | JSON prediction; probability ranges and sum verified |
| FastAPI `/health`, `/model-info`, `/predict` | All returned HTTP 200; API and CLI agreed on the same image |
| Actual `uvicorn` process / model selection through an environment variable | `/model-info` HTTP 200; the newly trained 224px/CLAHE model loaded |
| Actual Streamlit process | Local `/_stcore/health` HTTP 200; browser upload/interaction was not separately tested in this run |
| `python -m pytest` | **176 passed**, 0 failed / 0 skipped, 1 warning |
| `python -m ruff check .` | Passed |

`httpx` was added to development dependencies for API tests. The single test
warning is Starlette's deprecation of the `httpx` TestClient backend in the
installed version; no tests failed.

## Commands and limitations

Main usage commands are in the [README](../README.md). The training check used
the same ResNet18, CLAHE, 224px, ImageNet initialization, LR 0.0005, and recall
0.98 workflow; only `--epochs 3 --batch-size 4` shortened the check. The main
recipe uses `--epochs 16` and default batch size 24. Full 16-epoch Kermany
training was not rerun during this publication preparation.

The 48 synthetic PNGs are not real X-rays. This check is **evidence that the
software works end to end**, not evidence of medical performance, model
improvement, or actual patient independence. Kermany/SSMU/OpenI tests were not
rescored; the historical main model was unchanged. CPU, Linux/macOS, and other
Python/PyTorch versions were not verified in this fresh installation run.

Detailed local records are under `reports/metrics/release_check_v1/`:
`workflow_check.json`, `demo_servers_check.json`, `pytest.xml`, `logs/`, and
source/virtual environment copies. These large, machine-specific files remain outside Git.
