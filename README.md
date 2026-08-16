# fruit-match

CNN image classifier for the Fruits 360 dataset.

## Dataset

[Fruits 360 on Kaggle](https://www.kaggle.com/datasets/moltean/fruits) (moltean/fruits). This project uses the `fruits-360_100x100` variant: 260 classes, 137,221 training images, 45,724 test images, 100x100 px RGB, white background.

## Setup

```
python3 -m venv .venv && source .venv/bin/activate  # or use your Python 3.12 install directly
pip install -r requirements.txt
jupyter notebook notebooks/fruit_classifier.ipynb
```

Requires Python 3.12 (TensorFlow does not yet support the latest Python releases).

## Local dataset

Dataset files are already provided locally and are **not** committed to this repo (see `.gitignore`). Place (or symlink) the `fruits-360` folder — containing `Training/` and `Test/` subfolders — at:

```
data/fruits-360/
```

No Kaggle API download step is needed in the notebook since the data is already on hand.

## Model artifacts

Running the notebook saves trained model checkpoints to `models/` (`baseline_cnn.keras`,
`mobilenetv2_transfer.keras`) via `ModelCheckpoint`. This folder is **not** committed
(see `.gitignore` — `*.keras` and `models/` are both excluded) since checkpoints are
large binaries; re-run the notebook to regenerate them.

## Results

Full details, plots, and confusion matrix are in [notebooks/fruit_classifier.ipynb](notebooks/fruit_classifier.ipynb).

| Model | Test accuracy | Params | Train time (CPU, Apple M1 Pro) |
|---|---|---|---|
| Baseline CNN (from scratch) | 88.81% | 522,948 | ~10.3 hours |
| MobileNetV2 (transfer learning) | 94.81% | 2,591,044 | ~2.1 hours |

Neither model quite hit the >=95% test accuracy target on this run — training was
CPU-only (no GPU), so both runs were cut off by early stopping / a fixed epoch budget
rather than run to full convergence. The transfer-learning result suggests the target
is reachable with more training budget, not a different approach.

**Limitations:** Fruits-360 images are studio-quality (centered, single object,
uniform white background). This model has not been validated on real-world photos
and should not be treated as production-ready for that setting.
