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
