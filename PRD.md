# PRD: Fruit Image Classifier (Fruits 360 + CNN)

## 1. Summary

Build a convolutional neural network that classifies fruit images from the Fruits 360 dataset, delivered as a Jupyter notebook covering data loading, model training, evaluation, and inference on new images.

## 2. Problem / Motivation

Learn and demonstrate end-to-end image classification with CNNs on a real, moderately-sized multi-class dataset. Serves as a portfolio project showing data pipeline, model design, training, and evaluation practices.

## 3. Goals

- Train a CNN that classifies fruit images with high accuracy (target: >95% test accuracy — dataset images are clean/uniform, so this is achievable).
- Produce a single, well-organized Jupyter notebook that runs top-to-bottom without manual intervention (aside from Kaggle API credentials).
- Include a simple inference cell that classifies a new/uploaded image.

## 4. Non-Goals

- No deployment (API/web app) — notebook only, for this iteration.
- No real-time or mobile inference optimization.
- No hyperparameter search infrastructure (e.g., Optuna) — manual/simple tuning only.

## 5. Dataset

- Source: Kaggle — moltean/fruits (Fruits-360). Data files already downloaded and provided locally — no Kaggle API download step needed in the notebook.
- Scale: this project uses the `fruits-360_100x100` variant — 260 classes, 137,221 training images, 45,724 test images.
- Image format: 100x100 px, RGB, white background, pre-cropped — low noise, high class separability.
- Split: Dataset ships with pre-made Training/ and Test/ folders; a validation split (10-15%) is carved from Training.
- License: Confirm current license terms on the Kaggle dataset page before any redistribution or commercial use.
- Storage: Raw image files stay local only (excluded via .gitignore) — not committed to the repo given size.

## 6. Approach

- Framework: TensorFlow/Keras.
- Data pipeline: tf.data with augmentation (rotation, flip, zoom) to improve generalization, even though source images are clean.
- Model: Custom CNN (Conv2D + MaxPooling stack, BatchNorm, Dropout, Dense softmax output) as baseline; transfer learning (MobileNetV2) as a stretch-goal comparison.
- Training: Categorical cross-entropy loss, Adam optimizer, early stopping + model checkpointing on validation loss.
- Evaluation: Accuracy, per-class precision/recall/F1, confusion matrix (top confused classes), sample misclassification gallery.

## 7. Success Metrics

- Test accuracy ≥ 95%.
- Notebook runs end-to-end in reasonable time on the available hardware (GPU or CPU); runtime is documented.
- Clear visualizations: training curves, confusion matrix, misclassified examples.

## 8. Milestones

1. Data exploration on existing local files (class counts, sample images, image stats) — no download step required.
2. Data pipeline + augmentation setup.
3. Baseline CNN built, trained, evaluated.
4. Tuning pass (architecture/hyperparameters) to hit accuracy target.
5. (Stretch) Transfer learning comparison.
6. Final write-up: results, confusion matrix, inference demo on a novel image.

## 9. Risks / Open Questions

- Class imbalance across the 260 categories — checked during EDA.
- Fruits-360 images are studio-quality (white background, single object); a model trained here may not generalize to real-world photos — noted as a limitation, not oversold as production-ready.
- Full dataset size (~1.1GB for the 100x100 variant) — verify environment has enough disk/bandwidth.
- License review pending — flagged above, not yet confirmed for this PRD.

## 10. Deliverable

One Jupyter notebook (`notebooks/fruit_classifier.ipynb`) containing all of the above, plus a short markdown summary at the end (final metrics, key learnings, limitations).
