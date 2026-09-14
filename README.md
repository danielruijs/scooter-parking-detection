# Scooter Parking Detection

Fine-tune and benchmark vision classifiers and Vision-Language Models (VLMs) on the Kaggle [Improper Scooter Parking Detection](https://www.kaggle.com/datasets/prodigyanalysis/improper-scooter-parking-detection) dataset.

## 1. Prepare Dataset

Downloads the Kaggle dataset via `kagglehub` and organizes images into `data/train` and `data/val`:

```bash
uv run data.py
```

## 2. Benchmarking models

### [Vision Classifiers](classifiers/README.md) (`classifiers/`)

Fine-tune and benchmark lightweight vision backbones for binary classification (`proper` vs `improper`).

See [`classifiers/README.md`](classifiers/README.md) for training instructions, benchmarking commands, and results.

### [Vision-Language Models](vlms/README.md) (`vlms/`)

Zero-shot evaluation of modern VLMs for parking classification and actionable feedback generation.

See [`vlms/README.md`](vlms/README.md) for setup, supported models, benchmarking commands, and results.
