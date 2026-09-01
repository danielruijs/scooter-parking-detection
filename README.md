# Scooter Parking Detection

A repository to fine-tune and benchmark vision models on the Kaggle [Improper Scooter Parking Detection](https://www.kaggle.com/datasets/prodigyanalysis/improper-scooter-parking-detection) dataset.

## Quick Start

### 1. Prepare Dataset

Downloads the dataset via `kagglehub` and organizes images into `data/train` and `data/val` folders:

```bash
uv run data.py
```

### 2. Fine-Tune Models

```bash
# Train all models defined in models.py
uv run train.py --all --epochs 10

# Train a single model (e.g. resnet18, vit, dinov2, mobilenetv3, efficientnetb0, convnextv2)
uv run train.py --model resnet18 --epochs 10
```

### 3. Benchmark Quality & Speed

Evaluates accuracy, F1-score, and GPU latency/throughput:

```bash
uv run benchmark.py
```
