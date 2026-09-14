# Vision Classifiers

Fine-tune and benchmark vision classifiers for binary scooter parking detection (`proper` vs. `improper`).

## Supported Models

Supported models are defined in [`models.py`](models.py):

## 1. Fine-Tuning

Models are trained using Hugging Face's `Trainer` with AdamW and cross-entropy loss, selecting the checkpoint with the highest validation macro F1-score.

```bash
# Train a single model
uv run train.py --model resnet18 --epochs 10

# Train all models sequentially
uv run train.py --all --epochs 10
```

Trained weights and image processors are saved to `output/<model_key>/best_model/`.

## 2. Benchmarking Quality & Speed

Evaluates classification accuracy, macro F1-score, inference latency, and throughput:

```bash
uv run benchmark.py
```

Results are printed in tabular format and written to [`benchmark_results.md`](benchmark_results.md).
