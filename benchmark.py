import argparse
import time
from pathlib import Path

import evaluate
import numpy as np
import torch
from PIL import Image
from tabulate import tabulate
from transformers import AutoImageProcessor, AutoModelForImageClassification

from models import MODELS, VisionModel


def benchmark(
    model_cfg: VisionModel,
    ckpt_dir: Path,
    val_dir: Path,
    device: torch.device,
    runs: int = 100,
):
    src = (
        str(ckpt_dir / model_cfg.name / "best_model")
        if (ckpt_dir / model_cfg.name / "best_model").exists()
        else model_cfg.hf_id
    )
    image_processor = AutoImageProcessor.from_pretrained(src)
    model = AutoModelForImageClassification.from_pretrained(src).to(device).eval()

    lbl2id = model.config.label2id
    imgs, targets = [], []
    for cls in ["improper", "proper"]:
        for p in sorted((val_dir / cls).glob("*")):
            imgs.append(Image.open(p).convert("RGB"))
            targets.append(lbl2id.get(cls, 0 if cls == "improper" else 1))

    # 1. Accuracy & F1
    inputs = image_processor(imgs, return_tensors="pt").to(device)
    with torch.inference_mode():
        preds = torch.argmax(model(**inputs).logits, dim=1).cpu().numpy()

    acc = (
        evaluate.load("accuracy").compute(predictions=preds, references=targets)[
            "accuracy"
        ]
        * 100
    )
    f1 = (
        evaluate.load("f1").compute(
            predictions=preds, references=targets, average="macro"
        )["f1"]
        * 100
    )

    # 2. Latency & Throughput
    # Prepare single-image input (batch_size=1) for latency evaluation
    single = image_processor(imgs[0], return_tensors="pt").to(device)

    # Warm up GPU
    for _ in range(10):
        with torch.inference_mode():
            _ = model(**single)
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model(**single)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_latency_ms = float(np.mean(latencies))
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return [
        model_cfg.display_name,
        round(params, 2),
        f"{acc:.1f}%",
        f"{f1:.1f}%",
        f"{mean_latency_ms:.2f} ms",
        f"{1000.0 / mean_latency_ms:.1f} FPS",
    ]


def main():
    p = argparse.ArgumentParser(
        description="Benchmark vision models on scooter dataset"
    )
    p.add_argument("--val-dir", default="data/val")
    p.add_argument(
        "--model-dir",
        default="output",
        help="Directory with fine-tuned model checkpoints",
    )
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dev = torch.device(args.device)
    rows = [
        benchmark(cfg, Path(args.model_dir), Path(args.val_dir), dev)
        for cfg in MODELS.values()
    ]
    rows.sort(key=lambda x: x[3], reverse=True)  # Sort by F1 score
    headers = [
        "Model",
        "Params (M)",
        "Accuracy",
        "F1-Score",
        "Latency",
        "Throughput",
    ]

    table_str = tabulate(rows, headers=headers, tablefmt="github")
    print("\n" + table_str)
    Path("benchmark_results.md").write_text(
        f"# Vision Model Benchmark\n\n**Hardware**: `{torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'CPU'}`\n\n{table_str}\n"
    )
    print("\nResults saved to benchmark_results.md!")


if __name__ == "__main__":
    main()
