import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from vlm_config import AVAILABLE_VLM_MODELS, SYSTEM_PROMPT, USER_PROMPT
from vlm_models import VLMAdapter
from vlm_report import save_benchmark_reports
from vlm_types import BenchmarkResult, DatasetImage, PredictionRecord

sys.stdout.reconfigure(encoding="utf-8")


def load_dataset_images(
    data_dir: Path, split: str = "val", max_samples: int | None = None
) -> list[DatasetImage]:
    items: list[DatasetImage] = []
    split_dir = data_dir / split
    for cls_name, is_proper in [("improper", False), ("proper", True)]:
        for img_path in sorted((split_dir / cls_name).glob("*.*")):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
                items.append(
                    DatasetImage(
                        path=img_path,
                        rel_path=img_path.as_posix(),
                        gt_proper=is_proper,
                        gt_label=cls_name,
                    )
                )

    if max_samples is not None and max_samples < len(items):
        improper = [x for x in items if not x.gt_proper]
        proper = [x for x in items if x.gt_proper]
        half = max_samples // 2
        items = improper[:half] + proper[: (max_samples - half)]

    return items


def run_model_benchmark(
    adapter: VLMAdapter, items: list[DatasetImage]
) -> BenchmarkResult:
    print(
        f"\n{'=' * 56}\nBenchmarking {adapter.display_name} on {len(items)} images...\n{'=' * 56}"
    )
    adapter.load()
    records: list[PredictionRecord] = []

    for idx, item in enumerate(items, 1):
        try:
            with Image.open(item.path) as img:
                res = adapter.generate(
                    img, user_prompt=USER_PROMPT, system_prompt=SYSTEM_PROMPT
                )
        except Exception as e:
            res = {
                "raw_text": f"ERROR: {e}",
                "is_proper": None,
                "feedback": f"Inference failed: {e}",
                "parse_success": False,
                "latency_ms": 0.0,
            }

        pred_proper = res["is_proper"]
        rec = PredictionRecord(
            rel_path=item.rel_path,
            gt_proper=item.gt_proper,
            gt_label=item.gt_label,
            pred_proper=pred_proper,
            pred_label=(
                "proper"
                if pred_proper is True
                else ("improper" if pred_proper is False else "unknown")
            ),
            is_correct=(pred_proper == item.gt_proper)
            if pred_proper is not None
            else False,
            feedback=res["feedback"],
            raw_text=res["raw_text"],
            parse_success=res["parse_success"],
            latency_ms=round(res["latency_ms"], 2),
        )
        records.append(rec)

        status_sym = "[MATCH]" if rec.is_correct else "[MISS] "
        print(
            f"[{idx:02d}/{len(items):02d}] {status_sym} GT: {rec.gt_label:<8} "
            f"| Pred: {rec.pred_label:<8} | {rec.latency_ms:6.1f}ms | Feedback: {rec.feedback[:60]}..."
        )

    adapter.unload()
    result = BenchmarkResult.compute(adapter, records)
    print(
        f"\n--> {result.display_name} Results: Acc: {result.accuracy:.1f}% | Macro F1: {result.macro_f1:.1f}% "
        f"| JSON Parse: {result.parse_rate:.1f}% | Avg Latency: {result.mean_latency_ms:.1f}ms\n"
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark VLMs for scooter parking classification and feedback"
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=["all"] + list(AVAILABLE_VLM_MODELS.keys()),
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "train"],
        help="Dataset split to evaluate (val or train)",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root data directory containing train and val folders",
    )
    parser.add_argument("--output-dir", default="vlm_output")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    pred_json = out_path / f"vlm_predictions_{args.split}.json"
    items = load_dataset_images(
        Path(args.data_dir), split=args.split, max_samples=args.max_samples
    )
    print(
        f"Loaded {len(items)} images from [{args.split}] split "
        f"({sum(1 for x in items if x.gt_proper)} proper, "
        f"{sum(1 for x in items if not x.gt_proper)} improper)"
    )

    # Load previous benchmark results for this split so new models are added incrementally
    all_results: dict[str, BenchmarkResult] = {}
    if pred_json.exists():
        try:
            raw_data = json.loads(pred_json.read_text(encoding="utf-8"))
            all_results = {
                k: BenchmarkResult.model_validate(v) for k, v in raw_data.items()
            }
        except Exception:
            all_results = {}

    targets = (
        AVAILABLE_VLM_MODELS.values()
        if args.model == "all"
        else [AVAILABLE_VLM_MODELS[args.model]]
    )
    for adapter in targets:
        if adapter.name in all_results:
            print(
                f"\nSkipping {adapter.display_name} - already evaluated for [{args.split}]."
            )
        else:
            all_results[adapter.name] = run_model_benchmark(adapter, items)
        save_benchmark_reports(items, all_results, out_path, split=args.split)

    html_file = out_path / f"vlm_results_{args.split}.html"
    print(
        f"\nAll benchmark tasks complete for [{args.split}]! HTML gallery ready at {html_file.resolve()}"
    )


if __name__ == "__main__":
    main()
