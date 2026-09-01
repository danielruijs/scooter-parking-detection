import argparse
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)

from models import MODELS, VisionModel


def compute_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=1)
    acc = evaluate.load("accuracy").compute(
        predictions=preds, references=eval_pred.label_ids
    )["accuracy"]
    f1 = evaluate.load("f1").compute(
        predictions=preds, references=eval_pred.label_ids, average="macro"
    )["f1"]
    return {"accuracy": acc, "f1": f1}


def train_one(model_cfg: VisionModel, args, dataset):
    print(f"\n--- Training {model_cfg.display_name} ({model_cfg.hf_id}) ---")
    out = Path(args.output_dir) / model_cfg.name

    lbls = dataset["train"].features["label"].names
    image_processor = AutoImageProcessor.from_pretrained(model_cfg.hf_id)
    model = AutoModelForImageClassification.from_pretrained(
        model_cfg.hf_id,
        num_labels=len(lbls),
        id2label=dict(enumerate(lbls)),
        label2id={n: i for i, n in enumerate(lbls)},
        ignore_mismatched_sizes=True,
    )

    def collate_fn(batch):
        inputs = image_processor(
            [x["image"].convert("RGB") for x in batch], return_tensors="pt"
        )
        inputs["labels"] = torch.tensor([x["label"] for x in batch])
        return inputs

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=1,
            fp16=torch.cuda.is_available(),
            remove_unused_columns=False,
            report_to="none",
        ),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()

    best_p = out / "best_model"
    trainer.save_model(str(best_p))
    image_processor.save_pretrained(str(best_p))
    print(
        f"[Done] {model_cfg.name} | Val Acc: {metrics['eval_accuracy']:.4f} | F1: {metrics['eval_f1']:.4f}"
    )


def main():
    p = argparse.ArgumentParser(
        description="Train vision classifiers on scooter dataset"
    )
    p.add_argument("--model", choices=list(MODELS.keys()) + ["all"], default="all")
    p.add_argument("--all", action="store_true")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--output-dir", default="output")
    args = p.parse_args()

    ds = load_dataset("imagefolder", data_dir="data")
    targets = (
        MODELS.values() if (args.all or args.model == "all") else [MODELS[args.model]]
    )
    for cfg in targets:
        train_one(cfg, args, ds)


if __name__ == "__main__":
    main()
