from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

if TYPE_CHECKING:
    from vlm_models import VLMAdapter


class DatasetImage(BaseModel):
    path: Path
    rel_path: str
    gt_proper: bool
    gt_label: str


class PredictionRecord(BaseModel):
    rel_path: str
    gt_proper: bool
    gt_label: str
    pred_proper: bool | None
    pred_label: str
    is_correct: bool
    feedback: str
    raw_text: str
    parse_success: bool
    latency_ms: float


class BenchmarkResult(BaseModel):
    model_name: str
    display_name: str
    accuracy: float
    macro_f1: float
    precision_macro: float
    recall_macro: float
    parse_rate: float
    mean_latency_ms: float
    records: list[PredictionRecord]

    @classmethod
    def compute(
        cls, adapter: "VLMAdapter", records: list[PredictionRecord]
    ) -> "BenchmarkResult":
        y_true = [1 if r.gt_proper else 0 for r in records]
        y_pred = [
            1 if r.pred_proper is True else (0 if r.pred_proper is False else -1)
            for r in records
        ]

        acc = accuracy_score(y_true, y_pred) * 100
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0) * 100
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0) * 100
        parse_rate = (
            (sum(r.parse_success for r in records) / len(records)) * 100
            if records
            else 0.0
        )
        mean_latency = (
            (sum(r.latency_ms for r in records) / len(records)) if records else 0.0
        )

        return cls(
            model_name=adapter.name,
            display_name=adapter.display_name,
            accuracy=acc,
            macro_f1=f1,
            precision_macro=prec,
            recall_macro=rec,
            parse_rate=parse_rate,
            mean_latency_ms=mean_latency,
            records=records,
        )
