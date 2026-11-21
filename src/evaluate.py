"""Holdout evaluation and baseline comparison; no synthetic evaluation on training rows."""
from dataclasses import asdict, dataclass
from .data import Reading
from .model import LogisticModel


@dataclass(frozen=True)
class Metrics:
    tp: int
    tn: int
    fp: int
    fn: int
    accuracy: float
    precision: float
    recall: float
    f1: float

    def to_dict(self):
        return asdict(self)


def evaluate_binary(actual: list[int], predicted: list[int]) -> Metrics:
    if not actual or len(actual) != len(predicted):
        raise ValueError("Actual and predicted labels must be nonempty and equal length")
    if any(v not in (0, 1) for v in actual + predicted):
        raise ValueError("Labels must be 0 or 1")
    tp = sum(a == 1 and p == 1 for a, p in zip(actual, predicted))
    tn = sum(a == 0 and p == 0 for a, p in zip(actual, predicted))
    fp = sum(a == 0 and p == 1 for a, p in zip(actual, predicted))
    fn = sum(a == 1 and p == 0 for a, p in zip(actual, predicted))
    n = len(actual)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Metrics(tp, tn, fp, fn, (tp + tn) / n, precision, recall, f1)


def engineering_threshold(row: Reading) -> int:
    """Deliberately simple fixed rule for a fair model comparison."""
    return int(row.vibration_mm_s > 5.7 or row.temperature_c > 84.0 or row.current_a > 15.0)


def evaluate_holdout(model: LogisticModel, test: list[Reading], threshold: float = 0.5) -> dict:
    labels = [r.fault for r in test]
    ml_predictions = [model.predict(r, threshold) for r in test]
    baseline_predictions = [engineering_threshold(r) for r in test]
    rows = []
    for r, ml, baseline in zip(test, ml_predictions, baseline_predictions):
        rows.append({
            "machine_id": r.machine_id, "sample_id": r.sample_id,
            "vibration_mm_s": r.vibration_mm_s,
            "temperature_c": r.temperature_c,
            "current_a": r.current_a, "run_hours": r.run_hours,
            "synthetic_fault_label": r.fault,
            "predicted_fault_probability": round(model.predict_probability(r), 4),
            "model_prediction": ml,
            "rule_prediction": baseline,
        })
    return {"machine_learning": evaluate_binary(labels, ml_predictions).to_dict(),
            "engineering_rule": evaluate_binary(labels, baseline_predictions).to_dict(),
            "holdout_samples": len(test), "holdout_machines": len({r.machine_id for r in test}),
            "predictions": rows}
