"""Pure-Python logistic regression, with scaling fitted on training data only.

For education: no sklearn dependency, no claim of production-ready diagnosis.
"""
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
from .data import FEATURES, Reading


def features(row) -> list[float]:
    return [float(getattr(row, col)) for col in FEATURES]


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


@dataclass(frozen=True)
class LogisticModel:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float

    def predict_probability(self, row) -> float:
        values = features(row)
        z = self.bias + sum(w * ((x - mu) / sd) for x, mu, sd, w in
                            zip(values, self.means, self.stds, self.weights))
        return sigmoid(z)

    def predict(self, row, threshold: float = 0.5) -> int:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        return int(self.predict_probability(row) >= threshold)

    def to_dict(self) -> dict:
        return asdict(self)


def fit_logistic_regression(train: list[Reading], epochs: int = 1100,
                            learning_rate: float = 0.16, l2: float = 0.025) -> LogisticModel:
    if len(train) < 10:
        raise ValueError("At least ten training samples are required")
    if len({r.fault for r in train}) < 2:
        raise ValueError("Training set requires both positive and negative labels")
    if epochs < 1 or learning_rate <= 0 or l2 < 0:
        raise ValueError("Invalid optimizer settings")
    samples = [features(r) for r in train]
    dim = len(FEATURES)
    means = [sum(row[j] for row in samples) / len(samples) for j in range(dim)]
    stds = [max(1e-6, math.sqrt(sum((row[j] - means[j]) ** 2 for row in samples) / len(samples)))
            for j in range(dim)]
    normalized = [[(r[j] - means[j]) / stds[j] for j in range(dim)] for r in samples]
    weights = [0.0] * dim
    bias = 0.0
    n = len(train)
    for _ in range(epochs):
        grads = [0.0] * dim
        gbias = 0.0
        for x, sample in zip(normalized, train):
            prediction = sigmoid(bias + sum(w * a for w, a in zip(weights, x)))
            difference = prediction - sample.fault
            gbias += difference
            for i in range(dim):
                grads[i] += difference * x[i]
        for i in range(dim):
            weights[i] -= learning_rate * (grads[i] / n + l2 * weights[i])
        bias -= learning_rate * gbias / n
    return LogisticModel(FEATURES, tuple(means), tuple(stds), tuple(weights), bias)


def save_model(model: LogisticModel, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict(), indent=2) + "\n", encoding="utf-8")


def load_model(path: Path) -> LogisticModel:
    data = json.loads(path.read_text(encoding="utf-8"))
    if tuple(data["feature_names"]) != FEATURES:
        raise ValueError("Unexpected model schema")
    for field in ("means", "stds", "weights"):
        if len(data[field]) != len(FEATURES) or not all(math.isfinite(float(v)) for v in data[field]):
            raise ValueError(f"Invalid {field}")
    if any(float(v) <= 0 for v in data["stds"]) or not math.isfinite(float(data["bias"])):
        raise ValueError("Invalid scaler or intercept")
    return LogisticModel(FEATURES, tuple(data["means"]), tuple(data["stds"]),
                         tuple(data["weights"]), float(data["bias"]))
