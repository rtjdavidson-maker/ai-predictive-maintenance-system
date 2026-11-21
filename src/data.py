"""Generate synthetic machine telemetry with hidden (simulated) fault labels.

Data is artificial. Labels are simulated outcomes, NOT verified maintenance events.
A dedicated random generator ensures reproductions without mutating global randomness.
"""
from dataclasses import dataclass, asdict
import csv
import math
from pathlib import Path
import random

FEATURES = ("vibration_mm_s", "temperature_c", "current_a", "run_hours")

@dataclass(frozen=True)
class Reading:
    machine_id: str
    sample_id: int
    vibration_mm_s: float
    temperature_c: float
    current_a: float
    run_hours: float
    fault: int


def generate_readings(machine_count: int = 40, readings_per_machine: int = 10, seed: int = 2026) -> list[Reading]:
    if machine_count < 8 or readings_per_machine < 2:
        raise ValueError("Use at least 8 machines and 2 samples per machine")
    rng = random.Random(seed)
    out = []
    for machine in range(1, machine_count + 1):
        machine_id = f"MTR-{machine:03d}"
        # Different machines have stable operating conditions: group-aware splitting matters.
        operating_load = rng.uniform(-1.0, 1.0)
        machine_wear = rng.uniform(-1.0, 1.0)
        for sample in range(1, readings_per_machine + 1):
            vibration = max(0.3, 3.2 + machine_wear * 1.4 + rng.gauss(0, 1.7))
            temperature = 60.0 + operating_load * 9 + machine_wear * 5 + rng.gauss(0, 10)
            current = max(0.5, 9.0 + operating_load * 2.8 + rng.gauss(0, 2.3))
            hours = max(100.0, 5200 + machine_wear * 2300 + rng.uniform(-4200, 4200))
            # Hidden probabilistic ground truth: a nonlinear combination with natural overlap/noise.
            latent = (-1.55 + 0.57 * (vibration - 3.0) + 0.055 * (temperature - 60)
                      + 0.20 * (current - 9.0) + 0.00028 * (hours - 5000)
                      + 0.11 * max(vibration - 5.0, 0))
            probability = 1 / (1 + math.exp(-max(-30, min(30, latent))))
            fault = int(rng.random() < probability)
            out.append(Reading(machine_id, sample, round(vibration, 3), round(temperature, 2),
                               round(current, 3), round(hours, 1), fault))
    return out


def save_csv(readings: list[Reading], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Reading.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(item) for item in readings)


def split_by_machine(readings: list[Reading], holdout_ratio: float = 0.25, seed: int = 81):
    """Split whole machines, not individual rows, to avoid machine-level leakage."""
    if not 0 < holdout_ratio < 1:
        raise ValueError("holdout_ratio must be between zero and one")
    machine_ids = sorted({r.machine_id for r in readings})
    if len(machine_ids) < 2:
        raise ValueError("Need at least two distinct machines")
    random.Random(seed).shuffle(machine_ids)
    holdout_n = max(1, min(len(machine_ids) - 1, round(len(machine_ids) * holdout_ratio)))
    holdout = set(machine_ids[:holdout_n])
    train = [r for r in readings if r.machine_id not in holdout]
    test = [r for r in readings if r.machine_id in holdout]
    return train, test
