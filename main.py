"""Project #8: reproducible machine-learning experiment for synthetic motor maintenance."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from src.data import generate_readings, save_csv, split_by_machine
from src.evaluate import evaluate_holdout
from src.model import fit_logistic_regression, load_model, save_model
from src.report import write_reports


@dataclass(frozen=True)
class InputReading:
    vibration_mm_s: float
    temperature_c: float
    current_a: float
    run_hours: float


def build_parser():
    p = argparse.ArgumentParser(description="Educational ML predictive maintenance experiment using synthetic data")
    p.add_argument("--seed", type=int, default=2026, help="Synthetic dataset random seed")
    p.add_argument("--machines", type=int, default=40, help="Simulated machines, minimum 8")
    p.add_argument("--samples-per-machine", type=int, default=10)
    p.add_argument("--output", type=Path, default=Path("output"))
    p.add_argument("--predict", action="store_true", help="Predict from a previously saved model")
    p.add_argument("--vibration", type=float)
    p.add_argument("--temperature", type=float)
    p.add_argument("--current", type=float)
    p.add_argument("--hours", type=float)
    return p


def run(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.predict:
        if any(v is None for v in (args.vibration, args.temperature, args.current, args.hours)):
            parser.error("--predict requires --vibration, --temperature, --current and --hours")
        import math
        values = (args.vibration, args.temperature, args.current, args.hours)
        if not all(math.isfinite(v) and v >= 0 for v in values):
            parser.error("Prediction measurements must be nonnegative finite numbers")
        try:
            model = load_model(args.output / "model.json")
        except (OSError, ValueError, KeyError) as exc:
            parser.error(f"Load a model by running the demo first: {exc}")
        r = InputReading(*values)
        probability = model.predict_probability(r)
        print("MODE: EDUCATIONAL SYNTHETICALLY TRAINED MODEL — NOT PHYSICAL VALIDATION")
        print(f"Synthetic fault score: {probability:.1%} | Classification: {'FAULT' if probability >= 0.5 else 'NORMAL'}")
        return 0
    if args.machines < 8 or args.samples_per_machine < 2:
        parser.error("Use at least 8 machines and 2 samples per machine")
    readings = generate_readings(args.machines, args.samples_per_machine, args.seed)
    train, test = split_by_machine(readings)
    model = fit_logistic_regression(train)
    evaluation = evaluate_holdout(model, test)
    report = {"project": "AI Predictive Maintenance System", "mode": "SYNTHETIC EDUCATIONAL DATA",
              "seed": args.seed, "training_samples": len(train),
              "training_machines": len({r.machine_id for r in train}),
              "evaluation": evaluation}
    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(readings, args.output / "synthetic_telemetry.csv")
    save_model(model, args.output / "model.json")
    jp, hp = write_reports(report, args.output)
    ml, rule = evaluation["machine_learning"], evaluation["engineering_rule"]
    print("MODE: SYNTHETIC MAINTENANCE DATA — NO REAL EQUIPMENT")
    print(f"Simulated readings: {len(readings)} from {args.machines} machines")
    print(f"Training: {len(train)} samples | Holdout: {len(test)} samples from {evaluation['holdout_machines']} distinct machines")
    print(f"Held-out ML: accuracy={ml['accuracy']:.1%}, precision={ml['precision']:.1%}, recall={ml['recall']:.1%}, F1={ml['f1']:.1%}")
    print(f"Threshold rule: accuracy={rule['accuracy']:.1%}, precision={rule['precision']:.1%}, recall={rule['recall']:.1%}, F1={rule['f1']:.1%}")
    print(f"Saved model: {(args.output / 'model.json').resolve()}")
    print(f"Saved CSV: {(args.output / 'synthetic_telemetry.csv').resolve()}")
    print(f"Saved JSON: {jp.resolve()}")
    print(f"Saved HTML: {hp.resolve()}")
    print('Windows: start "" "output\\dashboard.html"')
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
