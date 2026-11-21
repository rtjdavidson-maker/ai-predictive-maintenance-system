"""Unit and integration tests. All datasets used in tests are synthetic."""
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from src.data import Reading, FEATURES, generate_readings, save_csv, split_by_machine
from src.evaluate import evaluate_binary, evaluate_holdout, engineering_threshold
from src.model import fit_logistic_regression, sigmoid, load_model, save_model
from src.report import render_dashboard, write_reports

ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def test_repeatable_data(self):
        self.assertEqual(generate_readings(seed=13), generate_readings(seed=13))

    def test_different_seeds(self):
        self.assertNotEqual(generate_readings(seed=13), generate_readings(seed=14))

    def test_number_of_rows(self):
        self.assertEqual(len(generate_readings(12, 3)), 36)

    def test_invalid_machine_count(self):
        with self.assertRaises(ValueError):
            generate_readings(7, 10)

    def test_invalid_readings_count(self):
        with self.assertRaises(ValueError):
            generate_readings(10, 1)

    def test_sensor_values_nonnegative(self):
        data = generate_readings()
        self.assertTrue(all(r.current_a > 0 and r.vibration_mm_s > 0 and r.run_hours > 0 for r in data))

    def test_binary_fault_labels(self):
        self.assertEqual({r.fault for r in generate_readings()}, {0, 1})

    def test_machine_split_disjoint(self):
        training, test = split_by_machine(generate_readings())
        self.assertTrue({r.machine_id for r in training}.isdisjoint({r.machine_id for r in test}))

    def test_machine_split_counts(self):
        training, test = split_by_machine(generate_readings())
        self.assertEqual((len(training), len(test)), (300, 100))

    def test_split_repeatable(self):
        rows = generate_readings()
        self.assertEqual(split_by_machine(rows), split_by_machine(rows))

    def test_bad_holdout_ratio(self):
        with self.assertRaises(ValueError):
            split_by_machine(generate_readings(), 1.2)

    def test_single_machine_split_rejected(self):
        with self.assertRaises(ValueError):
            split_by_machine(generate_readings()[:10])

    def test_csv_output(self):
        with TemporaryDirectory() as folder:
            dest = Path(folder) / 'telemetry.csv'
            save_csv(generate_readings(8, 2), dest)
            with dest.open(newline='', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 16)
            self.assertIn('temperature_c', rows[0])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.test = split_by_machine(generate_readings())
        cls.model = fit_logistic_regression(cls.train, epochs=900)

    def test_sigmoid_zero(self):
        self.assertEqual(sigmoid(0), 0.5)

    def test_sigmoid_safe_extremes(self):
        self.assertGreater(sigmoid(10000), 0.99)
        self.assertLess(sigmoid(-10000), 0.01)

    def test_feature_names_correct(self):
        self.assertEqual(self.model.feature_names, FEATURES)

    def test_training_data_scaling_only(self):
        expected = sum(r.vibration_mm_s for r in self.train) / len(self.train)
        self.assertAlmostEqual(self.model.means[0], expected)

    def test_probability_in_range(self):
        self.assertTrue(all(0 <= self.model.predict_probability(r) <= 1 for r in self.test))

    def test_predictions_are_binary(self):
        self.assertTrue(all(self.model.predict(r) in (0, 1) for r in self.test))

    def test_prediction_threshold_validated(self):
        with self.assertRaises(ValueError):
            self.model.predict(self.test[0], 1.1)

    def test_training_requires_two_labels(self):
        few = [replace(r, fault=0) for r in self.train[:12]]
        with self.assertRaises(ValueError):
            fit_logistic_regression(few)

    def test_training_requires_ten_rows(self):
        with self.assertRaises(ValueError):
            fit_logistic_regression(self.train[:9])

    def test_invalid_optimizer(self):
        with self.assertRaises(ValueError):
            fit_logistic_regression(self.train, epochs=0)

    def test_model_serialization_roundtrip(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'model.json'
            save_model(self.model, path)
            restored = load_model(path)
            for row in self.test[:3]:
                self.assertAlmostEqual(restored.predict_probability(row), self.model.predict_probability(row))

    def test_invalid_model_schema(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.json'
            path.write_text(json.dumps(dict(feature_names=['oops'], means=[1], stds=[1], weights=[1], bias=0)))
            with self.assertRaises(ValueError):
                load_model(path)

    def test_model_distinguishes_risk_inputs(self):
        low = Reading('X', 1, 1.0, 40, 5, 500, 0)
        high = Reading('Y', 1, 9.0, 100, 19, 16000, 1)
        self.assertGreater(self.model.predict_probability(high), self.model.predict_probability(low))


class EvaluationTests(unittest.TestCase):
    def test_confusion_counts(self):
        m = evaluate_binary([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual((m.tp, m.tn, m.fp, m.fn), (1, 1, 1, 1))

    def test_metrics_computed(self):
        m = evaluate_binary([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual((m.accuracy, m.precision, m.recall, m.f1), (0.5, 0.5, 0.5, 0.5))

    def test_zero_division_is_safe(self):
        m = evaluate_binary([0, 0], [0, 0])
        self.assertEqual((m.precision, m.recall, m.f1), (0, 0, 0))

    def test_bad_label_lengths(self):
        with self.assertRaises(ValueError):
            evaluate_binary([0], [0, 1])

    def test_bad_label_values(self):
        with self.assertRaises(ValueError):
            evaluate_binary([2], [1])

    def test_engineering_threshold(self):
        self.assertEqual(engineering_threshold(Reading('A', 1, 6, 60, 9, 50, 0)), 1)
        self.assertEqual(engineering_threshold(Reading('A', 2, 2, 50, 8, 50, 0)), 0)

    def test_holdout_has_correct_row_count(self):
        train, test = split_by_machine(generate_readings())
        report = evaluate_holdout(fit_logistic_regression(train, epochs=100), test)
        self.assertEqual(report['holdout_samples'], 100)
        self.assertEqual(len(report['predictions']), 100)

    def test_report_contains_comparison(self):
        train, test = split_by_machine(generate_readings())
        report = evaluate_holdout(fit_logistic_regression(train, epochs=100), test)
        self.assertIn('machine_learning', report)
        self.assertIn('engineering_rule', report)

    def test_html_escapes_device_id(self):
        train, test = split_by_machine(generate_readings())
        evaluation = evaluate_holdout(fit_logistic_regression(train, epochs=100), test)
        evaluation['predictions'][0]['machine_id'] = '<script>alert(1)</script>'
        html = render_dashboard(dict(evaluation=evaluation, training_samples=300, training_machines=30, seed=2026))
        self.assertIn('&lt;script&gt;', html)
        self.assertNotIn('<script>alert(1)</script>', html)

    def test_report_files_generated(self):
        train, test = split_by_machine(generate_readings())
        evaluation = evaluate_holdout(fit_logistic_regression(train, epochs=100), test)
        report = dict(evaluation=evaluation, training_samples=300, training_machines=30, seed=2026)
        with TemporaryDirectory() as folder:
            jp, hp = write_reports(report, Path(folder))
            self.assertTrue(jp.is_file() and hp.is_file())
            self.assertIn('Synthetic', hp.read_text(encoding='utf-8'))


class CLITests(unittest.TestCase):
    def test_demo_smoke_test(self):
        with TemporaryDirectory() as folder:
            process = subprocess.run([sys.executable, str(ROOT / 'main.py'), '--output', folder],
                                     capture_output=True, text=True, check=True)
            self.assertIn('SYNTHETIC MAINTENANCE DATA', process.stdout)
            self.assertTrue((Path(folder) / 'dashboard.html').exists())
            self.assertTrue((Path(folder) / 'model.json').exists())

    def test_prediction_smoke_test(self):
        with TemporaryDirectory() as folder:
            subprocess.run([sys.executable, str(ROOT / 'main.py'), '--output', folder],
                           capture_output=True, check=True)
            proc = subprocess.run([sys.executable, str(ROOT / 'main.py'), '--output', folder,
                                   '--predict', '--vibration', '7', '--temperature', '91',
                                   '--current', '16', '--hours', '9500'],
                                  capture_output=True, text=True, check=True)
            self.assertIn('Synthetic fault score', proc.stdout)

    def test_predict_arguments_required(self):
        proc = subprocess.run([sys.executable, str(ROOT / 'main.py'), '--predict', '--vibration', '2'],
                              capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('requires', proc.stderr)

    def test_prediction_without_model_rejected(self):
        with TemporaryDirectory() as folder:
            proc = subprocess.run([sys.executable, str(ROOT / 'main.py'), '--output', folder,
                                   '--predict', '--vibration', '2', '--temperature', '50',
                                   '--current', '8', '--hours', '5000'],
                                  capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)

    def test_bad_machine_count_rejected(self):
        proc = subprocess.run([sys.executable, str(ROOT / 'main.py'), '--machines', '1'],
                              capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)


if __name__ == '__main__':
    unittest.main()
