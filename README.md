# AI Predictive Maintenance System

**Python · Machine Learning · Industrial Analytics · Model Evaluation · HTML Dashboard · CI/CD**

An educational ML project that estimates possible motor faults from **synthetic** vibration, temperature, electrical current, and runtime measurements. A logistic regression classifier implemented in standard-library Python is compared with fixed engineering-threshold rules and evaluated on held-out simulated motors.

> **Important:** These readings and fault labels are simulated, not actual failures or maintenance records. **Never use the model to make real industrial maintenance or equipment-safety decisions.**

## What this project demonstrates

- Generate 400 reproducible synthetic measurements from 40 virtual motors.
- Split by **whole machine ID**, preventing the same motor from appearing in training and holdout sets.
- Fit logistic regression with feature scaling learned from **training data only**.
- Evaluate on 100 synthetic readings from 10 held-out motors.
- Compare model predictions with a transparent threshold rule.
- Calculate accuracy, precision, recall, F1, TP, TN, FP, and FN.
- Export CSV telemetry, saved JSON model, evaluation JSON, and an offline HTML dashboard.
- Run **41 automated unit/integration tests** and GitHub Actions CI.

**Requirements:** Python 3.10 or newer; no third-party dependencies.

## Windows instructions

Download this repository with **Code → Download ZIP**, choose **Extract All**, then open the extracted folder containing `main.py`. Click the File Explorer address bar, type `cmd`, and press Enter.

Run these commands **one at a time**:

```cmd
python main.py
start "" "output\dashboard.html"
python -m unittest discover -s tests -v
```

To test a new synthetic motor reading **after training**:

```cmd
python main.py --predict --vibration 7 --temperature 91 --current 16 --hours 9500
```

To generate a fresh simulation with a different seed and dataset:

```cmd
python main.py --seed 2027 --machines 50 --samples-per-machine 12
```

The model and dashboard are written to `output/`; this generated folder is excluded from Git tracking.

## Default holdout results

| Metric | Logistic regression | Engineering thresholds |
|---|---:|---:|
| Accuracy | 78.0% | 72.0% |
| Precision | 66.7% | 45.5% |
| Fault recall | 37.0% | 18.5% |
| F1 score | 47.6% | 26.3% |
| True positives | 10 | 5 |
| False negatives | 17 | 22 |

**Interpretation:** Logistic regression detected 10 of 27 synthetic faults, missing 17. This is a significant limitation, despite the 78% overall accuracy. The figures are for this artificial holdout experiment only—not demonstrated predictive performance on real industrial machinery.

## Screenshot evidence — Windows run and model evaluation

The screenshots below were captured while running this educational system locally on Windows.

### 1. Training and report generation

Default training uses 300 readings from 30 virtual motors and evaluates on 100 readings from 10 separate motor IDs. The application exports the JSON model, synthetic CSV data, evaluation JSON, and HTML dashboard.

![Windows terminal showing synthetic model training and exported results](Screenshot%202026-11-21%20160411.png)

### 2. Model versus fixed engineering thresholds

The terminal compares the ML classifier with conventional threshold rules, including accuracy, precision, recall, and F1.

![Console comparison of logistic regression and engineering thresholds](Screenshot%202026-11-21%20160430.png)

### 3. Automated unit testing

The Python tests cover synthetic data generation, group-aware train/test splitting, model fitting, serialization, evaluation, command-line handling, and reports.

![Windows console running machine learning unit and integration tests](Screenshot%202026-11-21%20160618.png)

### 4. All 41 tests passed and example prediction

The test suite concluded with `Ran 41 tests` and `OK`. A separate example with high simulated vibration, temperature, current, and runtime returned a 98.3% **synthetic model fault score**.

![All 41 unit tests passed followed by a synthetic fault classification](Screenshot%202026-11-21%20160828.png)

### 5. Predictive maintenance dashboard overview

The HTML dashboard presents the holdout size, 78% ML accuracy, 37% fault recall, and an F1 comparison with engineering threshold rules.

![HTML dashboard overview with holdout metrics and model comparison](Screenshot%202026-11-21%20161306.png)

### 6. Confusion matrix and example predictions

This dashboard section compares true positives, false positives, true negatives, and false negatives, then displays the first held-out sensor readings and model predictions.

![Model evaluation confusion counts and held-out predictions table](Screenshot%202026-11-21%20161326.png)

### 7. Additional held-out motor classifications

Individual readings show vibration, temperature, current, running hours, synthetic fault labels, model scores and predicted classifications—including some missed synthetic faults.

![Detailed held-out motor sample predictions](Screenshot%202026-11-21%20161426.png)

### 8. Final predictions and project limitations

The final table rows and disclaimer clearly distinguish experimental classification results from validated industrial condition monitoring.

![Final prediction rows and educational safety disclaimer](Screenshot%202026-11-21%20161450.png)

## Repository structure

```text
main.py                            Training and inference CLI
src/data.py                        Synthetic motor telemetry and machine-aware split
src/model.py                       Pure Python logistic regression, scaling, JSON model
src/evaluate.py                    Classification metrics and baseline rules
src/report.py                      Offline HTML and JSON reports
tests/test_project.py              41 automated tests
.github/workflows/python-tests.yml Continuous integration
requirements.txt                   No third-party dependencies
.gitignore                         Keeps generated outputs out of Git
```

## Engineering limitations and future development

- Logistic regression is **supervised machine learning**, not deep learning or generative AI.
- Synthetic fault labels are generated from an artificial hidden risk relationship and noise.
- Evaluating new machine IDs reduces leakage but does **not** establish performance with actual equipment.
- The fault score is not a calibrated probability of real physical failure.
- No ESP32, sensor, or live industrial equipment is connected.
- Before any real deployment, obtain labeled real maintenance data, time-aware validation, expert safety review, and continuous monitoring of missed faults and model drift.

**Suggested next learning exercise:** Change the prediction threshold and quantify how the number of missed faults changes. Then explain the trade-off with false alarms in an interview.
