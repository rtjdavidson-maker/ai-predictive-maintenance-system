"""Generate readable offline HTML and JSON model evaluation reports."""
import html
import json
from pathlib import Path


def write_reports(report: dict, output: Path) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    jp, hp = output / "evaluation.json", output / "dashboard.html"
    jp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    hp.write_text(render_dashboard(report), encoding="utf-8")
    return jp, hp


def render_dashboard(report: dict) -> str:
    model = report["evaluation"]["machine_learning"]
    rule = report["evaluation"]["engineering_rule"]
    rows = []
    for entry in report["evaluation"]["predictions"][:25]:
        label = "Fault" if entry["synthetic_fault_label"] else "Normal"
        pred = "Fault" if entry["model_prediction"] else "Normal"
        status_cls = "warn" if entry["model_prediction"] else "good"
        rows.append("<tr>" +
                    f'<td>{html.escape(entry["machine_id"])}</td><td>{entry["sample_id"]}</td>' +
                    f'<td>{entry["vibration_mm_s"]:.2f}</td><td>{entry["temperature_c"]:.1f}</td>' +
                    f'<td>{entry["current_a"]:.2f}</td><td>{entry["run_hours"]:.0f}</td>' +
                    f'<td>{html.escape(label)}</td><td>{entry["predicted_fault_probability"]:.1%}</td>' +
                    f'<td class="{status_cls}">{pred}</td></tr>')
    def metric_card(label, value):
        return f'<div class="card"><span>{html.escape(label)}</span><strong>{value}</strong></div>'
    cards = ''.join((metric_card("Machines in holdout", report["evaluation"]["holdout_machines"]),
                     metric_card("Holdout readings", report["evaluation"]["holdout_samples"]),
                     metric_card("ML accuracy", f'{model["accuracy"]:.1%}'),
                     metric_card("ML fault recall", f'{model["recall"]:.1%}')))
    score_rows = ''.join(
        f'<tr><th>{name}</th><td>{v["accuracy"]:.1%}</td><td>{v["precision"]:.1%}</td>'
        f'<td>{v["recall"]:.1%}</td><td>{v["f1"]:.1%}</td>'
        f'<td>{v["tp"]}</td><td>{v["fp"]}</td><td>{v["tn"]}</td><td>{v["fn"]}</td></tr>'
        for name, v in (("Logistic regression", model), ("Engineering thresholds", rule)))
    score_bars = ''.join(
        '<div class="barlabel"><span>' + name + '</span><strong>' + f'{v["f1"]:.1%}' + '</strong></div>' +
        '<div class="track"><div class="bar" style="width:' + f'{v["f1"] * 100:.1f}' + '%"></div></div>'
        for name, v in (("ML F1 score", model), ("Threshold rule F1", rule)))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Predictive Maintenance Evaluation</title>
<style>
:root{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:#0b1424;color:#e7effa}}
*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1170px;padding:22px;margin:24px auto}}
h1{{font-size:clamp(1.6rem,4vw,2.5rem);margin:0 0 8px}}p{{color:#b8cce0;line-height:1.5}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:13px;margin:22px 0}}
.card,.panel{{background:#1a2b44;border:1px solid #37506d;border-radius:14px;padding:20px}}
.card span{{color:#bed1e5;font-size:.87rem}}.card strong{{display:block;font-size:2rem;margin-top:9px}}
.panel{{margin:18px 0;overflow-x:auto}}table{{border-collapse:collapse;width:100%;min-width:790px}}
caption{{text-align:left;padding:5px 0 16px;font-weight:bold;font-size:1.1rem}}
th,td{{text-align:left;border-bottom:1px solid #364c67;padding:11px 9px;font-size:.86rem}}
th{{color:#cbdcf1}}.good{{color:#6be3b9;font-weight:700}}.warn{{color:#ff9eaa;font-weight:700}}
.track{{background:#334966;height:15px;border-radius:10px;margin:6px 0 15px;overflow:hidden}}
.bar{{height:100%;background:#56b4df}}.barlabel{{display:flex;justify-content:space-between;color:#c4d9ec}}
.notice{{border-left:4px solid #efbd60;padding:12px 16px;background:#2a3041;color:#f5e3be;border-radius:5px}}
</style></head><body><main>
<h1>AI Predictive Maintenance Dashboard</h1>
<p class="notice">Educational model trained and evaluated on <strong>synthetic motor data</strong>.
Fault labels are simulated. These scores do not establish real-world predictive accuracy or equipment safety.</p>
<p>Model: logistic regression (built from Python standard library) · Split: held-out machine IDs ·
Train: {report["training_samples"]} readings / {report["training_machines"]} machines · Seed: {report["seed"]}</p>
<section class="cards">{cards}</section>
<section class="panel"><h2>Holdout performance vs. fixed engineering rules</h2>
{score_bars}
<table><tr><th>Method</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>TP</th><th>FP</th><th>TN</th><th>FN</th></tr>
{score_rows}</table><p>Recall measures how many synthetic fault labels were detected. High accuracy alone can hide missed faults, so also consider F1 and the confusion counts.</p></section>
<section class="panel"><h2>Held-out device predictions</h2><p>First 25 rows from previously unseen machine IDs.</p>
<table><tr><th>Machine</th><th>Sample</th><th>Vibration (mm/s)</th><th>Temp (°C)</th><th>Current (A)</th><th>Runtime (hr)</th><th>Synthetic label</th><th>ML score</th><th>Prediction</th></tr>
{''.join(rows)}</table></section>
<p>Not a certified predictive maintenance tool. Before any industrial use, validate against real maintenance outcomes and engineering experts; monitor false negatives and drift.</p>
</main></body></html>'''
