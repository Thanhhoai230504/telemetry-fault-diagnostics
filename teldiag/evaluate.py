"""Score the diagnoser over many simulated runs: accuracy, detection latency and false alarms."""
from __future__ import annotations

import numpy as np

from .diagnose import CODE_FOR_FAULT, Thresholds, diagnose
from .simulate import FAULTS, simulate

LABELS = ["OK"] + [CODE_FOR_FAULT[f] for f in FAULTS if f != "none"]


def truth_code(fault: str) -> str:
    return "OK" if fault == "none" else CODE_FOR_FAULT[fault]


def evaluate(runs_per_class: int = 20, seed: int = 0, th: Thresholds | None = None,
             severity: float = 1.0) -> dict:
    results = []
    for c, fault in enumerate(FAULTS):
        for i in range(runs_per_class):
            run_seed = seed + 1000 * c + i
            run = simulate(fault, seed=run_seed, severity=severity)
            report = diagnose(run, th)
            results.append(score_run(run.fault, run.onset, float(run.t[-1] + run.spec.dt), report, run_seed))
    return summarize(results)


def score_run(fault: str, onset: float | None, duration: float, report: dict, seed: int = 0) -> dict:
    truth = truth_code(fault)
    healthy_until = duration if onset is None else onset
    false_alarms = [a for a in report["alarms"] if a["time_s"] < healthy_until]
    correct = [a for a in report["alarms"] if a["code"] == truth and onset is not None and a["time_s"] >= onset]
    return {
        "seed": seed,
        "truth": truth,
        "predicted": report["diagnosis"],
        "latency_s": correct[0]["time_s"] - onset if correct else None,
        "false_alarms": len(false_alarms),
        "healthy_hours": healthy_until / 3600.0,
    }


def summarize(results: list[dict]) -> dict:
    idx = {lab: i for i, lab in enumerate(LABELS)}
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for r in results:
        confusion[idx[r["truth"]], idx[r["predicted"]]] += 1

    per_class = {}
    for lab in LABELS:
        rows = [r for r in results if r["truth"] == lab]
        entry = {"runs": len(rows), "correct": sum(r["predicted"] == lab for r in rows)}
        if lab != "OK":
            lat = [r["latency_s"] for r in rows if r["latency_s"] is not None]
            entry["detected"] = len(lat)
            entry["latency_median_s"] = float(np.median(lat)) if lat else None
            entry["latency_p90_s"] = float(np.percentile(lat, 90)) if lat else None
        per_class[lab] = entry

    normal = [r for r in results if r["truth"] == "OK"]
    healthy_hours = sum(r["healthy_hours"] for r in results)
    return {
        "runs": len(results),
        "accuracy": float(np.trace(confusion) / max(1, confusion.sum())),
        "false_positive_rate_runs": (sum(r["false_alarms"] > 0 for r in normal) / len(normal)) if normal else None,
        "false_alarms_per_healthy_hour": sum(r["false_alarms"] for r in results) / healthy_hours,
        "labels": LABELS,
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def sweep(severities: list[float], runs_per_class: int = 20, seed: int = 0) -> list[dict]:
    """Evaluate at several fault severities to show where detection starts to fail."""
    return [{"severity": sev, **evaluate(runs_per_class, seed, severity=sev)} for sev in severities]


def format_sweep(rows: list[dict]) -> str:
    codes = [lab for lab in LABELS if lab != "OK"]
    head = f"{'severity':>8}{'accuracy':>10}" + "".join(f"{c + ' det/lat':>16}" for c in codes)
    lines = ["per fault: detected runs / median latency", head]
    for r in rows:
        cells = ""
        for c in codes:
            e = r["per_class"][c]
            lat = "-" if e["latency_median_s"] is None else f"{e['latency_median_s']:.0f}s"
            cells += f"{str(e['detected']) + '/' + str(e['runs']) + ' ' + lat:>16}"
        lines.append(f"{r['severity']:>8.2f}{r['accuracy']:>10.1%}" + cells)
    return "\n".join(lines)


def format_summary(s: dict) -> str:
    lines = [f"runs: {s['runs']}   accuracy: {s['accuracy']:.1%}   "
             f"false-positive rate (healthy runs with any alarm): {s['false_positive_rate_runs']:.1%}   "
             f"false alarms per healthy hour: {s['false_alarms_per_healthy_hour']:.3f}", "",
             "confusion matrix (rows = truth, cols = predicted)",
             "        " + "".join(f"{lab:>6}" for lab in s["labels"])]
    for lab, row in zip(s["labels"], s["confusion_matrix"]):
        lines.append(f"{lab:>6}  " + "".join(f"{v:>6}" for v in row))
    lines += ["", f"{'class':<6}{'correct':>10}{'latency median':>17}{'latency p90':>14}"]
    for lab, e in s["per_class"].items():
        med = "-" if e.get("latency_median_s") is None else f"{e['latency_median_s']:.0f} s"
        p90 = "-" if e.get("latency_p90_s") is None else f"{e['latency_p90_s']:.0f} s"
        lines.append(f"{lab:<6}{e['correct']:>6}/{e['runs']:<3}{med:>17}{p90:>14}")
    return "\n".join(lines)
