"""Command-line interface: everything runs headless."""
from __future__ import annotations

import argparse
import json
import os
import sys

from .diagnose import diagnose
from .evaluate import evaluate, format_summary, format_sweep, sweep
from .simulate import FAULTS, Run, simulate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="teldiag", description="Multi-sensor telemetry fault diagnostics")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("simulate", help="write one simulated run to an .npz file")
    p.add_argument("--fault", choices=FAULTS, default="none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--duration", type=float, default=1800.0, help="seconds")
    p.add_argument("--onset", type=float, default=None, help="fault start in seconds (default: random 600-1200)")
    p.add_argument("--severity", type=float, default=1.0, help="fault size multiplier (default 1.0)")
    p.add_argument("--out", required=True)

    p = sub.add_parser("diagnose", help="diagnose a run file and print a JSON report")
    p.add_argument("run")
    p.add_argument("--out", help="also write the report to this file")

    p = sub.add_parser("evaluate", help="score the diagnoser over many simulated runs")
    p.add_argument("--runs-per-class", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write metrics JSON here")
    p.add_argument("--min-accuracy", type=float, default=None, help="exit 1 if accuracy is below this")
    p.add_argument("--max-fp-rate", type=float, default=None, help="exit 1 if the run-level false-positive rate is above this")
    p.add_argument("--severity", type=float, default=1.0, help="fault size multiplier (default 1.0)")

    p = sub.add_parser("sweep", help="evaluate across fault severities")
    p.add_argument("--severities", default="0.1,0.2,0.35,0.5,1.0", help="comma-separated multipliers")
    p.add_argument("--runs-per-class", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write sweep JSON here")

    args = parser.parse_args(argv)

    if args.cmd == "simulate":
        run = simulate(args.fault, seed=args.seed, duration=args.duration, onset=args.onset, severity=args.severity)
        run.save(args.out)
        onset = "-" if run.onset is None else f"{run.onset:.0f} s"
        print(f"wrote {args.out}: fault={run.fault} onset={onset} params={run.params}")
        return 0

    if args.cmd == "diagnose":
        report = diagnose(Run.load(args.run))
        text = json.dumps(report, indent=2)
        print(text)
        if args.out:
            _write(args.out, text)
        return 0

    if args.cmd == "sweep":
        rows = sweep([float(v) for v in args.severities.split(",")], args.runs_per_class, args.seed)
        print(format_sweep(rows))
        if args.out:
            _write(args.out, json.dumps(rows, indent=2))
        return 0

    summary = evaluate(args.runs_per_class, args.seed, severity=args.severity)
    print(format_summary(summary))
    if args.out:
        _write(args.out, json.dumps(summary, indent=2))
    ok = True
    if args.min_accuracy is not None and summary["accuracy"] < args.min_accuracy:
        print(f"FAIL: accuracy {summary['accuracy']:.3f} < {args.min_accuracy}", file=sys.stderr)
        ok = False
    if args.max_fp_rate is not None and summary["false_positive_rate_runs"] > args.max_fp_rate:
        print(f"FAIL: false-positive rate {summary['false_positive_rate_runs']:.3f} > {args.max_fp_rate}", file=sys.stderr)
        ok = False
    return 0 if ok else 1


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
