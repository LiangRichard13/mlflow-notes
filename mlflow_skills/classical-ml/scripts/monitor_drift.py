#!/usr/bin/env python3
"""Production drift monitor: evaluate a deployed model on a new batch, apply
thresholds, tag the run healthy/degraded, and emit an alert flag.

Designed to be run on a cron / Airflow / K8s CronJob schedule.

Usage:
    # Basic: evaluate champion on today's batch
    python scripts/monitor_drift.py \\
        --model-uri models:/WineQualityClassifier@champion \\
        --data today_batch.csv --targets label \\
        --model-type classifier --experiment-name drift-monitor

    # With thresholds + alert
    python scripts/monitor_drift.py \\
        --model-uri models:/Wine@champion \\
        --data today.csv --targets label --model-type classifier \\
        --thresholds-json '{"accuracy_score": 0.85, "roc_auc": 0.90}' \\
        --alert-webhook https://hooks.example.com/drift

Exit codes:
    0  Drift check passed (healthy).
    1  Drift detected (degraded).
    2  Backend / data error.
    3  Usage error.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from typing import Any

EXIT_HEALTHY = 0
EXIT_DEGRADED = 1
EXIT_BACKEND = 2
EXIT_USAGE = 3


def load_thresholds(spec: str) -> dict[str, dict[str, Any]]:
    """Parse 'metric=value[,greater|less]' or JSON into {metric: {threshold, greater_is_better}}."""
    if spec.startswith("{"):
        return json.loads(spec)
    out: dict[str, dict[str, Any]] = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            print(f"Error: bad threshold pair '{pair}' (expected key=value)", file=sys.stderr)
            sys.exit(EXIT_USAGE)
        key, value = pair.split("=", 1)
        # Heuristic: "less" or "lower" in name → greater_is_better=False
        greater = not any(t in key.lower() for t in ("loss", "error", "mse", "mae"))
        try:
            out[key.strip()] = {"threshold": float(value), "greater_is_better": greater}
        except ValueError:
            print(f"Error: threshold value '{value}' not numeric", file=sys.stderr)
            sys.exit(EXIT_USAGE)
    return out


def check_thresholds(metrics: dict[str, float], thresholds: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    """Return (passed, list_of_failures)."""
    failures = []
    for k, spec in thresholds.items():
        v = metrics.get(k)
        if v is None:
            failures.append(f"{k}: metric missing from result")
            continue
        threshold = spec["threshold"]
        greater = spec.get("greater_is_better", True)
        ok = (v >= threshold) if greater else (v <= threshold)
        if not ok:
            failures.append(f"{k}: {v:.4f} {'<' if greater else '>'} threshold {threshold:.4f}")
    return (len(failures) == 0), failures


def send_webhook(url: str, payload: dict[str, Any]) -> None:
    """Best-effort webhook post; never raises."""
    try:
        import requests
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Warning: webhook post failed: {e}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-uri", required=True, help="Model URI to evaluate (typically champion)")
    parser.add_argument("--data", required=True, help="Path to new batch CSV (must include --targets column)")
    parser.add_argument("--targets", required=True, help="Name of the label/target column (skip if no ground truth)")
    parser.add_argument("--model-type", required=True, choices=["classifier", "regressor"])
    parser.add_argument("--experiment-name", default="drift-monitor", help="MLflow experiment to log run under")
    parser.add_argument("--thresholds", dest="thresholds", help="Thresholds: 'metric=value,metric=value' OR JSON")
    parser.add_argument("--alert-webhook", help="Optional URL to POST when drift detected")
    parser.add_argument("--no-evaluate", action="store_true",
                        help="Skip mlflow.models.evaluate; just log dataset summary (use when no ground truth available)")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    args = parser.parse_args()

    import pandas as pd
    import mlflow

    mlflow.set_experiment(args.experiment_name)
    df = pd.read_csv(args.data)
    if args.targets not in df.columns:
        print(f"Error: --targets column '{args.targets}' not in data (columns: {list(df.columns)})", file=sys.stderr)
        return EXIT_USAGE

    dataset = mlflow.data.from_pandas(df, source=args.data, name="prod-batch", targets=args.targets)

    metrics: dict[str, float] = {}
    drift_status = "unknown"
    failures: list[str] = []

    run_name = f"drift-check-{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}-{dataset.digest[:8]}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_input(dataset, context="production")
        mlflow.set_tag("dataset_digest", dataset.digest)
        mlflow.log_dict(df.describe(include="all").fillna("").to_dict(), "batch_summary.json")

        if not args.no_evaluate:
            try:
                result = mlflow.models.evaluate(
                    model=args.model_uri,
                    data=df,
                    targets=args.targets,
                    model_type=args.model_type,
                )
                metrics = {k: float(v) for k, v in result.metrics.items() if isinstance(v, (int, float))}
            except Exception as e:
                print(f"Error: evaluate failed: {str(e)[:500]}", file=sys.stderr)
                return EXIT_BACKEND

            if args.thresholds:
                thresholds = load_thresholds(args.thresholds)
                passed, failures = check_thresholds(metrics, thresholds)
                drift_status = "healthy" if passed else "degraded"
            else:
                drift_status = "monitored"  # logged but no gate

            mlflow.set_tag("drift_status", drift_status)
            for k, v in metrics.items():
                if isinstance(v, float):
                    mlflow.log_metric(f"drift_{k}", v)

    # Report
    if args.format == "json":
        out = {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "dataset_digest": dataset.digest,
            "drift_status": drift_status,
            "metrics": metrics,
            "failures": failures,
        }
        print(json.dumps(out, indent=2, default=str))
    else:
        print(f"Drift check: {drift_status}")
        print(f"  Run: {run.info.run_id}")
        print(f"  Dataset digest: {dataset.digest}")
        if metrics:
            print(f"  Metrics:")
            for k, v in metrics.items():
                print(f"    {k:<28} {v:.4f}")
        if failures:
            print(f"  Failures:")
            for f in failures:
                print(f"    - {f}")

    # Alert
    if drift_status == "degraded" and args.alert_webhook:
        send_webhook(args.alert_webhook, {
            "drift_status": drift_status,
            "run_id": run.info.run_id,
            "dataset_digest": dataset.digest,
            "metrics": metrics,
            "failures": failures,
        })

    return EXIT_HEALTHY if drift_status != "degraded" else EXIT_DEGRADED


if __name__ == "__main__":
    sys.exit(main())