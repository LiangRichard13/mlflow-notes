#!/usr/bin/env python3
"""Run mlflow.models.evaluate on any logged/registered model from the CLI.

Wraps the `mlflow.models.evaluate` Python API so you can evaluate a model on
a tabular CSV without writing a script. Useful for CI checks and ad-hoc
quality gates.

Usage:
    # Basic
    python scripts/evaluate_tabular.py \\
        --model-uri models:/WineQualityClassifier@champion \\
        --data eval.csv --targets quality --model-type classifier

    # With custom metric module + threshold gate
    python scripts/evaluate_tabular.py \\
        --model-uri models:/Wine@champion --data eval.csv \\
        --targets quality --model-type classifier \\
        --extra-metrics-module mypkg.custom_metrics \\
        --thresholds-json '{"accuracy_score": {"threshold": 0.85, "greater_is_better": true}}'

    # Compare candidate vs baseline (enforces min_absolute_change)
    python scripts/evaluate_tabular.py \\
        --model-uri models:/Wine@v5 --baseline-uri models:/Wine@champion \\
        --data eval.csv --targets quality --model-type classifier

Exit codes:
    0  Evaluation succeeded and thresholds passed.
    1  Threshold gate failed (ModelValidationFailedException).
    2  Usage error.
    3  Backend error.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any

EXIT_OK = 0
EXIT_THRESHOLD_FAILED = 1
EXIT_USAGE = 2
EXIT_BACKEND = 3


def load_extra_metrics(module_spec: str) -> list:
    """Import 'pkg.mod' or 'pkg.mod:attr_list' and return list of metric callables.

    Examples:
        mypkg.custom_metrics             → imports module, returns ALL public callables ending in '_metric'
        mypkg.custom_metrics:CUSTOM      → returns [module.CUSTOM]
        mypkg.custom_metrics:m1,m2       → returns [module.m1, module.m2]
    """
    if ":" in module_spec:
        mod_name, names = module_spec.split(":", 1)
        mod = importlib.import_module(mod_name)
        return [getattr(mod, n.strip()) for n in names.split(",") if n.strip()]
    mod = importlib.import_module(module_spec)
    return [v for k, v in vars(mod).items() if k.endswith("_metric") and callable(v)]


def parse_thresholds(json_str: str) -> dict[str, Any]:
    """Parse --thresholds-json into {metric_name: MetricThreshold(...)}."""
    from mlflow.models import MetricThreshold
    raw = json.loads(json_str)
    return {
        k: MetricThreshold(
            threshold=v["threshold"],
            greater_is_better=v.get("greater_is_better", True),
            min_absolute_change=v.get("min_absolute_change", 0.0),
            min_relative_change=v.get("min_relative_change"),
        )
        for k, v in raw.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-uri", required=True, help="Model URI (models:/, runs:/, local path)")
    parser.add_argument("--data", required=True, help="Path to evaluation CSV (must include --targets column)")
    parser.add_argument("--targets", required=True, help="Name of the label/target column")
    parser.add_argument("--model-type", required=True, choices=["classifier", "regressor"])
    parser.add_argument("--baseline-uri", help="Optional baseline URI for gate comparison")
    parser.add_argument("--extra-metrics-module", help="Python module path to load extra metrics (see docstring)")
    parser.add_argument("--thresholds-json", help="JSON dict of {metric_name: {threshold, greater_is_better?, min_absolute_change?}}")
    parser.add_argument("--log-explainer", action="store_true", help="Compute SHAP feature importance")
    parser.add_argument("--evaluator-config-json", help="JSON dict of evaluator_config overrides")
    parser.add_argument("--experiment-name", default="evaluation-runs", help="Experiment to log the eval run under")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    args = parser.parse_args()

    import pandas as pd
    import mlflow
    mlflow.set_experiment(args.experiment_name)

    df = pd.read_csv(args.data)
    if args.targets not in df.columns:
        print(f"Error: --targets column '{args.targets}' not in data (columns: {list(df.columns)})", file=sys.stderr)
        return EXIT_USAGE

    extra_metrics = []
    if args.extra_metrics_module:
        try:
            extra_metrics = load_extra_metrics(args.extra_metrics_module)
        except Exception as e:
            print(f"Error loading extra metrics: {e}", file=sys.stderr)
            return EXIT_USAGE

    evaluator_config = {}
    if args.log_explainer:
        evaluator_config["log_explainer"] = True
    if args.evaluator_config_json:
        evaluator_config.update(json.loads(args.evaluator_config_json))

    kwargs: dict[str, Any] = {
        "model": args.model_uri,
        "data": df,
        "targets": args.targets,
        "model_type": args.model_type,
        "evaluators": "default",
    }
    if extra_metrics:
        kwargs["extra_metrics"] = extra_metrics
    if evaluator_config:
        kwargs["evaluator_config"] = evaluator_config

    try:
        candidate_result = mlflow.models.evaluate(**kwargs)
    except Exception as e:
        print(f"Error: mlflow.models.evaluate failed: {str(e)[:500]}", file=sys.stderr)
        return EXIT_BACKEND

    baseline_result = None
    if args.baseline_uri:
        try:
            baseline_kwargs = {**kwargs, "model": args.baseline_uri}
            baseline_result = mlflow.models.evaluate(**baseline_kwargs)
        except Exception as e:
            print(f"Error evaluating baseline: {str(e)[:500]}", file=sys.stderr)
            return EXIT_BACKEND

    if args.format == "json":
        out = {
            "metrics": dict(candidate_result.metrics),
            "artifact_paths": list(candidate_result.artifacts.keys()) if hasattr(candidate_result, "artifacts") else [],
        }
        if baseline_result:
            out["baseline_metrics"] = dict(baseline_result.metrics)
        print(json.dumps(out, indent=2, default=str))
    else:
        print(f"✓ Evaluated {args.model_uri} on {len(df)} rows")
        print(f"  Experiment: {args.experiment_name}")
        print()
        print(f"  Metric                  Value")
        print(f"  " + "-" * 50)
        for k, v in candidate_result.metrics.items():
            print(f"  {k:<24} {v:.4f}" if isinstance(v, float) else f"  {k:<24} {v}")
        if baseline_result:
            print()
            print("  Baseline:")
            for k, v in baseline_result.metrics.items():
                print(f"  {k:<24} {v:.4f}" if isinstance(v, float) else f"  {k:<24} {v}")

    if args.thresholds_json:
        thresholds = parse_thresholds(args.thresholds_json)
        try:
            mlflow.validate_evaluation_results(
                validation_thresholds=thresholds,
                candidate_result=candidate_result,
                baseline_result=baseline_result,
            )
            print("\n✓ All thresholds passed.")
            return EXIT_OK
        except mlflow.models.evaluation.ModelValidationFailedException as e:
            print(f"\n✗ Threshold gate failed: {e}", file=sys.stderr)
            return EXIT_THRESHOLD_FAILED

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())