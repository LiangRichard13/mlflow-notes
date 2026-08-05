#!/usr/bin/env python3
"""Search and rank LoggedModel entities across one or more experiments (MLflow 3+).

This fills a CLI gap: there is no `mlflow models search` command. Use this to find
the best-performing model across experiments, optionally by metric threshold,
ordered by any metric or param.

Usage:
    python scripts/search_logged_models.py --experiment-ids 1,2
    python scripts/search_logged_models.py --experiment-ids 1 --filter "metrics.accuracy > 0.9"
    python scripts/search_logged_models.py --experiment-ids 1 --order metrics.accuracy:desc --max 10
    python scripts/search_logged_models.py --experiment-ids 1 --format json

Exit codes:
    0  Search executed (may be 0 results).
    1  Backend error or invalid arguments.
    2  Usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


def parse_order_by(spec: str) -> dict[str, Any]:
    """Parse 'metrics.accuracy:desc' or 'params.lr:asc' into a search_logged_models order dict."""
    if ":" in spec:
        field, direction = spec.rsplit(":", 1)
        ascending = direction.lower().startswith("asc")
    else:
        field = spec
        ascending = False  # default desc for ranking
    return {"field_name": field, "ascending": ascending}


def search(
    experiment_ids: list[str],
    filter_string: str | None,
    order_by_specs: list[str],
    max_results: int,
) -> list[Any]:
    """Run mlflow.search_logged_models and return list of LoggedModel entities."""
    import mlflow
    order_by = [parse_order_by(s) for s in order_by_specs]
    kwargs: dict[str, Any] = {
        "experiment_ids": experiment_ids,
        "max_results": max_results,
        "output_format": "list",
    }
    if filter_string:
        kwargs["filter_string"] = filter_string
    if order_by:
        kwargs["order_by"] = order_by
    return mlflow.search_logged_models(**kwargs)


def _extract_metric_kv(metric: Any) -> tuple[str | None, Any]:
    """Extract (key, value) from a LoggedModel metric entity, dict, or NamedTuple."""
    # Entity (has .key/.value)
    k = getattr(metric, "key", None)
    v = getattr(metric, "value", None)
    if k is not None:
        return k, v
    # Dict
    if isinstance(metric, dict):
        return metric.get("key"), metric.get("value")
    return None, None


def _extract_param_kv(p: Any) -> tuple[str | None, Any]:
    """Extract (key, value) from a LoggedModel param entity, dict, or NamedTuple."""
    k = getattr(p, "key", None)
    v = getattr(p, "value", None)
    if k is not None:
        return k, v
    if isinstance(p, dict):
        return p.get("key"), p.get("value")
    return None, None


def format_table(models: list[Any]) -> str:
    """Render results as a table."""
    if not models:
        return "  (no models matched)"
    rows = ["  model_id                         name                         source_run_id                   metrics"]
    rows.append("  " + "-" * 120)
    for m in models:
        # LoggedModel.metrics is a list in MLflow 3.15
        pairs = []
        for metric in (m.metrics or []):
            k, v = _extract_metric_kv(metric)
            if k is not None:
                pairs.append((k, v))
        metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in pairs)
        if len(metrics_str) > 60:
            metrics_str = metrics_str[:57] + "..."
        rows.append(f"  {m.model_id:<33} {(m.name or ''):<27} {(m.source_run_id or ''):<32} {metrics_str}")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--experiment-ids", required=True, help="Comma-separated experiment IDs (names not allowed by API)")
    parser.add_argument("--filter", help="Filter string (e.g. 'metrics.accuracy > 0.9')")
    parser.add_argument("--order", dest="order", action="append", default=[],
                        help="Order spec 'field[:asc|desc]'; can be repeated. Default: metrics.accuracy:desc")
    parser.add_argument("--max", type=int, default=10, dest="max_results", help="Max results (default 10)")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    args = parser.parse_args()

    experiment_ids = [e.strip() for e in args.experiment_ids.split(",") if e.strip()]
    if not experiment_ids:
        print("Error: --experiment-ids is required (comma-separated list)", file=sys.stderr)
        return EXIT_USAGE

    # Default order: by accuracy_score descending if available
    order_specs = args.order or ["metrics.accuracy_score:desc"]

    try:
        models = search(experiment_ids, args.filter, order_specs, args.max_results)
    except Exception as e:
        # Translate common errors into friendlier messages
        msg = str(e)
        if "experiment_ids" in msg.lower():
            print("Error: experiment_ids must be numeric IDs, not names. Run 'mlflow experiments search' to find IDs.", file=sys.stderr)
        elif "filter" in msg.lower() or "parse" in msg.lower():
            print(f"Error: invalid filter syntax. Check quoting; see references/tracking.md §7. Detail: {msg[:300]}", file=sys.stderr)
        else:
            print(f"Error: {msg[:500]}", file=sys.stderr)
        return EXIT_FAIL

    if args.format == "json":
        out = []
        for m in models:
            metrics_dict = {}
            for metric in (m.metrics or []):
                k, v = _extract_metric_kv(metric)
                if k is not None:
                    metrics_dict[k] = v
            params_dict = {}
            for p in (m.params or []):
                k, v = _extract_param_kv(p)
                if k is not None:
                    params_dict[k] = v
            out.append({
                "model_id": m.model_id,
                "name": m.name,
                "model_uri": m.model_uri,
                "source_run_id": m.source_run_id,
                "experiment_id": m.experiment_id,
                "model_type": getattr(m, "model_type", None),
                "metrics": metrics_dict,
                "params": params_dict,
            })
        print(json.dumps(out, indent=2, default=str))
    else:
        print(f"Found {len(models)} LoggedModel(s) in experiments {experiment_ids}")
        if args.filter:
            print(f"Filter: {args.filter}")
        print(f"Order: {order_specs}")
        print()
        print(format_table(models))

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())