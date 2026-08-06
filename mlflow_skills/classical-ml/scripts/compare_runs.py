#!/usr/bin/env python3
"""Compare two or more runs side-by-side: aligned metrics, params, and tags.

Prints a clean table; useful for code review or promotion decisions.

Usage:
    python scripts/compare_runs.py --run-ids abc123,def456
    python scripts/compare_runs.py --run-ids abc,def,ghi --metrics accuracy,f1,roc_auc
    python scripts/compare_runs.py --run-ids abc,def --format json

Exit codes:
    0  Comparison complete.
    1  One or more run IDs not found.
    2  Usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

EXIT_OK = 0
EXIT_NOT_FOUND = 1
EXIT_USAGE = 2


def fetch_runs(run_ids: list[str]) -> list[Any]:
    """Fetch Run objects via MlflowClient."""
    from mlflow import MlflowClient
    client = MlflowClient()
    out = []
    for rid in run_ids:
        try:
            run = client.get_run(rid)
            out.append(run)
        except Exception as e:
            print(f"Error fetching run {rid}: {e}", file=sys.stderr)
            sys.exit(EXIT_NOT_FOUND)
    return out


def collect_metrics(runs: list[Any], keys: list[str] | None) -> list[str]:
    """Union of requested + observed metric keys, in stable order."""
    seen: list[str] = []
    if keys:
        seen.extend(keys)
    for r in runs:
        for k in (r.data.metrics or {}).keys():
            if k not in seen:
                seen.append(k)
    return seen


def format_table(runs: list[Any], metric_keys: list[str]) -> str:
    """Aligned table: rows are metrics, columns are runs."""
    if not runs:
        return "  (no runs)"
    headers = ["metric"] + [r.info.run_id[:8] for r in runs]
    rows = [headers]
    for k in metric_keys:
        row = [k]
        for r in runs:
            v = (r.data.metrics or {}).get(k)
            row.append(f"{v:.4f}" if isinstance(v, float) else (str(v) if v is not None else "—"))
        rows.append(row)

    # Compute column widths
    col_widths = [max(len(str(row[i])) for row in rows) for i in range(len(headers))]
    lines = []
    for ri, row in enumerate(rows):
        line = "  " + "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
        lines.append(line)
        if ri == 0:
            lines.append("  " + "  ".join("-" * w for w in col_widths))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-ids", required=True, help="Comma-separated run IDs")
    parser.add_argument("--metrics", help="Comma-separated metric keys to compare (others discovered automatically)")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    args = parser.parse_args()

    run_ids = [r.strip() for r in args.run_ids.split(",") if r.strip()]
    if len(run_ids) < 2:
        print("Error: --run-ids requires at least 2 IDs to compare", file=sys.stderr)
        return EXIT_USAGE

    keys = [m.strip() for m in args.metrics.split(",")] if args.metrics else None
    runs = fetch_runs(run_ids)
    metric_keys = collect_metrics(runs, keys)

    if args.format == "json":
        out = []
        for r in runs:
            out.append({
                "run_id": r.info.run_id,
                "run_name": r.info.run_name,
                "status": r.info.status,
                "start_time": r.info.start_time,
                "end_time": r.info.end_time,
                "params": dict(r.data.params or {}),
                "metrics": dict(r.data.metrics or {}),
                "tags": {k: v for k, v in (r.data.tags or {}).items() if not k.startswith("mlflow.")},
            })
        print(json.dumps(out, indent=2, default=str))
    else:
        # Header
        for r in runs:
            print(f"Run: {r.info.run_id[:8]}  name={r.info.run_name!r}  status={r.info.status}")
        print()
        print("Metrics (side-by-side):")
        print(format_table(runs, metric_keys))

        # Param diffs (only show params that differ)
        all_params: set[str] = set()
        for r in runs:
            all_params.update((r.data.params or {}).keys())
        diffs = {}
        for p in sorted(all_params):
            values = {(r.data.params or {}).get(p, "—") for r in runs}
            if len(values) > 1:
                diffs[p] = [(r.info.run_id[:8], (r.data.params or {}).get(p, "—")) for r in runs]
        if diffs:
            print()
            print("Param differences:")
            for p, vals in diffs.items():
                v_str = ", ".join(f"{rid}={v!r}" for rid, v in vals)
                print(f"  {p}: {v_str}")

    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())