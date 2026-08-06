#!/usr/bin/env python3
"""Pre-flight check: is this environment ready to run a classical MLflow workflow?

Validates Python version, MLflow install, optional deps, tracking URI connectivity,
backend store type, and experiment presence. Reports ✓/⚠/✗ for each item.

Usage:
    python scripts/validate_environment.py
    python scripts/validate_environment.py --tracking-uri sqlite:///mlflow.db --experiment-name wine
    python scripts/validate_environment.py --format json
    python scripts/validate_environment.py --check-deps    # also probe psutil, timm, openai, requests

Exit codes:
    0  All checks passed (warnings allowed).
    1  One or more blocking checks failed (will not work).
    2  Usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# Exit codes documented above; constants referenced by callers
EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


def check_python_version(min_major: int = 3, min_minor: int = 9) -> dict[str, Any]:
    """Verify Python version meets minimum (default 3.9)."""
    import sys as _sys
    v = _sys.version_info
    ok = (v.major, v.minor) >= (min_major, min_minor)
    return {
        "name": "Python version",
        "status": "ok" if ok else "fail",
        "value": f"{v.major}.{v.minor}.{v.micro}",
        "required": f">= {min_major}.{min_minor}",
    }


def check_mlflow_version(min_version: str = "3.0.0") -> dict[str, Any]:
    """Verify mlflow is installed and at minimum version."""
    try:
        import mlflow
        from packaging.version import Version
        ok = Version(mlflow.__version__) >= Version(min_version)
        return {
            "name": "MLflow version",
            "status": "ok" if ok else "fail",
            "value": mlflow.__version__,
            "required": f">= {min_version}",
        }
    except ImportError:
        return {"name": "MLflow version", "status": "fail", "value": "not installed", "required": f">= {min_version}"}


def check_tracking_uri(uri: str | None) -> dict[str, Any]:
    """Probe tracking URI; report backend store type and connectivity."""
    if uri:
        import mlflow
        mlflow.set_tracking_uri(uri)
    import mlflow
    actual_uri = mlflow.get_tracking_uri()
    # Backend type classification
    if actual_uri.startswith(("sqlite://", "postgresql://", "mysql://")):
        backend = "db-backed"
        registry_ok = True
    elif actual_uri.startswith(("http://", "https://")):
        backend = "remote-server"
        registry_ok = True
    elif actual_uri.startswith("file:") or actual_uri == "./mlruns" or "mlruns" in actual_uri:
        backend = "file"
        registry_ok = False
    else:
        backend = "unknown"
        registry_ok = False

    # Connectivity test (only for db-backed or remote)
    connected = True
    if backend != "file":
        try:
            mlflow.search_experiments(max_results=1)
        except Exception as e:
            connected = False
            return {
                "name": "Tracking URI",
                "status": "fail",
                "value": actual_uri,
                "backend": backend,
                "registry_supported": registry_ok,
                "error": str(e)[:200],
            }
    status = "ok" if connected else "fail"
    if not registry_ok:
        status = "warn"  # works for tracking but NOT for Registry
    return {
        "name": "Tracking URI",
        "status": status,
        "value": actual_uri,
        "backend": backend,
        "registry_supported": registry_ok,
    }


def check_experiment(name: str | None) -> dict[str, Any]:
    """Check if a given experiment exists."""
    if not name:
        return {"name": "Experiment", "status": "skip", "value": "(not provided)"}
    import mlflow
    try:
        exp = mlflow.get_experiment_by_name(name)
        if exp is None:
            return {"name": "Experiment", "status": "warn", "value": f"'{name}' does not exist", "action": "set_experiment() will create it"}
        return {"name": "Experiment", "status": "ok", "value": f"'{name}'", "id": exp.experiment_id}
    except Exception as e:
        return {"name": "Experiment", "status": "fail", "value": name, "error": str(e)[:200]}


def check_optional_deps() -> list[dict[str, Any]]:
    """Probe commonly-needed optional dependencies."""
    results = []
    for mod, label in [
        ("psutil", "system metrics (built-in)"),
        ("nvidia_ml_py", "GPU metrics"),
        ("requests", "cross-platform HTTP for serve smoke tests"),
        ("openai", "optional: only if also doing GenAI workflows"),
        ("timm", "PyTorch image models"),
    ]:
        try:
            __import__(mod)
            results.append({"name": f"dep: {mod}", "status": "ok", "purpose": label})
        except ImportError:
            status = "warn" if mod in ("openai", "timm") else "warn"
            results.append({"name": f"dep: {mod}", "status": status, "value": "not installed", "purpose": label})
    return results


def format_table(results: list[dict[str, Any]]) -> str:
    """Format results as aligned table."""
    rows = []
    for r in results:
        icon = {"ok": "✓", "warn": "⚠", "fail": "✗", "skip": "⊘"}.get(r["status"], "?")
        name = r["name"]
        value = r.get("value") or r.get("id") or r.get("backend") or r.get("purpose") or ""
        rows.append(f"  [{icon}] {name:<30} {value}")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tracking-uri", help="Override MLFLOW_TRACKING_URI for this check")
    parser.add_argument("--experiment-name", help="Check that this experiment exists")
    parser.add_argument("--check-deps", action="store_true", help="Also probe optional dependencies")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    results.append(check_python_version())
    results.append(check_mlflow_version())
    results.append(check_tracking_uri(args.tracking_uri))
    if args.experiment_name:
        results.append(check_experiment(args.experiment_name))
    if args.check_deps:
        results.extend(check_optional_deps())

    if args.format == "json":
        print(json.dumps({"results": results, "all_ok": all(r["status"] in ("ok", "warn", "skip") for r in results)}, indent=2))
    else:
        print("MLflow Environment Validation")
        print("=" * 60)
        print(format_table(results))
        print()
        # Summary
        failures = [r for r in results if r["status"] == "fail"]
        warnings = [r for r in results if r["status"] == "warn"]
        print(f"Summary: {len(results)} checks, {len(failures)} failed, {len(warnings)} warnings")
        if failures:
            print("\nFAILURES:")
            for r in failures:
                print(f"  - {r['name']}: {r.get('error', 'see above')}")
        if warnings:
            print("\nWARNINGS:")
            for r in warnings:
                print(f"  - {r['name']}: {r.get('value', r.get('purpose', ''))}")

    return EXIT_OK if not any(r["status"] == "fail" for r in results) else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())