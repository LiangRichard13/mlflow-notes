# Troubleshooting & Debug Workflow

A six-layer debugging methodology for classical ML workflows in MLflow. When something is wrong, work top-down: environment → run → artifact → metrics → logs → model. Each layer has known CLI/Python commands and common pitfalls.

## Table of Contents
1. [Debug Workflow — Six Layers](#1-debug-workflow)
2. [Environment Layer](#2-environment-layer)
3. [Run Layer](#3-run-layer)
4. [Artifact Layer](#4-artifact-layer)
5. [Metrics Layer](#5-metrics-layer)
6. [Logs / System Metrics Layer](#6-logs-layer)
7. [Model Layer](#7-model-layer)
8. [Pitfall Index by Symptom](#8-pitfall-index)
9. [Reference Cross-Walk](#9-reference-cross-walk)

---

## 1. Debug Workflow

```
1. Environment     mlflow --version, tracking URI, backend type
2. Run             mlflow runs describe, search_runs, UI
3. Artifact        mlflow artifacts list, download
4. Metrics         mlflow runs search with metric filters, UI Compare
5. Logs / System   stderr, log files, system metrics tab
6. Model           mlflow.pyfunc.load_model + predict single sample
```

Use this top-down to localize the failure. Most production bugs live at one of layers 1, 3, or 6.

---

## 2. Environment Layer

**Goal**: verify your MLflow install, tracking URI, and backend store are correct.

```bash
mlflow --version                              # expect 3.0+
echo $MLFLOW_TRACKING_URI                     # expect sqlite:///<abs path> or http://...
python -c "import mlflow; print(mlflow.get_tracking_uri())"
python -c "import mlflow; print(mlflow.__version__)"
```

Common failures:

| Symptom | Cause | Fix |
|---|---|---|
| `MlflowException: UnsupportedModelRegistryStoreURIException` | Tracking URI is `file:` | Switch to sqlite/postgres |
| `sqlite3.OperationalError: no such table: registered_models` | First run on fresh DB | `mlflow db upgrade <uri>` |
| Browser cannot connect to tracking UI | MLflow 3.5+ missing `--allowed-hosts` | Restart server with `--allowed-hosts "*"` for local |
| `ImportError: No module named 'mlflow'` | Wrong Python env | `pip install mlflow==3.15` in the right venv |

Helper script: `scripts/validate_environment.py` automates all of the above.

---

## 3. Run Layer

**Goal**: confirm the run exists, has the right params/metrics, and is in the expected state.

```bash
mlflow runs describe --run-id <id> --output json > run.json
jq '.info.status, .info.start_time, .info.end_time, .info.artifact_uri' run.json
jq '.data.params, .data.metrics, .data.tags' run.json

# Find a run by name
mlflow runs list --experiment-id 1 --filter-string "tags.`mlflow.runName` = 'baseline'" --output json

# Restore a deleted run
mlflow runs restore --run-id <id>
```

Python:
```python
from mlflow import MlflowClient
client = MlflowClient()
runs = client.search_runs(experiment_ids=["1"],
                          filter_string="metrics.accuracy > 0.9",
                          order_by=["metrics.accuracy DESC"],
                          max_results=5)
for r in runs:
    print(r.info.run_id, r.info.status, r.data.metrics)
```

Common failures:

| Symptom | Cause | Fix |
|---|---|---|
| Run not appearing in UI | Backend mismatch (logged to file, viewing sqlite) | Verify `MLFLOW_TRACKING_URI` is the same everywhere |
| Status stuck in `RUNNING` | Process killed mid-run | `client.set_terminated(run_id, status="KILLED")` |
| Params empty | Logged before `start_run()` context, or wrong run | Verify `log_param` happens inside the `with` block |

---

## 4. Artifact Layer

**Goal**: confirm the model file, signature, and dependencies are present and correct.

```bash
mlflow artifacts list --run-id <id>                      # top-level files/dirs
mlflow artifacts download --run-id <id> -d ./artifacts   # pull everything

# Pull just the model
mlflow artifacts download --run-id <id> --artifact-path wine-classifier -d ./model

# Inspect MLmodel YAML
cat model/MLmodel
```

Python:
```python
import mlflow.pyfunc
model = mlflow.pyfunc.load_model(f"runs:/<id>/wine-classifier")
preds = model.predict(X_test.head(3))
```

Common failures:

| Symptom | Cause | Fix |
|---|---|---|
| `RESOURCE_DOES_NOT_EXIST` when loading | Path mismatch with `name=` parameter | Check `log_model(name=...)` matches URI path |
| Model loads but predictions are wrong | Feature columns out of order | Check `MLmodel` signature; pass DataFrame in correct order |
| Pickle errors loading custom pyfunc | Class definition drifted across versions | Re-log with `models_from_code` (see `model-packaging.md`) |
| Missing dependency at load time | `requirements.txt` incomplete | `pip install -r requirements.txt`; re-log with explicit deps |

---

## 5. Metrics Layer

**Goal**: find outliers, regressions, and contradictions across runs.

```bash
# Compare top-5 runs in one experiment
mlflow runs search --experiment-id 1 \
  --order-by "metrics.accuracy DESC" --max-results 5 \
  --output json > top.json
jq '.[] | {run_id: .info.run_id, acc: .data.metrics.accuracy_score, f1: .data.metrics.f1_score}' top.json
```

Python (full DataFrame analysis):
```python
import mlflow, pandas as pd
df = mlflow.search_runs(experiment_ids=["1"], max_results=1000)
print(df.groupby("params.model")["metrics.accuracy_score"].agg(["mean", "std", "max"]))
```

UI: select runs → Compare → parallel coordinates plot shows param→metric interactions.

Common failures:

| Symptom | Cause | Fix |
|---|---|---|
| Metric missing from UI but `log_metric` succeeded | Different `step=` values across metrics | Pick consistent step semantics |
| Metrics of `-inf` or `NaN` | Training crashed mid-epoch | Inspect stderr; gate promotions with `MetricThreshold` |
| Comparing runs across experiments shows nothing | Used `experiment_ids` instead of `experiment_names` | `search_runs` requires `experiment_ids`; pass the IDs |

---

## 6. Logs / System Metrics Layer

**Goal**: understand what was happening on the host during training.

```python
import mlflow
with mlflow.start_run(log_system_metrics=True):
    train()

# After training, read system metrics
from mlflow import MlflowClient
client = MlflowClient()
for m in ["system/cpu_utilization_percentage", "system/gpu_memory_usage_megabytes"]:
    hist = client.get_metric_history(run_id, m)
    print(f"{m}: max={max(h.value for h in hist)}, avg={sum(h.value for h in hist)/len(hist):.2f}")
```

UI: Run → Metrics tab → toggle `system/*` metrics.

Common failures:

| Symptom | Cause | Fix |
|---|---|---|
| No `system/*` metrics visible | `enable_system_metrics_logging()` not called, or `psutil` missing | `pip install psutil`; set `MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true` env var |
| GPU metrics empty | `nvidia-ml-py` not installed | `pip install nvidia-ml-py` |
| Custom manually-logged metrics not in System Metrics tab | Missing `system/` prefix (this is OK — they show in regular Metrics tab) | None; just be aware |

---

## 7. Model Layer

**Goal**: verify the model behaves correctly on a single, controlled sample.

```python
import mlflow.pyfunc
import pandas as pd

model = mlflow.pyfunc.load_model("models:/WineQualityClassifier@champion")

# 1. Predict on a known sample
sample = X_test.iloc[[0]]
expected = y_test.iloc[0]
pred = model.predict(sample)
assert pred[0] == expected, f"Expected {expected}, got {pred[0]}"

# 2. If a custom PythonModel, inspect load_context errors
#    (re-instantiate with same artifacts dict)

# 3. If model predictions are NaN, check feature preprocessing
import numpy as np
assert not np.any(np.isnan(model.predict(X_test))), "Predictions contain NaN"
```

For serve debugging:
```bash
# Test the same model URI your serve command uses
mlflow models predict -m models:/WineQualityClassifier@champion \
  -i input.csv -o output.csv -t csv

# If serve fails to start but predict works, the problem is the env
mlflow models serve --env-manager local -m ...
```

Common failures:

| Symptom | Cause | Fix |
|---|---|
| Model returns same class for all inputs | Feature scaling missing or wrong | Verify preprocessing is inside `predict()` |
| Predictions are 0.5 for everything | Sigmoid/logit bug; wrong output type | Check `predict()` returns the right type |
| Custom pyfunc raises at `load_context` | Missing artifact path | Pass `artifacts={"key": "relative/path"}` to `log_model` |
| `mlflow.pyfunc.spark_udf` returns wrong shape | Spark column ordering | Pass features as a struct column |

---

## 8. Pitfall Index

A-Z quick lookup. Each entry links to the section above or to the dedicated reference.

| Symptom | Section |
|---|---|
| `artifact_path` deprecation warning | `mlflow-3-api.md` §1 |
| Cannot load by `runs:/...` | §4, `tracking.md` §10 |
| Champion swap race | `optimize.md` §6 |
| Confusion matrix is tiny in UI | `evaluate.md` §3 |
| `dataset_inputs` empty in run | `monitor.md` §2 |
| `db upgrade` not run on fresh DB | §2, `registry.md` §10 |
| Drift run explodes storage | `monitor.md` §6 |
| `enable_system_metrics_logging` not logging | §6, `monitor.md` §3 |
| `extra_metrics` not in `result.metrics` | `evaluate.md` §6 |
| `file:` backend + Registry | `tracking.md` §1, §10 |
| GridSearchCV child runs missing | `optimize.md` §3 |
| `--allowed-hosts` missing on 3.5+ | §2, `mlflow-3-api.md` §2 |
| `log_param` same key twice | `tracking.md` §10 |
| Manual psutil metrics not in System tab | §6, `monitor.md` §3 |
| `mlflow.evaluate` deprecated | `mlflow-3-api.md` §1, `evaluate.md` §1 |
| `mlflow.runs compare` doesn't exist | `tracking.md` §10, `deploy.md` §10 |
| `models serve` fails to start | §7, `deploy.md` §10 |
| `OR` not supported in filter_string | `tracking.md` §7 |
| Pickle drift on custom pyfunc | §7, `model-packaging.md` §6 |
| `psutil.cpu_percent()` returns 0 | `monitor.md` §6 |
| Serve 400: column not found | §7, `deploy.md` §10 |
| Stage API removed | `mlflow-3-api.md` §1, `registry.md` §10 |
| System metrics tab empty | §6, `monitor.md` §3 |

---

## 9. Reference Cross-Walk

| Topic | Reference |
|---|---|
| Tracking URI / autolog / search_runs syntax | `tracking.md` |
| MLflow 3 API changes (full) | `mlflow-3-api.md` |
| Model flavors, signatures, dependencies, code_paths | `model-packaging.md` |
| Model Registry, aliases, LoggedModel | `registry.md` |
| `mlflow.models.evaluate`, `make_metric`, `MetricThreshold`, SHAP | `evaluate.md` |
| Serve, batch predict, containerize | `deploy.md` |
| Dataset lineage, system metrics, drift monitoring | `monitor.md` |
| Optuna / GridSearchCV / champion selection | `optimize.md` |
| UI navigation | `ui.md` |