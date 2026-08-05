# Monitoring Reference

MLflow doesn't ship a "data drift dashboard" — instead it gives you the primitives (dataset lineage + digest + system metrics + threshold gates) to build one. This reference covers both **MLflow built-in mechanisms** and a **production drift monitoring recipe** you can deploy.

## Table of Contents
1. [`mlflow.data` Sub-APIs Quick Reference](#1-mlflowdata-sub-apis)
2. [Dataset Lineage — `log_input` and Reverse Lookup](#2-dataset-lineage)
3. [System Metrics — Two Distinct Mechanisms](#3-system-metrics)
4. [Drift Monitoring Recipe](#4-drift-monitoring-recipe)
5. [Combining Drift with Threshold Gates](#5-combining-drift-with-threshold-gates)
6. [Pitfalls](#6-pitfalls)

---

## 1. `mlflow.data` Sub-APIs

| Source type | Sub-API | Required extra dep |
|---|---|---|
| Pandas DataFrame | `mlflow.data.from_pandas(df, source, name, targets)` | (core) |
| NumPy array | `mlflow.data.from_numpy(arr, source, name)` | (core) |
| Spark DataFrame | `mlflow.data.from_spark(df_spark, source, name)` | pyspark |
| Polars DataFrame | `mlflow.data.from_polars(df, source, name)` | polars |
| Hugging Face Dataset | `mlflow.data.from_huggingface(hf_ds, source, name)` | datasets |
| TensorFlow Dataset | `mlflow.data.from_tensorflow(tf_ds, source, name)` | tensorflow |
| Delta Lake table | `mlflow.data.from_delta(delta_table, source, name)` | delta-spark |
| Custom (any) | subclass `mlflow.data.Dataset`, implement `load()` + `digest` + `source` + `profile` + `schema` | — |

```python
import mlflow.data

# pandas
ds = mlflow.data.from_pandas(df, source="s3://bucket/wine.csv", name="wine-v3", targets="quality")

# delta (requires delta-spark)
from delta.tables import DeltaTable
ds = mlflow.data.from_delta(DeltaTable.forPath(spark, "/mnt/delta/wine"), source="delta://wine@v2")

# custom
class MyDataset(mlflow.data.Dataset):
    def __init__(self, path):
        self.path = path
        from mlflow.data.pandas_dataset import PandasDataset
        super().__init__(source=path, name="my-set")
    def load(self):
        return pd.read_parquet(self.path)
    @property
    def digest(self):
        import hashlib
        return hashlib.md5(open(self.path, "rb").read()).hexdigest()
```

`Dataset` properties you can read after `from_*`:
- `dataset.name`, `dataset.digest` (content hash), `dataset.source`, `dataset.schema`, `dataset.profile`
- `dataset.tags` (set at log time via context)

---

## 2. Dataset Lineage

Log at training time:
```python
with mlflow.start_run() as run:
    train_ds = mlflow.data.from_pandas(X_train, source="s3://wine/train.parquet",
                                       name="wine-train-v3", targets="quality")
    mlflow.log_input(train_ds, context="training")
    test_ds = mlflow.data.from_pandas(X_test, source="s3://wine/test.parquet",
                                      name="wine-test-v3", targets="quality")
    mlflow.log_input(test_ds, context="testing")
```

Reverse lookup from a run (which data trained this model?):
```python
from mlflow import MlflowClient
client = MlflowClient()
run = client.get_run(run_id)
for ds_input in run.inputs.dataset_inputs:
    ds = ds_input.dataset
    ctx_tag = next((t.value for t in ds_input.tags
                    if t.key == "mlflow.dataset_context"), None)
    print(f"name={ds.name} digest={ds.digest} ctx={ctx_tag} source={ds.source}")

# Re-load the data from the recorded source
src = mlflow.data.get_source(ds)
df_again = src.load()    # uses mlflow.data.get_source under the hood
```

`digest` is a content hash — if the underlying data changes, digest changes, and you can flag "this run was trained on different data than the current prod set."

---

## 3. System Metrics — Two Distinct Mechanisms

> ⚠️ Many tutorials conflate these. They are **two different mechanisms** producing metrics with different names and storage paths. Mixing them up is a common cause of "where did my metrics go?" confusion.

### A. MLflow built-in `enable_system_metrics_logging` (server-side daemon)

The tracking server / autolog spawns a background thread that samples every N seconds and logs metrics to the active run.

```python
import mlflow

mlflow.enable_system_metrics_logging()   # global, persists across runs
# OR per-run:
with mlflow.start_run(log_system_metrics=True):
    train()

# Tuning
mlflow.set_system_metrics_sampling_interval(5)        # seconds between samples
mlflow.set_system_metrics_samples_before_logging(3)    # aggregate N samples before write
```

Metric names (all prefixed `system/`):
- `system/cpu_utilization_percentage`
- `system/system_memory_usage_megabytes`
- `system/system_memory_usage_percentage`
- `system/gpu_utilization_percentage` (needs `nvidia-ml-py`)
- `system/gpu_memory_usage_megabytes` (needs `nvidia-ml-py`)
- `system/gpu_power_usage_watts`
- `system/network_receive_megabytes`
- `system/network_transmit_megabytes`
- `system/disk_usage_megabytes`
- `system/disk_available_megabytes`

Dependencies: `psutil` (CPU/memory/disk/net), `nvidia-ml-py` or `pyrsmi` (GPU). View with `mlflow ui` → Run → Metrics tab.

### B. User-driven manual sampling with `psutil` + `log_metric`

You sample inside your own training loop and log yourself. Useful when you need finer control, custom metrics, or system metrics outside an active run.

```python
import psutil, mlflow, time

with mlflow.start_run():
    for step in range(100):
        cpu = psutil.cpu_percent(interval=None)     # must be called once before to set baseline
        mem = psutil.virtual_memory().percent
        gpu = ...
        mlflow.log_metric("cpu_percent", cpu, step=step)
        mlflow.log_metric("mem_percent", mem, step=step)
        # ... train ...
        time.sleep(0.1)
```

Metric names here are whatever you choose — convention is no `system/` prefix (you'd want to differentiate built-in from custom in the UI).

### When to use which?

| Use case | Recommendation |
|---|---|
| Default CPU/memory/GPU per-run dashboard | A. Built-in (`log_system_metrics=True` or global enable) |
| Custom metric (e.g., specific GPU power, network on a named iface) | B. Manual sampling |
| Sampling when no run is active (long-running process) | B. Manual, with `mlflow.log_metric` outside `start_run` won't persist — use `MlflowClient().log_metric(run_id, ...)` |
| Air-gapped env without `psutil` | Manual psutil sample (you control import) |

---

## 4. Drift Monitoring Recipe

MLflow has no "drift dashboard" — here's the canonical production pattern:

```
┌──────────────────┐
│ Cron / Airflow   │
│ every 1h         │
└────────┬─────────┘
         │ load production batch (CSV/Parquet/Delta)
         ▼
   mlflow.data.from_pandas(...)         ← log as "production" context
         │
         ▼
   mlflow.models.evaluate(...)          ← evaluate on prod batch
         │
         ▼
   MetricThreshold + validate_evaluation_results
         │
         ▼
   mlflow.set_tag("drift_status", "degraded"|"healthy")   ← tag the run
         │
         ▼
   mlflow.search_runs(filter_string="tags.drift_status = 'degraded'")   ← alert
```

### Step-by-step

```python
# monitor_drift.py — run on cron / schedule
import mlflow
import pandas as pd
from mlflow.models import MetricThreshold

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("drift-monitor")

# 1. Load today's production batch
prod_df = pd.read_parquet("s3://prod-bucket/inference/today.parquet")
prod_dataset = mlflow.data.from_pandas(
    prod_df, source="s3://prod-bucket/inference/today.parquet",
    name="prod-batch-today", targets="true_label",  # if ground truth available
)

# 2. Evaluate current champion on it
with mlflow.start_run(run_name=f"drift-check-{prod_dataset.digest[:8]}") as run:
    mlflow.log_input(prod_dataset, context="production")

    result = mlflow.models.evaluate(
        model="models:/WineQualityClassifier@champion",
        data=prod_df,
        targets="true_label",
        model_type="classifier",
    )

    # 3. Apply thresholds (drift detection = performance drift on prod data)
    thresholds = {
        "accuracy_score": MetricThreshold(threshold=0.85, greater_is_better=True),
        "roc_auc":        MetricThreshold(threshold=0.90, greater_is_better=True),
    }
    try:
        mlflow.validate_evaluation_results(thresholds=thresholds, candidate_result=result)
        mlflow.set_tag("drift_status", "healthy")
    except mlflow.models.evaluation.ModelValidationFailedException:
        mlflow.set_tag("drift_status", "degraded")

    # 4. Log distributional summary for forensics
    mlflow.log_dict(prod_df.describe().to_dict(), "prod_batch_summary.json")
    mlflow.set_tag("dataset_digest", prod_dataset.digest)
```

### Alerting layer

```python
# alert_runner.py — separate cron
import mlflow
degraded = mlflow.search_runs(
    experiment_names=["drift-monitor"],
    filter_string="tags.drift_status = 'degraded'",
    max_results=10,
)
if not degraded.empty:
    send_pager_alert(degraded[["run_id", "metrics.accuracy_score", "start_time"]])
```

---

## 5. Combining Drift with Threshold Gates

For CI/CD-style regression tests:

```python
# ci_eval.py — runs before promoting a new model version
import mlflow

# 1. Compare candidate vs current champion on a holdout set
candidate_result = mlflow.models.evaluate(
    "models:/WineQualityClassifier@candidate", data=holdout_df,
    targets="quality", model_type="classifier",
)
champion_result = mlflow.models.evaluate(
    "models:/WineQualityClassifier@champion", data=holdout_df,
    targets="quality", model_type="classifier",
)

# 2. Block promotion if candidate is worse, OR if both fail absolute thresholds
thresholds = {
    "accuracy_score": MetricThreshold(
        threshold=0.85, greater_is_better=True,
        min_absolute_change=0.01,    # candidate must be ≥1% better than champion
    ),
}
try:
    mlflow.validate_evaluation_results(thresholds=thresholds,
                                       candidate_result=candidate_result,
                                       baseline_result=champion_result)
    promote_to_champion("WineQualityClassifier", new_version)
except mlflow.models.evaluation.ModelValidationFailedException as e:
    print(f"Promotion blocked: {e}")
```

---

## 6. Pitfalls

1. **No `system/` prefix on manually-logged metrics**: if you `log_metric("cpu_percent", ...)`, it won't show up under the System Metrics tab in UI. Use the `system/` prefix to align with the built-in convention (or accept the visual distinction).
2. **`psutil.cpu_percent()` first call returns 0**: must be called once to set the baseline before subsequent calls return meaningful values. Built-in MLflow system metrics handles this automatically; manual sampling must too.
3. **`from_delta` requires delta-spark**: easy to forget. `from_pandas` is the safest default.
4. **Drift run explodes storage**: if you log every hourly drift check as a full run, you'll accumulate runs fast. Consider:
   - Only log when `drift_status = degraded` (gate the write)
   - Or shorten retention with `mlflow gc --older-than 30d`
5. **Ground truth not always available**: `targets=` in `evaluate` requires labels. For unsupervised drift, use a custom `make_metric` that computes PSI/KS-test on feature distributions.
6. **Threshold gates require both metrics exist**: if `accuracy_score` isn't in `result.metrics` (e.g., regression model with no accuracy), the threshold check raises. Inspect `result.metrics.keys()` first.
7. **No `mlflow.data.Dataset.load()` for custom subclasses that don't override `load()`**: subclass and implement, or use `from_pandas` etc. that already implement it.
8. **Dataset digest is content-hash**: two snapshots of "the same" data with one new row produce different digests. If your definition of "drift" is "data changed at all", digest comparison works; if you want "data changed enough to matter", use distributional metrics instead.

---

## See also

- `tracking.md` — manual `log_metric`, `log_input`, run search
- `evaluate.md` — `mlflow.models.evaluate`, `MetricThreshold`, `validate_evaluation_results`
- `troubleshooting.md` — "missing system metrics", "drift run not appearing"
- `mlflow-3-api.md` — `log_input` context values and Dataset class evolution