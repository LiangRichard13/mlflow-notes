# Tracking Reference

Complete reference for classical ML experiment tracking: tracking URIs, experiments, runs, autolog, manual logging, model logging with signatures, dataset logging, run search syntax, and CLI.

## Table of Contents
1. [Tracking URI & Backend Store](#1-tracking-uri--backend-store)
2. [Experiments & Runs](#2-experiments--runs)
3. [Autolog — Full Framework List](#3-autolog--full-framework-list)
4. [Manual Logging (Params/Metrics/Artifacts/Tags)](#4-manual-logging)
5. [Model Logging + Signature](#5-model-logging--signature)
6. [Dataset Logging (`mlflow.data` & `log_input`)](#6-dataset-logging)
7. [`search_runs` Filter Syntax (MLflow 3)](#7-search_runs-filter-syntax-mlflow-3)
8. [`MlflowClient` API Surface](#8-mlflowclient-api-surface)
9. [CLI — Runs / Experiments / Artifacts](#9-cli)
10. [Pitfalls](#10-pitfalls)

---

## 1. Tracking URI & Backend Store

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")          # local file DB
mlflow.set_tracking_uri("postgresql://user:pw@host/db") # production
mlflow.set_tracking_uri("http://localhost:5000")        # tracking server
mlflow.set_tracking_uri("file:./mlruns")                # legacy file-only (no Registry)
```

**Backend type rules** (this is where many production deployments fail):

| Backend URI prefix | Registry support | Notes |
|---|---|---|
| `sqlite:///<file>` | ✅ Yes | First time: `mlflow db upgrade sqlite:///mlflow.db` |
| `postgresql://...` | ✅ Yes | Production-grade; supports concurrent writes |
| `mysql://...` | ✅ Yes | Community plugin |
| `http(s)://<server>` | ✅ Yes | When pointing at a running tracking server |
| `file:<path>` | ❌ No | **No Model Registry.** Will raise `MlflowException` if you call `register_model`. |
| `./mlruns` (default) | ❌ No | Same as `file:`; only suitable for local toy runs |

> ⛔ **CRITICAL**: Before any registry call, ensure your URI is **DB-backed** AND you've run `mlflow db upgrade <uri>` at least once. See `registry.md` for details.

Pre-flight check (run from project root):
```bash
echo "MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI"
cd /path/to/project   # sqlite relative path needs project root
```

---

## 2. Experiments & Runs

```python
mlflow.set_experiment("wine-classifier-v3")  # creates if missing, returns Experiment
exp = mlflow.get_experiment_by_name("wine-classifier-v3")
print(exp.experiment_id, exp.artifact_location)
```

Run lifecycle (always use the context manager — never manually `start_run` without `end_run`):
```python
with mlflow.start_run(run_name="rf-baseline") as run:
    print(run.info.run_id)  # immediately available
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", 0.92)
    mlflow.set_tag("stage", "baseline")
# auto-end_run() on context exit
```

Nested runs (for sweeps / Optuna trials — see `optimize.md`):
```python
with mlflow.start_run(run_name="study-parent") as parent:
    with mlflow.start_run(run_name="trial-1", nested=True):
        mlflow.log_param("lr", 0.01)
```

---

## 3. Autolog — Full Framework List

> Before reaching for manual logging, **check whether your framework has an autolog call**. If yes, one line replaces 20+ lines of `log_param`/`log_metric`.

| Framework | Autolog call | What it captures | Caveats |
|---|---|---|---|
| **scikit-learn** | `mlflow.sklearn.autolog()` | params (incl. defaults), metrics, model with signature, input example, feature importance, GridSearchCV parent+child | Best classical ML coverage |
| **XGBoost** | `mlflow.xgboost.autolog()` | params, importance plot, model | Both `xgboost` and `xgboost.spark` flavors |
| **LightGBM** | `mlflow.lightgbm.autolog()` | params, importance, model | |
| **CatBoost** | `mlflow.catboost.autolog()` | params, model | |
| **PyTorch (Lightning)** | `mlflow.pytorch.autolog()` | model checkpoint, metrics, system metrics | **Only supports PyTorch Lightning**, not raw `torch` |
| **TensorFlow / Keras** | `mlflow.tensorflow.autolog()` | model checkpoint, metrics, callback hooks | Disable with `disable=True` to opt out per-call |
| **Keras (standalone)** | `mlflow.keras.autolog()` | same as TF | alias |
| **statsmodels** | `mlflow.statsmodels.autolog()` | params, summary, model | |
| **Prophet** | `mlflow.prophet.autolog()` | model, forecast info | |
| **pmdarima** | `mlflow.pmdarima.autolog()` | order params, model | |
| **Spark MLlib (pyspark.ml)** | `mlflow.spark.autolog()` | pipeline stages, model, params | Requires SparkSession |
| **fastai** | `mlflow.fastai.autolog()` | metrics, model | Removed in MLflow 3.0+ in some channels; verify your version |
| **Generic** | `mlflow.autolog()` | detects which framework is in use and routes | One call covers everything if you don't care which one fires |

Common options (any autolog):
```python
mlflow.autolog(
    log_model_signatures=True,   # default False for some flavors; set True in production
    log_models=True,            # default True
    log_input_examples=True,     # default True for most
    extra_tags={"team": "ml-ops"},  # custom tags applied to every run
    disable_for_unsupported_versions=True,
)
```

Disable selectively:
```python
mlflow.sklearn.autolog(disable=True)
```

---

## 4. Manual Logging

When autolog doesn't fit (custom training loops, framework unsupported, you want full control):

```python
with mlflow.start_run(run_name="custom-loop"):
    # params — one-time values, same key rejected on second call
    mlflow.log_params({"lr": 0.01, "batch_size": 32})

    # metrics — supports `step` for time-series, `timestamp` for custom time
    for epoch in range(10):
        mlflow.log_metric("train_loss", loss_epoch, step=epoch)
        mlflow.log_metric("val_acc", acc_epoch, step=epoch)

    # artifacts — single file or whole directory
    mlflow.log_artifact("confusion_matrix.png", artifact_path="plots")
    mlflow.log_artifacts("./outputs", artifact_path="batch_outputs")

    # images / tables / dicts (rich objects)
    mlflow.log_image(img_array, "sample.png")
    mlflow.log_table(df, "predictions.parquet")    # since 2.10
    mlflow.log_dict({"config": cfg}, "config.json")

    # tags — mutable; use for status, git SHA, environment
    mlflow.set_tags({"git_sha": sha, "env": "staging"})
```

> ⚠️ **`log_param` is one-shot**: a second `log_param("lr", 0.005)` in the same run raises or warns. If a value may change across iterations, use `set_tag` instead.

---

## 5. Model Logging + Signature

```python
from mlflow.models import infer_signature

# 1. infer signature from training data and predictions
signature = infer_signature(X_train, model.predict(X_train))

# 2. log the model with signature + input example (auto-used at serve time)
mlflow.sklearn.log_model(
    sk_model=model,
    name="wine-classifier",                  # MLflow 3: `name=` not `artifact_path=`
    signature=signature,
    input_example=X_train.head(3),
    registered_model_name=None,              # or set to register immediately
)
```

Signature modes:
- **Inferred from data** (most common): `infer_signature(X, y)` or `infer_signature(X, y, params)`
- **Explicit**: build `ModelSignature(inputs=Schema([...]), outputs=Schema([...]), params=ParamSchema([...]))`
- **No signature**: `mlflow.sklearn.log_model(sk_model, name="m")` — server-side inference will accept any shape (dangerous in prod)

See `model-packaging.md` for full signature/dependencies/code-paths detail.

---

## 6. Dataset Logging

```python
import mlflow.data
from mlflow.data import from_pandas

dataset = from_pandas(
    df=X_train,
    source="data/wine.csv",          # any URI; helps lineage
    name="wine-train-v1",
    targets="quality",               # column name; tells UI which is the label
)
mlflow.log_input(dataset, context="training")  # context: training/validation/testing/production
```

Reverse lookup (find which data trained this run):
```python
from mlflow import MlflowClient
client = MlflowClient()
run = client.get_run(run_id)
for ds_input in run.inputs.dataset_inputs:
    ds = ds_input.dataset
    ctx = next((t.value for t in ds_input.tags if t.key == "mlflow.dataset_context"), None)
    print(ds.name, ds.digest, ds.source, ctx)
```

See `monitor.md` for the full `mlflow.data` family (numpy/spark/delta/custom).

---

## 7. `search_runs` Filter Syntax (MLflow 3)

```python
df = mlflow.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.accuracy > 0.9 AND params.model = 'rf'",
    order_by=["metrics.accuracy DESC"],
    max_results=100,
    output_format="pandas",   # or "list"
)
```

**Prefixes you can filter on**: `metrics.`, `params.`, `tags.`, `attributes.`, `datasets.`, `inputs.`

**Operators**:
- Numeric (`metrics.`, `attributes.`): `= != > >= < <=`
- String (`params.`, `tags.`, `datasets.`): `= != LIKE ILIKE RLIKE IN NOT IN`
- Logical: `AND`, `NOT`, parentheses
- Existence: `IS NULL`, `IS NOT NULL`

**Quoting rules** (these trip people up constantly):
- String values: **double quotes** preferred (`metrics.accuracy > 0.9 AND params.model = "rf"`)
- Keys with dots/special chars: **backticks** (`` tags.`mlflow.runName` = "x" ``)
- Outer shell: single quotes for bash, double quotes for Python
- IN / NOT IN: supported on params, metrics, tags, datasets — **not restricted to datasets/attributes** (a common misconception)
- Same `filter_string` does NOT support OR — write two `search_runs` calls and `pd.concat` results

**`params` are stored as strings**:
- ✅ `params.n_estimators = "100"`, `params.n_estimators != "50"`
- ✅ `params.n_estimators LIKE "10%"`, `params.n_estimators IN ("100", "200")`
- ❌ `params.n_estimators > 50` — silently fails or coerces; numeric comparisons only work on `metrics.`

**Examples**:
```bash
# Recent runs with high accuracy
mlflow runs search --experiment-id 1 \
  --filter-string "metrics.accuracy > 0.9 AND attributes.status = 'FINISHED'" \
  --order-by "metrics.accuracy DESC" --max-results 20 --output json > runs.json

# Slow runs in last hour
mlflow runs search --experiment-id 1 \
  --filter-string "trace.execution_time_ms > 1000 AND trace.timestamp_ms > $(( $(date +%s)000 - 3600000 ))"

# Failed runs by tag
mlflow runs search --experiment-id 1 --filter-string "tags.`mlflow.runName` = 'baseline' AND attributes.status = 'FAILED'"

# All child runs of a parent
mlflow runs search --experiment-id 1 --filter-string "tags.`mlflow.parentRunId` = '<parent_run_id>'"
```

---

## 8. `MlflowClient` API Surface

```python
from mlflow import MlflowClient
client = MlflowClient()

# Runs
client.search_runs(experiment_ids=["1"], max_results=50, order_by=["metrics.acc DESC"])
client.get_run(run_id)
client.delete_run(run_id)              # soft-delete; runs are restorable
client.restore_run(run_id)
client.set_terminated(run_id, status="FINISHED")  # mark KILLED/FAILED if needed

# Experiments
client.create_experiment(name, artifact_location=None, tags=None)
client.search_experiments(filter_string="name LIKE 'wine%'")
client.delete_experiment(exp_id)
client.restore_experiment(exp_id)
```

For registry / logged-model operations, see `registry.md`.

---

## 9. CLI

```bash
# Server / UI
mlflow ui --port 5000 [--backend-store-uri <uri>] [--allowed-hosts "*"]   # MLflow 3.5+ needs --allowed-hosts
mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri postgresql://... \
    --default-artifact-root s3://bucket/path

# Experiments
mlflow experiments create --experiment-name foo
mlflow experiments search --view-type active_only --output json
mlflow experiments get --experiment-id 1 --output json | jq '.experiment_id, .tags'

# Runs
mlflow runs list --experiment-id 1 --view-type active_only --output json
mlflow runs describe --run-id <id> --output json
mlflow runs delete --run-id <id>
mlflow runs restore --run-id <id>
mlflow runs search --experiment-id 1 --filter-string "metrics.acc > 0.9"

# Artifacts
mlflow artifacts list --run-id <id>
mlflow artifacts download --run-id <id> -d ./artifacts
mlflow artifacts log-artifact --run-id <id> --local-path foo.txt

# DB upgrade (one-time per DB)
mlflow db upgrade sqlite:///mlflow.db
```

> Always redirect CLI output to a file then `jq` (Bash 30KB stdout limit; piping silently drops large payloads).
> ```bash
> mlflow runs search --experiment-id 1 --output json > runs.json
> jq '.[] | .info.run_id' runs.json
> ```

---

## 10. Pitfalls

1. **`file:` backend silently breaks Registry**: `register_model` raises `MlflowException`. Switch to sqlite/postgres first.
2. **First run on a fresh sqlite DB → Registry fails**: run `mlflow db upgrade sqlite:///mlflow.db` once before any `register_model` call.
3. **`log_param` same key twice**: warn or raise; use `set_tag` if value will change.
4. **`metrics.X > N` works; `params.X > N` does not**: params are strings; use `=`/`!=`/`LIKE`/`IN`.
5. **Tracking URI not set**: defaults to `./mlruns` (file:). Use absolute `sqlite:///<abs path>` to be safe.
6. **CLI output piped to `jq`**: silently truncated past ~30KB. Always `> file && jq < file`.
7. **MLflow 3.5+ server without `--allowed-hosts`**: browser fails to connect (DNS rebinding defense). Pass `--allowed-hosts "*"` for local dev or your actual hostnames.
8. **`search_runs` order_by without index**: large experiments can be slow; ensure experiment_ids is scoped.
9. **Filter using `OR`**: same `filter_string` rejects OR. Use two calls + concat, or pre-filter in pandas.

---

## See also

- `registry.md` — registering the model you just tracked, alias lifecycle
- `model-packaging.md` — deep dive on signature, dependencies, code_paths
- `troubleshooting.md` — "my run isn't appearing", "metrics are wrong", etc.
- `mlflow-3-api.md` — what changed from MLflow 2