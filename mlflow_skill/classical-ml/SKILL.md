---
name: classical-ml
description: |
  For classical ML / deep learning research projects using scikit-learn, PyTorch, TensorFlow, XGBoost, LightGBM, CatBoost. **Proactively consult** this skill's SKILL.md when you observe ANY of the following:
  
  **Code signals** (auto-detected via cwd/file scan):
  - Training scripts using `from sklearn`, `import torch`, `import xgboost`, `import lightgbm`, `import catboost`, `import tensorflow`
  - Existing MLflow artifacts: `mlflow.db`, `mlruns/` directory, `.env` with MLflow keys, `MLFLOW_TRACKING_URI` set
  - Models already registered: `models:/<name>@<alias>` patterns, Model Registry queries
  
  **User intent signals** (any of these words/phrases in conversation):
  - "train / training", "model", "evaluate", "compare runs", "register model", "champion / challenger"
  - "track my training", "log my model", "MLflow", "autolog"
  - "deploy model", "serve", "build-docker", "batch predict"
  - "monitor", "drift", "system metrics", "dataset lineage"
  - "hyperparameter", "Optuna", "GridSearchCV", "best trial"
  - "model registry", "model version", "alias", "promote model"
  - "metrics", "confusion matrix", "ROC", "AUC"
  
  **What this skill provides** (read SKILL.md for details):
  - 6-step classical ML lifecycle: tracking → registry → evaluate → deploy → monitor → optimize
  - MLflow 3 vs 2 breaking changes (silent failures if ignored)
  - Backend store requirements (file:// breaks Registry)
  - Autolog patterns for 8+ frameworks
  - Champion/challenger alias workflow
  - Deployment via `models serve` / `build-docker` / SageMaker
  - Drift monitoring recipe + scripts in `scripts/`
  
  **CRITICAL behavior**: This skill is **advisory only**. After consulting:
  1. Suggest 1-3 concrete MLflow actions relevant to user's task
  2. Wait for user confirmation before running anything
  3. Never force autolog, registry calls, or any side effects
  4. If user's task is unrelated to MLflow (e.g., pure pandas exploration), don't suggest
  
  **Don't consult** for GenAI/LLM/agent workflows — use those specific skills.
allowed-tools: Read, Write, Bash, Grep, Glob, WebFetch
disable-model-invocation: false
---

# Classical ML Model Lifecycle with MLflow

End-to-end guide for classical ML workflows — covering **tracking, registry, evaluation, deployment, monitoring, and optimization**. Use this when the user wants to debug, evaluate, monitor, or optimize a trained model, or take it through production stages.

For **GenAI / agent / LLM** workflows (tracing, prompt management, LLM-as-judge evaluation), use the **`agent-evaluation`**, **`instrumenting-with-mlflow-tracing`**, or **`querying-mlflow-metrics`** skills instead.

## When to consult this skill (from Agent perspective)

You (the Agent) should consult this skill when you observe **any of these signals** in the user's context:

**Auto-detectable from cwd/filesystem** (use Read/Glob/Bash to scan):
- Training script with `from sklearn` / `import torch` / `import xgboost` / `import lightgbm` / `import catboost` / `import tensorflow`
- Existing MLflow artifacts: `mlflow.db`, `mlruns/` directory
- `MLFLOW_TRACKING_URI` env var set, or `.env` with MLflow keys
- Model files in cwd: `*.pkl`, `*.pt`, `*.h5`, `*.joblib`

**User mentions in conversation** (semantic match):
- "train", "model", "evaluate", "compare runs", "register", "champion"
- "MLflow", "autolog", "tracking", "tracking URI", "experiment"
- "deploy", "serve", "build-docker", "batch predict", "production"
- "monitor", "drift", "dataset lineage", "system metrics"
- "Optuna", "GridSearchCV", "hyperparameter"

**DON'T consult** for: pure LLM/agent work, pandas exploration, general data engineering, web apps, etc.

### How to consult

1. Read `SKILL.md` (this file) for the 6-step overview
2. Read relevant `references/*.md` (only the ones needed, not all)
3. Suggest **1-3 concrete actions** to the user, with exact code/commands
4. **Wait for confirmation** before executing anything
5. **Don't dump the whole skill** — be selective based on user's actual task

### Example interactions

| User says | You should do |
|-----------|--------------|
| "Train an XGBoost model" | Read tracking.md + optimize.md, suggest `mlflow.xgboost.autolog()` + prompt Registry setup |
| "Compare my 2 runs" | Suggest `search_logged_models.py` or `mlflow runs search` with specific filter |
| "How do I deploy?" | Read deploy.md, suggest `mlflow models serve` first, `build-docker` for prod |
| "XGBoost training is slow" | Don't consult MLflow (perf tuning, not tracking). Suggest `Optuna` only if user mentions tuning |
| User has `mlflow.db` in cwd but no autolog | Read tracking.md, suggest adding `mlflow.<framework>.autolog()` to training script |

## ⛔ CRITICAL: Must Use MLflow 3 APIs and Pick the Right Backend

**MLflow 2 → 3 breaking changes** (every one of these will silently produce wrong artifacts if you ignore them):

| MLflow 2 (deprecated) | MLflow 3 (must use) |
|---|---|
| `mlflow.sklearn.log_model(model, artifact_path="m")` | `mlflow.sklearn.log_model(model, name="m")` |
| `transition_model_version_stage(..., "Production")` | `client.set_registered_model_alias(name, "champion", version)` |
| `mlflow.evaluate(..., custom_metrics=[fn])` | `mlflow.models.evaluate(..., extra_metrics=[make_metric(eval_fn=fn, ...)])` |
| `mlflow.evaluate(..., baseline_model=uri)` | Two `models.evaluate` calls + `mlflow.validate_evaluation_results(...)` with `MetricThreshold` |
| Model lives inside a run (`runs:/<id>/<path>`) | Model is a first-class **LoggedModel** (`models:/<model_id>`, `search_logged_models`) |
| Tracking server with no `--allowed-hosts` | **MLflow 3.5+ requires** `--allowed-hosts` (DNS-rebinding defense) |

**Backend store must be DB-backed for Registry to work**:

| URI | Tracking | Registry |
|---|---|---|
| `sqlite:///<file>` | ✓ | ✓ (run `mlflow db upgrade` once first) |
| `postgresql://...`, `mysql://...` | ✓ | ✓ |
| `http(s)://<server>` | ✓ | ✓ |
| `file:./mlruns` (default) | ✓ | ✗ — `register_model` raises |
| `./mlruns` (default bare path) | ✓ | ✗ — same as `file:` |

> If you skipped `set_tracking_uri` and a script silently wrote to `./mlruns`, downstream `register_model` calls will fail with `MlflowException`. Verify the URI matches the one used at log time.

**Pre-flight**:

```bash
export MLFLOW_TRACKING_URI="sqlite:///$(pwd)/mlflow.db"   # or postgresql://...
mlflow db upgrade "$MLFLOW_TRACKING_URI"                  # once per DB
python ../mlflow_skill/classical-ml/scripts/validate_environment.py --check-deps
```

**Documentation protocol**: classical ML topics are **NOT** in the `llms.txt` index (which covers only GenAI). Fetch directly from `https://mlflow.org/docs/latest/ml/...`.

---

## Command Conventions

**Run all scripts through the same Python env that has `mlflow` installed**:

```bash
# Use the skill's helper scripts
python classical-ml/scripts/validate_environment.py --tracking-uri "$MLFLOW_TRACKING_URI"
python classical-ml/scripts/search_logged_models.py --experiment-ids 1 --filter "metrics.accuracy > 0.9"

# Capture CLI output to file (Bash ~30KB stdout limit; piping silently drops large payloads)
mlflow runs search --experiment-id 1 --output json > /tmp/runs.json
jq '.[] | .info.run_id' /tmp/runs.json

# Separate stderr when capturing for parsing
uv run mlflow experiments get --experiment-id 1 --output json > /tmp/exp.json 2> /tmp/exp.log
```

**Working directory matters**: sqlite with a relative path resolves from CWD. Always `cd` to your project root before running scripts.

---

## Determine Your Entry Point

| If the user wants to... | Enter at |
|---|---|
| Log a training run, autolog, set params/metrics | **Step 1 Tracking** |
| Register a model, set champion/challenger alias, load from registry | **Step 2 Registry** |
| Evaluate a model, compute metrics, compare two versions | **Step 3 Evaluate** |
| Serve / batch-predict / containerize / deploy to a target | **Step 4 Deploy** |
| Track which data trained a run, monitor system resources, detect drift | **Step 5 Monitor** |
| Tune hyperparameters, run a sweep, pick the best across experiments | **Step 6 Optimize** |
| Something is wrong and they don't know where | Run `scripts/validate_environment.py` first, then read `references/troubleshooting.md` |
| Just getting started with MLflow (any use case) | `mlflow-onboarding` skill |

---

## Step 1: Tracking

**Goal**: capture every training run with params, metrics, model artifact, and lineage.

### 1a. Environment

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")        # see CRITICAL for backend type
mlflow.set_experiment("wine-classifier-v3")
```

### 1b. Autolog (one line replaces 20+ lines of manual logging)

```python
mlflow.sklearn.autolog()                             # sklearn
mlflow.xgboost.autolog()                             # XGBoost
mlflow.lightgbm.autolog()                            # LightGBM
mlflow.catboost.autolog()                            # CatBoost
mlflow.pytorch.autolog()                             # PyTorch (Lightning only)
mlflow.tensorflow.autolog()                          # TF/Keras
mlflow.statsmodels.autolog()                         # statsmodels
mlflow.spark.autolog()                               # pyspark.ml
mlflow.autolog()                                      # generic dispatcher

# Per-call overrides
mlflow.sklearn.autolog(log_model_signatures=True, extra_tags={"team": "ml-ops"})
```

Confirm your framework is in `references/tracking.md` §3 before assuming autolog exists.

### 1c. Manual logging + model with signature

```python
from mlflow.models import infer_signature

with mlflow.start_run(run_name="rf-baseline") as run:
    mlflow.log_params({"n_estimators": 300, "max_depth": 10})
    for epoch in range(10):
        mlflow.log_metric("train_loss", loss_e, step=epoch)
    mlflow.set_tag("stage", "baseline")

    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        sk_model=model, name="wine-classifier",     # MLflow 3: `name=` not `artifact_path=`
        signature=signature, input_example=X_train.head(3),
    )
```

### 1d. Find and inspect runs

```python
df = mlflow.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.accuracy > 0.9 AND params.model = 'rf'",
    order_by=["metrics.accuracy DESC"], max_results=20,
)
```

```bash
mlflow runs search --experiment-id 1 --filter-string "metrics.accuracy > 0.9" --output json > /tmp/runs.json
```

**Checkpoint**:
- [ ] `mlflow.set_tracking_uri` set to DB-backed URI before any script
- [ ] Autolog fires (or manual `log_params`/`log_metric`/`log_model` called)
- [ ] Model logged with `signature` and `input_example`
- [ ] Run visible via `mlflow search_runs` or UI

**Read more**: `references/tracking.md` (full autolog list, search_runs filter syntax, CLI matrix), `references/model-packaging.md` (signature/dependencies/code_paths deep dive).

---

## Step 2: Registry

**Goal**: promote trained models to versions, set aliases, load by alias for production.

### 2a. Register and set alias

```python
from mlflow import MlflowClient
client = MlflowClient()

# Either at log time
mlflow.sklearn.log_model(model, name="wine-classifier",
                         registered_model_name="WineQualityClassifier")

# Or later
result = mlflow.register_model(
    model_uri=f"runs:/{run.info.run_id}/wine-classifier",
    name="WineQualityClassifier",
)
client.set_registered_model_alias("WineQualityClassifier", "champion", version=result.version)
```

### 2b. Load and predict

```python
prod = mlflow.sklearn.load_model("models:/WineQualityClassifier@champion")
preds = prod.predict(X_new)
```

The four URI forms: `runs:/<id>/<path>`, `models:/<name>/<version>`, `models:/<name>@<alias>`, `models:/<model_id>` (MLflow 3 LoggedModel).

**Checkpoint**:
- [ ] Tracking URI is DB-backed (`sqlite://` / `postgresql://` / `http(s)://`)
- [ ] `mlflow db upgrade <uri>` was run at least once on this DB
- [ ] Model registered with `register_model` or `registered_model_name=`
- [ ] `champion` alias set on the version you want in production
- [ ] `load_model("models:/<name>@champion")` returns a working model

**Read more**: `references/registry.md` (Stage→Alias migration, MlflowClient full API, LoggedModel, pitfalls).

---

## Step 3: Evaluate

**Goal**: produce metrics, artifacts (confusion matrix, ROC), and threshold gates for promotion decisions.

### 3a. Built-in evaluation

```python
result = mlflow.models.evaluate(
    model="models:/WineQualityClassifier@champion",   # keyword `model=`, not `model_uri=`
    data=eval_df, targets="quality",
    model_type="classifier",                            # or "regressor"
    evaluator_config={"log_explainer": True},           # SHAP
)
print(result.metrics)                                  # accuracy_score, roc_auc, ...
```

> ⛔ Use `mlflow.models.evaluate` (classical) — NOT `mlflow.genai.evaluate` (GenAI only).

### 3b. Custom metric

```python
from mlflow.models import make_metric
from mlflow.metrics.base import MetricValue

def my_eval(predictions, targets, metrics):
    score = ((predictions == 1) & (targets == 1)).sum() / len(predictions)
    return MetricValue(aggregate_results={"my_accuracy": float(score)})

result = mlflow.models.evaluate(
    model=uri, data=df, targets="y", model_type="classifier",
    extra_metrics=[make_metric(eval_fn=my_eval, greater_is_better=True, name="my_accuracy")],
)
```

`eval_fn` must be **pure** (MLflow parallelizes).

### 3c. Threshold gate (CI/CD use)

```python
from mlflow.models import MetricThreshold
thresholds = {
    "accuracy_score": MetricThreshold(threshold=0.85, greater_is_better=True,
                                      min_absolute_change=0.01),
}
mlflow.validate_evaluation_results(
    validation_thresholds=thresholds,
    candidate_result=result_new,
    baseline_result=result_old,                          # optional; absent → just absolute check
)
```

> ⚠️ `min_absolute_change` must be ≥ 0. Negative values let worse models pass.

Or from CLI:
```bash
python scripts/evaluate_tabular.py --model-uri models:/Wine@champion \
  --data eval.csv --targets quality --model-type classifier \
  --thresholds-json '{"accuracy_score": {"threshold": 0.85, "greater_is_better": true}}'
```

**Checkpoint**:
- [ ] `model_type` matches the actual model (classifier vs regressor)
- [ ] If using a custom metric, `eval_fn` is pure
- [ ] Threshold gate applied before any `set_registered_model_alias("champion", ...)` call
- [ ] Evaluation result run visible in UI with metrics + artifacts

**Read more**: `references/evaluate.md` (three modes, custom metrics, SHAP, pitfalls).

---

## Step 4: Deploy

**Goal**: expose the model as a service (REST, batch, container).

### 4a. Local REST server

```bash
mlflow models serve \
  -m models:/WineQualityClassifier@champion \
  -p 5001 --host 127.0.0.1 --env-manager local
```

Test:
```bash
python scripts/probe_endpoint.py --url http://127.0.0.1:5001/invocations \
  --input eval.json --format dataframe_split
```

Note the payload keys: `dataframe_split` / `dataframe_records` / `instances` / `inputs` (JSON) or `text/csv`. The server validates against your signature.

### 4b. Batch scoring

```bash
mlflow models predict -m models:/Wine@champion -i input.csv -o output.csv -t csv
```

Or in Python:
```python
model = mlflow.pyfunc.load_model("models:/Wine@champion")
preds = model.predict(pd.read_csv("input.csv"))
preds.to_csv("output.csv", index=False)
```

### 4c. Container

```bash
mlflow models build-docker -m models:/Wine@champion -n wine-classifier
docker run -p 5001:8080 wine-classifier
```

For K8s production, consider `models build-docker --enable-mlserver` (Seldon MLServer runtime).

### 4d. Production targets

```bash
mlflow sagemaker deploy -m models:/Wine@champion --region us-east-1
# Azure ML, Modal, etc. via mlflow deployments plugin
```

**Checkpoint**:
- [ ] `models serve` starts and `/health` returns 200
- [ ] Real-shape payload against `/invocations` returns predictions (or 400 with helpful error)
- [ ] If signature mismatch, fix the model — don't bypass the signature
- [ ] Batch predict on a small sample before scaling to full data
- [ ] MLflow 3.5+: tracking server started with `--allowed-hosts`

**Read more**: `references/deploy.md` (all flavors, payload formats, spark_udf, build-docker, deployment targets, pitfalls).

---

## Step 5: Monitor

**Goal**: capture which data trained each run, watch system resources, detect drift in production.

### 5a. Dataset lineage (at training time)

```python
import mlflow.data

train_ds = mlflow.data.from_pandas(X_train, source="s3://wine/train.parquet",
                                   name="wine-train-v3", targets="quality")
mlflow.log_input(train_ds, context="training")
```

Reverse lookup (which data trained this run?):
```python
from mlflow import MlflowClient
run = MlflowClient().get_run(run_id)
for ds_input in run.inputs.dataset_inputs:
    print(ds_input.dataset.name, ds_input.dataset.digest, ds_input.dataset.source)
```

### 5b. System metrics — TWO mechanisms (do not confuse)

**A. MLflow built-in** (`system/*` prefix, server-side daemon):
```python
mlflow.enable_system_metrics_logging()                   # global
with mlflow.start_run(log_system_metrics=True): train()    # per-run
```
Requires `psutil` + (`nvidia-ml-py` for GPU).

**B. User manual** (your own names, in your code):
```python
import psutil
with mlflow.start_run():
    for step in range(100):
        mlflow.log_metric("cpu_percent", psutil.cpu_percent(), step=step)
```

These produce different metric names and live in different tabs. See `references/monitor.md` §3.

### 5c. Drift monitoring recipe

```bash
python scripts/monitor_drift.py \
  --model-uri models:/WineQualityClassifier@champion \
  --data today_batch.csv --targets label --model-type classifier \
  --thresholds "accuracy_score=0.85,roc_auc=0.90" \
  --alert-webhook https://hooks.example.com/drift
```

This evaluates the champion on today's data, tags the run `drift_status=healthy|degraded`, and (if degraded) fires the webhook. Schedule with cron / Airflow / K8s CronJob.

**Checkpoint**:
- [ ] `log_input` called at training time with `context="training"` (and `"testing"` for test set)
- [ ] Built-in system metrics OR manual psutil sampling chosen (not both mixed up)
- [ ] Drift monitor script scheduled (daily / hourly) and alerting tested
- [ ] Drift run queryable: `mlflow.search_runs(filter_string="tags.drift_status = 'degraded'")`

**Read more**: `references/monitor.md` (mlflow.data family, drift recipe, two system-metrics mechanisms, pitfalls).

---

## Step 6: Optimize

**Goal**: search hyperparameter space, pick the best model across experiments, promote it.

### 6a. Pick the best across experiments

```bash
python scripts/search_logged_models.py \
  --experiment-ids 1,2,3 \
  --filter "metrics.accuracy_score > 0.85" \
  --order metrics.accuracy_score:desc \
  --max 10
```

Or in Python:
```python
candidates = mlflow.search_logged_models(
    experiment_ids=["1", "2"], max_results=10, output_format="list",
    filter_string="metrics.accuracy_score > 0.85",
    order_by=[{"field_name": "metrics.accuracy_score", "ascending": False}],
)
best = candidates[0]
prod = mlflow.pyfunc.load_model(f"models:/{best.model_id}")
```

### 6b. Optuna sweep (manual nested runs)

```python
import optuna

with mlflow.start_run(run_name="study-parent") as parent:
    def objective(trial):
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            params = {"lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True)}
            mlflow.log_params(params)
            score = train_eval(params)
            mlflow.log_metric("f1", score)
            return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    mlflow.log_metric("best_f1", study.best_value)
```

### 6c. Optuna via `MLflowCallback` (cleaner)

```python
from optuna_integration.mlflow import MLflowCallback
cb = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(),
                    mlflow_kwargs={"nested": True}, metric_name="f1")
study.optimize(objective, n_trials=50, callbacks=[cb])
```

### 6d. GridSearchCV + autolog

```python
mlflow.sklearn.autolog(log_models=True)
with mlflow.start_run(run_name="grid"):
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X_train, y_train)
# Autolog creates parent + child runs; best_estimator_ is the logged model
```

**Checkpoint**:
- [ ] Nested runs visible in UI under their parent (`tags.mlflow.parentRunId`)
- [ ] Best trial/model identified via `search_logged_models` (cross-experiment)
- [ ] Before promoting, new champion passes `validate_evaluation_results` against old champion
- [ ] If using `optuna-integration`, installed `pip install optuna-integration[mlflow]`

**Read more**: `references/optimize.md` (Optuna both modes, GridSearchCV interpretation, champion selection, pitfalls).

---

## Verification

After walking through any step, verify with:

```bash
# 1. Environment + tracking URI
python scripts/validate_environment.py --check-deps

# 2. Recent runs visible
mlflow runs list --experiment-id 1 --output json > /tmp/runs.json
jq '.[] | .info.run_id' /tmp/runs.json

# 3. Registry content
mlflow models list --filter-string "name LIKE 'Wine%'" --output json

# 4. Drift / quality monitor healthy
mlflow runs search --experiment-names drift-monitor \
  --filter-string "tags.drift_status = 'degraded'" --max-results 5
```

If something is wrong, run `scripts/validate_environment.py` first, then read `references/troubleshooting.md` (6-layer debug workflow).

---

## References

| File | When to load |
|---|---|
| `references/tracking.md` | Step 1 deep dive: autolog list, search_runs syntax, CLI |
| `references/registry.md` | Step 2 deep dive: Stage→Alias, MlflowClient, LoggedModel |
| `references/evaluate.md` | Step 3 deep dive: three modes, custom metrics, SHAP, gates |
| `references/deploy.md` | Step 4 deep dive: flavors, serve params, payload formats, spark_udf |
| `references/monitor.md` | Step 5 deep dive: mlflow.data, two system-metrics mechanisms, drift recipe |
| `references/optimize.md` | Step 6 deep dive: Optuna both modes, GridSearchCV, champion selection |
| `references/model-packaging.md` | Signature, dependencies, code_paths, `models_from_code` |
| `references/ui.md` | UI navigation: compare runs, logged models, dataset lineage |
| `references/troubleshooting.md` | Six-layer debug workflow, pitfall index, symptom → fix |
| `references/mlflow-3-api.md` | MLflow 2→3 migration cheat sheet |

---

## Related Skills

- **`mlflow-onboarding`** — first-time setup and use-case determination
- **`searching-mlflow-docs`** — fetch official docs (note: llms.txt does NOT cover classical ML — use direct `/ml/` URLs)
- **`agent-evaluation`** — for GenAI/LLM agent evaluation (DO NOT confuse with `mlflow.models.evaluate`)
- **`querying-mlflow-metrics`** — aggregated metrics across many traces (GenAI focus; for tabular metrics use `scripts/search_logged_models.py` and `scripts/compare_runs.py`)
- **`instrumenting-with-mlflow-tracing`** — LLM tracing; classical ML training is NOT traced via spans