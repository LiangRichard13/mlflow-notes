# Hyperparameter Optimization Reference

Two Optuna integration patterns, sklearn GridSearchCV + autolog, and picking the champion across all experiments via `search_logged_models`.

## Table of Contents
1. [Optuna — Manual Nested Runs](#1-optuna-manual-nested-runs)
2. [Optuna — `MLflowCallback` (Recommended)](#2-optuna-mlflowcallback)
3. [GridSearchCV / RandomizedSearchCV + autolog](#3-gridsearchcv-with-autolog)
4. [Picking the Champion Across Experiments](#4-picking-the-champion)
5. [Aggregate / Compare Trials](#5-aggregate--compare-trials)
6. [Pitfalls](#6-pitfalls)

---

## 1. Optuna — Manual Nested Runs

```python
import optuna
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("wine-tuning")

def objective(trial: optuna.Trial) -> float:
    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }
        mlflow.log_params(params)

        model = RandomForestClassifier(**params, random_state=42)
        score = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro").mean()
        mlflow.log_metric("f1_macro", score)
        return score

with mlflow.start_run(run_name="study-parent") as parent_run:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    mlflow.log_metric("best_f1_macro", study.best_value)
    mlflow.log_params(study.best_params)
    mlflow.set_tag("best_trial_number", study.best_trial.number)

print(f"Best: {study.best_value} with {study.best_params}")
```

The parent run aggregates study-level metadata; each trial gets its own child run under it.

---

## 2. Optuna — `MLflowCallback` (Recommended)

The official `optuna-integration` package provides a callback that handles nested-run creation for you. Less boilerplate, harder to forget tags.

```bash
pip install optuna optuna-integration[mlflow]
```

```python
import optuna
import mlflow
from optuna_integration.mlflow import MLflowCallback

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("wine-tuning")

cb = MLflowCallback(
    tracking_uri=mlflow.get_tracking_uri(),
    create_experiment=False,    # we already created the experiment above
    mlflow_kwargs={"nested": True},
    metric_name="f1_macro",
)

def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
    }
    model = RandomForestClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro").mean()
    trial.set_user_attr("f1_macro", score)   # callback reads this
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, callbacks=[cb])
```

### Which to pick?

| Aspect | Manual nested | MLflowCallback |
|---|---|---|
| Setup | No new dep | `pip install optuna-integration[mlflow]` |
| Trial run metadata | You control every param/tag/log | Callback auto-logs params + user attrs |
| Custom run naming | Full control (e.g. `lr-sweep-trial-7`) | Callback names generically |
| Nested flag handling | Manual | Auto (`mlflow_kwargs={"nested": True}`) |
| Parent aggregation | Manual (you log `best_value` yourself) | Manual still — callback doesn't aggregate |
| Best for | When you need custom trial annotations | Standard tuning flows (less code) |

---

## 3. GridSearchCV + autolog

```python
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

mlflow.sklearn.autolog(log_models=True)   # crucial: GridSearchCV wrapping

with mlflow.start_run(run_name="grid-search") as parent:
    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [5, 10, 20],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=5, scoring="f1_macro", n_jobs=-1,
    )
    grid.fit(X_train, y_train)

# Autolog records:
#   - One parent run with best params, best score
#   - One child run per (param combination, fold)
#   - The best_estimator_ as a logged model with signature
```

### Reading the results

```python
import mlflow
runs = mlflow.search_runs(
    experiment_ids=["1"],
    filter_string=f"tags.`mlflow.parentRunId` = '{parent.info.run_id}'",
    order_by=["metrics.f1_macro DESC"],
)
print(runs[["run_id", "params.n_estimators", "params.max_depth", "metrics.f1_macro"]])

# The logged model in the parent run IS the best_estimator_ — load it directly
```

> ⚠️ Autolog records the **best model** in the parent run, not the search process. To inspect per-fold metrics, drill into child runs.

### Caveats

- Only sklearn-compatible estimators are autolog'd by `mlflow.sklearn.autolog()`.
- `n_jobs > 1` in GridSearchCV can cause MLflow logging races — log inside a single-process wrapper if you see missing child runs.
- For `RandomizedSearchCV`, behavior is identical.

---

## 4. Picking the Champion

After running multiple experiments, the question is "which model from which experiment is the best?". `search_logged_models` answers this across experiments:

```python
import mlflow

# Find the top-10 models across ALL experiments by accuracy
candidates = mlflow.search_logged_models(
    experiment_ids=["1", "2", "3"],           # multiple experiments
    filter_string="metrics.accuracy_score > 0.85",
    order_by=[{"field_name": "metrics.accuracy_score", "ascending": False}],
    max_results=10,
    output_format="list",                     # or "pandas"
)

for i, m in enumerate(candidates, 1):
    print(f"{i}. {m.name} run_id={m.run_id} acc={m.metrics.get('accuracy_score'):.4f}")

best = candidates[0]
print(f"Best model_id: {best.model_id}")

# Promote to champion
from mlflow import MlflowClient
client = MlflowClient()
client.set_registered_model_alias(best.name, "champion", version=...)
# Or load directly:
prod = mlflow.pyfunc.load_model(f"models:/{best.model_id}")
```

The filter syntax is the same as `search_runs` (see `tracking.md` §7).

---

## 5. Aggregate / Compare Trials

### All trials of an Optuna study

```bash
mlflow runs search --experiment-id 1 \
  --filter-string "tags.\`mlflow.parentRunId\` = '<parent_run_id>'" \
  --order-by "metrics.f1_macro DESC" \
  --output json > trials.json
```

### Best N from any sweep

Use `search_logged_models` as above, then optionally register the winner:

```python
from mlflow import MlflowClient

best = candidates[0]
client = MlflowClient()

# If the best model isn't yet registered, register it
client.create_model_version(
    name="WineQualityClassifier",
    source=f"models:/{best.model_id}",
    run_id=best.run_id,
)

# Then point the alias to its version (look it up)
versions = client.search_model_versions(f"name='WineQualityClassifier' and run_id='{best.run_id}'")
client.set_registered_model_alias("WineQualityClassifier", "champion", versions[0].version)
```

### Compare two specific runs

```python
import mlflow
df = mlflow.search_runs(
    experiment_ids=["1"],
    filter_string=f"run_id IN ('{run_a_id}', '{run_b_id}')",
)
print(df[["run_id", "params.n_estimators", "metrics.accuracy_score", "metrics.f1_macro"]])
```

Or use the UI: select two runs → Compare button.

---

## 6. Pitfalls

1. **`MLflowCallback` requires `optuna-integration`**: bare `optuna` won't import `optuna_integration`. Install `optuna-integration[mlflow]`.
2. **Nested runs without a parent**: `start_run(nested=True)` outside an active parent raises. Open the parent first.
3. **`params` filter limitations**: `params.n_estimators > 100` doesn't work (params are strings). Use `=`/`!=`/`LIKE`/`IN`.
4. **Autolog + GridSearchCV with many folds + many params**: child-run explosion. Limit `cv` and `param_grid` for sanity.
5. **Champion swap race conditions**: if two parallel pipelines both set `champion` on the same model, last write wins. Use a CI gate (see `monitor.md` §5) to enforce sequential promotion.
6. **`search_logged_models` requires LoggedModel entities (MLflow 3 only)**: if you used MLflow 2-style `artifact_path`, the model is hidden inside the run's artifacts, not a top-level LoggedModel. Re-log with `name=` to upgrade.
7. **Manual `start_run(nested=True)` doesn't automatically log Optuna trial params**: you must call `mlflow.log_params(trial.params)` yourself inside the trial block.

---

## See also

- `tracking.md` — search_runs filter syntax, MlflowClient
- `registry.md` — `set_registered_model_alias`, LoggedModel loading
- `evaluate.md` — `validate_evaluation_results` for promotion gates
- `monitor.md` — drift check before promoting a new champion