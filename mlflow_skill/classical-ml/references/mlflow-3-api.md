# MLflow 3 Breaking Changes & Migration

Quick reference for what changed from MLflow 2.x to 3.x, organized by version. If you're working with an older codebase or older tutorials, this is the cheat sheet.

## Table of Contents
1. [MLflow 3.0 — Core API Rewrite](#1-mlflow-30)
2. [MLflow 3.5+ — Security & Server](#2-mlflow-35)
3. [Removed Features](#3-removed-features)
4. [Migration Checklist](#4-migration-checklist)

---

## 1. MLflow 3.0

### `log_model(artifact_path=)` → `log_model(name=)`

```python
# MLflow 2
mlflow.sklearn.log_model(model, artifact_path="wine")

# MLflow 3 (current)
mlflow.sklearn.log_model(model, name="wine")
```

If you see `DeprecationWarning: artifact_path is deprecated`, switch to `name=`.

### Stage API removed → Aliases

```python
# MLflow 2 (deprecated, may still work with warning)
client.transition_model_version_stage(name, version, "Production")

# MLflow 3 (canonical)
client.set_registered_model_alias(name, "champion", version)
```

Aliases are many-to-many (a version can have multiple aliases; an alias can move freely), whereas Stages were a fixed enum (None/Staging/Production/Archived).

### `mlflow.evaluate(custom_metrics=)` → `models.evaluate(extra_metrics=)`

```python
# MLflow 2
mlflow.evaluate(model, data, targets, custom_metrics=[fn])

# MLflow 3
mlflow.models.evaluate(model, data, targets, extra_metrics=[make_metric(eval_fn=fn, ...)])
```

### `mlflow.evaluate(baseline_model=...)` → `validate_evaluation_results(...)`

```python
# MLflow 2
result = mlflow.evaluate(model, data, targets, baseline_model=baseline_model_uri)

# MLflow 3 (separated concerns)
result_new = mlflow.models.evaluate(new_model_uri, data, targets)
result_old = mlflow.models.evaluate(old_model_uri, data, targets)
thresholds = {"accuracy_score": MetricThreshold(threshold=0.85, greater_is_better=True)}
mlflow.validate_evaluation_results(validation_thresholds=thresholds,
                                   candidate_result=result_new,
                                   baseline_result=result_old)
```

### Models as first-class entities (`LoggedModel`)

In MLflow 3, a logged model is a separate entity from its run:
- URI: `models:/<model_id>` (new) instead of `runs:/<run_id>/<artifact_path>` (still works)
- Search: `mlflow.search_logged_models(...)` instead of grepping runs
- UI: dedicated "Logged Models" page alongside runs

```python
with mlflow.start_run():
    info = mlflow.sklearn.log_model(model, name="wine")
    print(info.model_id)   # new attribute
```

### Recipes / Pipelines removed

The `mlflow.recipes` / `mlflow.pipelines` modules are gone. Use the standard tracking/registry/deploy flow described in this skill's main steps.

---

## 2. MLflow 3.5+

### `--allowed-hosts` required on tracking server

Browser-side defense against DNS rebinding. Without it, the UI fails to load.

```bash
mlflow server \
  --host 0.0.0.0 --port 5000 \
  --backend-store-uri postgresql://... \
  --default-artifact-root s3://... \
  --allowed-hosts "*"               # local dev; restrict in prod
```

### Enhanced security defaults

- Strict CORS
- Stricter cookie settings
- `MLFLOW_FLASK_SERVER_KEY` for signed sessions (when deployed on shared infra)

---

## 3. Removed Features

| Feature | Removed in | Replacement |
|---|---|---|
| `mlflow.pipelines` / `mlflow.recipes` | 3.0 | Use tracking/registry directly |
| `mlflow.fastai` autolog | 3.0+ in some channels | Wrap with custom `pyfunc.PythonModel` |
| `mlflow.gluon` flavor | 3.0 | pyfunc + `code_paths` |
| `mlflow.diviner` flavor | 3.0 | pyfunc |
| `mlflow.mleap` flavor | 3.0 | Use native ONNX or pyfunc instead |
| Legacy deployment server (`mlflow deployments`) CLI for built-in targets | replaced by plugin-based `mlflow deployments` for custom targets | Use `mlflow models serve` / `mlflow models build-docker` |
| `MlflowSignatureDict` legacy class | 3.0 | Use `mlflow.models.signature.ModelSignature` |
| `example_no_conversion`, `code_path`, `inference_config` params in `log_model` | mostly consolidated | Use `code_paths=[...]` (list) |

---

## 4. Migration Checklist

```text
[ ] Switch every log_model(artifact_path=...) to log_model(name=...)
[ ] Replace transition_model_version_stage calls with set_registered_model_alias
[ ] Rewrite mlflow.evaluate to mlflow.models.evaluate with extra_metrics=[make_metric(...)]
[ ] If you had baseline_model= on mlflow.evaluate, split into two evaluate calls + validate_evaluation_results
[ ] Update tutorials copy-pasted from 2022-era blog posts
[ ] Run mlflow db upgrade <uri> on every DB-backed backend after upgrade
[ ] If you use models_from_code (2.18+), audit code_paths= vs the older code_path=
[ ] If you serve mlflow via a reverse proxy, add --allowed-hosts "yourhost.example.com"
[ ] Verify search_logged_models works (re-log any models using artifact_path= to upgrade them)
[ ] Audit dependencies: psutil + nvidia-ml-py for system metrics
[ ] If you depended on mlflow.pipelines, port to plain tracking/registry code
```

---

## See also

- `tracking.md` — full tracking reference
- `registry.md` — Stage→Alias migration, `set_registered_model_alias`
- `evaluate.md` — full evaluate reference, including `validate_evaluation_results`
- `troubleshooting.md` — symptoms of missed migrations