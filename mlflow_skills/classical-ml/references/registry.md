# Model Registry Reference

Register trained models, version them, set aliases (champion/challenger), load by URI. Covers MLflow 3 changes: Stage API is deprecated, aliases are the canonical promotion mechanism, LoggedModel is a first-class entity.

## Table of Contents
1. [MLflow 3 Model on Disk — MLmodel YAML](#1-mlflow-3-model-on-disk)
2. [`log_model` (MLflow 3 `name=` Parameter)](#2-log_model-mlflow-3-name-parameter)
3. [`register_model` — Programmatic Registration](#3-register_model)
4. [Aliases — The Replacement for Stages](#4-aliases)
5. [Loading URIs — Four Forms](#5-loading-uris)
6. [Version Metadata, Tags, Description](#6-version-metadata)
7. [Delete and Archive](#7-delete-and-archive)
8. [`LoggedModel` — First-Class Model Entity (MLflow 3)](#8-loggedmodel)
9. [CLI — Registry Operations](#9-cli)
10. [Pitfalls](#10-pitfalls)

---

## 1. MLflow 3 Model on Disk

Every logged model is a directory containing `MLmodel` (YAML) plus flavor-specific payloads and dependency files.

```
my-model/
├── MLmodel                        # REQUIRED: flavors, signature, run_id, mlflow_version
├── model.pkl                      # sklearn: pickled model (or pickle for XGBoost)
├── python_env.yaml                # MLflow 2.x: conda + pip requirements
├── requirements.txt               # MLflow 3: preferred for pip-only envs
├── conda.yaml                     # legacy; optional
├── input_example.json             # sample input (auto-generated if input_example= given)
├── serving_input_example.json     # transformed for serving
├── environment_variables.txt      # env vars captured during log_model
└── code/                          # when code_paths= passed
    └── predict.py
```

`MLmodel` excerpt:
```yaml
flavors:
  sklearn:
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: 1.5.1
signature:
  inputs: '[{"name": "fixed_acidity", "type": "double"}, ...]'
  outputs: '[{"type": "long"}]'
run_id: abc123def456
mlflow_version: 3.15.1
model_uuid: ...
utc_time_created: '2025-08-05T...'
```

See `model-packaging.md` for details on signatures, dependencies, and `models-from-code`.

---

## 2. `log_model` (MLflow 3 `name=` Parameter)

```python
# MLflow 3: use `name=`, NOT `artifact_path=` (deprecated)
mlflow.sklearn.log_model(
    sk_model=model,
    name="wine-classifier",
    signature=signature,
    input_example=X_train.head(3),
    registered_model_name="WineQualityClassifier",   # register immediately (optional)
    code_paths=["src/"],                            # for custom code dep
)
```

Flavor-specific calls (each has the same `name=`, `signature=`, `input_example=` API):
- `mlflow.sklearn.log_model(...)`
- `mlflow.xgboost.log_model(...)`
- `mlflow.lightgbm.log_model(...)`
- `mlflow.catboost.log_model(...)`
- `mlflow.pytorch.log_model(...)` (note: requires pickled state + model class accessible)
- `mlflow.onnx.log_model(...)`
- `mlflow.pyfunc.log_model(python_model=MyModel(), name=...)` — for custom wrappers
- `mlflow.transformers.log_model(...)` — for HF transformers

After `log_model`, retrieve the model URI:
```python
model_uri = f"runs:/{run.info.run_id}/wine-classifier"
```

---

## 3. `register_model` — Programmatic Registration

Two-step pattern (when you want to inspect a run before registering):
```python
# 1. log the model (don't auto-register)
mlflow.sklearn.log_model(model, name="wine-classifier", signature=signature)
# 2. register from that URI
result = mlflow.register_model(
    model_uri=f"runs:/{run.info.run_id}/wine-classifier",
    name="WineQualityClassifier",
)
print(f"Registered version {result.version}")
```

Or one-step (set `registered_model_name=` on `log_model`) — see above.

---

## 4. Aliases — The Replacement for Stages

> ⛔ **`transition_model_version_stage` is deprecated in MLflow 3.** Don't write new code against it. Old tutorials showing "Staging → Production" are obsolete.

```python
from mlflow import MlflowClient

client = MlflowClient()

# Promote version 3 to "champion"
client.set_registered_model_alias(
    name="WineQualityClassifier",
    alias="champion",
    version=3,
)

# Look up which version is champion
v = client.get_model_version_by_alias("WineQualityClassifier", "champion")
print(v.version, v.status, v.creation_timestamp)

# Demote: move champion to version 5
client.set_registered_model_alias("WineQualityClassifier", "champion", version=5)

# Compare against a challenger
client.set_registered_model_alias("WineQualityClassifier", "challenger", version=4)
```

**Common alias conventions**:
- `champion` — current production
- `challenger` — running A/B alongside champion
- `baseline` — frozen reference for regression tests
- `archived` — old champion, kept for audit

Multiple aliases per version are allowed (e.g., a model can be both `champion` and `baseline`).

---

## 5. Loading URIs — Four Forms

| URI form | Use case |
|---|---|
| `runs:/<run_id>/<artifact_path>` | Load from a specific run (debugging, before registry exists) |
| `models:/<name>/<version>` | Load a specific version of a registered model |
| `models:/<name>@<alias>` | Load whatever version currently has the alias (production pattern) |
| `models:/<model_id>` | MLflow 3: load by LoggedModel ID (returned by `search_logged_models`) |

```python
import mlflow.pyfunc

# Form 1 — direct run
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/wine-classifier")

# Form 2 — by version
model = mlflow.pyfunc.load_model("models:/WineQualityClassifier/3")

# Form 3 — by alias (most common in production)
model = mlflow.pyfunc.load_model("models:/WineQualityClassifier@champion")

# Form 4 — by LoggedModel ID
model = mlflow.pyfunc.load_model(f"models:/{logged_model.model_id}")
preds = model.predict(X_new)
```

> ⚠️ The path component (e.g. `wine-classifier`) **must match** the `name=` passed to `log_model`. Typos cause `RESOURCE_DOES_NOT_EXIST` with confusing messages.

---

## 6. Version Metadata

```python
client = MlflowClient()

# Description (visible in UI)
client.update_model_version(
    name="WineQualityClassifier",
    version=3,
    description="Trained on 2025-08-05, n_estimators=300, accuracy=0.94. Approved by Alice.",
)

# Tags (free-form key-value)
client.set_model_version_tag(name, version, key="validation_status", value="approved")
client.delete_model_version_tag(name, version, key="validation_status")
```

Common tags: `validation_status`, `git_sha`, `deployed_by`, `data_digest`.

---

## 7. Delete and Archive

```python
client = MlflowClient()

# Delete a version (soft; can be restored)
client.delete_model_version(name="WineQualityClassifier", version=2)

# Delete the whole registered model (and all versions)
client.delete_registered_model(name="WineQualityClassifier")

# Restore
client.restore_model_version(name="WineQualityClassifier", version=2)
client.restore_registered_model(name="WineQualityClassifier")
```

There is no first-class "archive" — alias management + deletion covers all use cases.

---

## 8. `LoggedModel` — First-Class Model Entity (MLflow 3)

In MLflow 3, models are no longer "owned by" a run. They are first-class entities with their own IDs and lifecycle. The UI shows them in a separate "Logged Models" page.

```python
# After log_model, retrieve the LoggedModel
import mlflow
from mlflow import MlflowClient

with mlflow.start_run() as run:
    info = mlflow.sklearn.log_model(model, name="wine-classifier")
    print(info.model_uri, info.model_id)

# Search logged models across an experiment
logged_models = mlflow.search_logged_models(
    experiment_ids=["1"],
    filter_string="metrics.accuracy > 0.9",
    order_by=[{"field_name": "metrics.accuracy", "ascending": False}],
    max_results=10,
    output_format="list",   # or "pandas"
)

best = logged_models[0]
print(best.model_id, best.metrics["accuracy"])
prod = mlflow.pyfunc.load_model(f"models:/{best.model_id}")
```

Use this for "pick best model across all experiments" — there is no equivalent CLI.

---

## 9. CLI — Registry Operations

```bash
# Search registered models
mlflow models list --filter-string "name LIKE 'Wine%'"

# List versions of one model
mlflow models list-model-versions --name "WineQualityClassifier" --output json

# Get version details (incl. aliases, tags, status)
mlflow models get-model-version-details --name "WineQualityClassifier" --version 3 --output json
# or by stage (deprecated):
mlflow models get-model-version-details --name "WineQualityClassifier" --stage Production --output json

# Delete / restore
mlflow models delete-model-version --name "WineQualityClassifier" --version 2
mlflow models restore-model-version --name "WineQualityClassifier" --version 2

# ⚠️ There is NO `mlflow models set-alias` CLI in current MLflow.
# Use the Python API for aliases.
```

---

## 10. Pitfalls

1. **Backend must be DB-backed.** `file:` URI → `register_model` raises `MlflowException`. Switch to sqlite/postgres.
2. **First use on a fresh sqlite DB**: run `mlflow db upgrade sqlite:///mlflow.db` **once** before any registry call. Symptom: `RESOURCE_DOES_NOT_EXIST` even though you see runs in the UI.
3. **`artifact_path=` is MLflow 2.** MLflow 3 uses `name=`. Old tutorials will silently produce the wrong artifact layout.
4. **Stage API is gone.** Don't use `transition_model_version_stage` for new work; use aliases.
5. **Alias case-sensitive, must not be purely numeric.** Use `champion` (not `Champion` or `1`).
6. **Loading by version vs alias**: in prod, use alias (`@champion`); for reproducibility/audit, use exact version (`/3`).
7. **`set_registered_model_alias` is atomic**: no race when swapping champion from v3 → v5.
8. **`models:/<name>@<alias>` requires the serving environment to have the same `MLFLOW_TRACKING_URI`** as the registration environment. If serving can't reach your tracking server, you'll get `RESOURCE_DOES_NOT_EXIST`.
9. **Deleting a registered model cascades to all versions** — you can't undo after `delete_registered_model` in some configurations. Move to `archived` alias first.

---

## See also

- `tracking.md` — how to log the model before registering it
- `model-packaging.md` — signature, dependencies, code_paths
- `troubleshooting.md` — registry error → cause → fix
- `mlflow-3-api.md` — Stage→Alias migration, `artifact_path`→`name` migration