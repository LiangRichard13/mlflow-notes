# Deployment Reference

Serve classical ML models locally, in containers, batch-scoring, or to production targets (K8s, SageMaker, Azure, Modal). Covers MLflow 3 flavors, `models serve`, request payload formats, and the `models_from_code` pattern that avoids pickle.

## Table of Contents
1. [Built-in Flavors Quick Reference](#1-built-in-flavors)
2. [Serve Locally with `mlflow models serve`](#2-serve-locally)
3. [Request Payload Formats](#3-request-payload-formats)
4. [Batch Scoring with `models predict` & `pyfunc.predict`](#4-batch-scoring)
5. [Custom PythonModel with `pyfunc.log_model`](#5-custom-pythonmodel)
6. [`models_from_code` (Avoid Pickle)](#6-models-from-code)
7. [Containerize with `build-docker`](#7-containerize)
8. [Deployment Targets](#8-deployment-targets)
9. [CLI Matrix](#9-cli-matrix)
10. [Pitfalls](#10-pitfalls)

---

## 1. Built-in Flavors

| Flavor | `log_model` / `save_model` | `load_model` | Notes |
|---|---|---|---|
| `python_function` (pyfunc) | `mlflow.pyfunc.log_model(python_model=...)` | `mlflow.pyfunc.load_model` | Universal interface; wraps any model |
| `sklearn` | `mlflow.sklearn.log_model(model, name=...)` | `mlflow.sklearn.load_model` | Most common classical ML |
| `xgboost` | `mlflow.xgboost.log_model(...)` | `mlflow.xgboost.load_model` | Booster / sklearn API |
| `lightgbm` | `mlflow.lightgbm.log_model(...)` | `mlflow.lightgbm.load_model` | |
| `catboost` | `mlflow.catboost.log_model(...)` | `mlflow.catboost.load_model` | |
| `onnx` | `mlflow.onnx.log_model(...)` | `mlflow.onnx.load_model` | Cross-framework inference |
| `pytorch` | `mlflow.pytorch.log_model(...)` | `mlflow.pytorch.load_model` | Loads state_dict + class |
| `keras` / `tensorflow` | `mlflow.keras.log_model(...)` | `mlflow.keras.load_model` | |
| `transformers` | `mlflow.transformers.log_model(...)` | `mlflow.transformers.load_model` | For HF models used as classifiers |
| `sentence_transformers` | `mlflow.sentence_transformers.log_model(...)` | `mlflow.sentence_transformers.load_model` | Embedding models |
| `spacy` | `mlflow.spacy.log_model(...)` | `mlflow.spacy.load_model` | |
| `prophet` | `mlflow.prophet.log_model(...)` | `mlflow.prophet.load_model` | Time-series forecasting |
| `statsmodels` | `mlflow.statsmodels.log_model(...)` | `mlflow.statsmodels.load_model` | |
| `pmdarima` | `mlflow.pmdarima.log_model(...)` | `mlflow.pmdarima.load_model` | |
| `pyspark.ml` | `mlflow.spark.log_model(...)` | `mlflow.spark.load_model` | Requires SparkSession |
| `h2o` | `mlflow.h2o.log_model(...)` | `mlflow.h2o.load_model` | |
| `models_from_code` | `mlflow.pyfunc.log_model(python_model=MyClass, code_paths=[...], name=...)` | `mlflow.pyfunc.load_model` | **Pickle-free**, recommended for production |

> ⚠️ MLflow 3 removed some older flavors: `fastai` (varies by channel), `mleap`, `diviner`, `gluon`. Use pyfunc + models_from_code for unsupported frameworks.

---

## 2. Serve Locally with `mlflow models serve`

```bash
mlflow models serve \
  -m models:/WineQualityClassifier@champion \
  -p 5001 \
  --host 127.0.0.1 \
  --workers 1 \
  --env-manager local        # conda | virtualenv | local (no isolation)
```

**MLflow 3.5+ requirement**: pass `--allowed-hosts` to the tracking server (not the serve command). The serve command itself accepts the model URI normally — but if your tracking server isn't allow-listed, browsers and external clients can't connect.

Endpoints exposed by `models serve`:
- `POST /invocations` — predict
- `GET /ping` — liveness
- `GET /health` — readiness (`{"status": "OK"}`)
- `GET /version` — MLflow version

Full flag list:
```bash
mlflow models serve --help
```

| Flag | Default | Notes |
|---|---|---|
| `-m, --model-uri` | (required) | `runs:/`, `models:/`, local path |
| `-p, --port` | 5000 | |
| `--host` | 127.0.0.1 | Use 0.0.0.0 in containers |
| `--workers` | 1 | gunicorn workers |
| `--timeout` | 60 | seconds per request |
| `--env-manager` | virtualenv | conda for strict repro; local for speed |
| `--install-mlflow` | False | install mlflow into the served env |
| `--engine` | default (waitress/gunicorn) | MLflow 3 supports Flask/Waitress/Gunicorn |

---

## 3. Request Payload Formats

All formats accepted at `POST /invocations`:

### A. JSON with `dataframe_split` (most common for tabular sklearn)

```json
{
  "dataframe_split": {
    "columns": ["fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
                "pH", "sulphates", "alcohol"],
    "data": [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]
  }
}
```

### B. JSON with `dataframe_records`

```json
{
  "dataframe_records": [
    {"fixed_acidity": 7.4, "volatile_acidity": 0.7, ..., "alcohol": 9.4},
    {"fixed_acidity": 7.8, "volatile_acidity": 0.88, ..., "alcohol": 9.8}
  ]
}
```

### C. JSON with `instances` (TensorFlow Serving style)

```json
{"instances": [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]}
```

### D. JSON with `inputs` (Keras style)

```json
{"inputs": [[7.4, 0.7, ...]]}
```

### E. CSV (content-type `text/csv`)

```
7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4
7.8,0.88,0.0,2.6,0.098,25.0,67.0,0.9968,3.2,0.68,9.8
```

### F. Parameters (signature must declare params)

```json
{
  "dataframe_split": {"columns": [...], "data": [...]},
  "params": {"threshold": 0.5}
}
```

> ⛔ The serve side **validates against your signature**. If you used `infer_signature(X, model.predict(X))`, requests MUST match that shape — columns, types, ordering. Mismatches raise `400 DataFrame column not found`.

Cross-platform smoke test with Python (no curl):
```python
import requests, json
r = requests.post(
    "http://127.0.0.1:5001/invocations",
    headers={"Content-Type": "application/json"},
    json={"dataframe_split": {"columns": cols, "data": rows}},
)
print(r.status_code, r.json())
```

---

## 4. Batch Scoring

### CLI

```bash
mlflow models predict \
  -m models:/WineQualityClassifier@champion \
  -i input.csv \
  -o output.csv \
  -t csv
```

Supported input types: `csv`, `json`, `parquet`. Output types: `csv`, `json`, `parquet` (use `-t json` if input is json and you want json output).

### Python

```python
import mlflow.pyfunc
import pandas as pd

model = mlflow.pyfunc.load_model("models:/WineQualityClassifier@champion")
preds = model.predict(pd.read_csv("input.csv"))
preds.to_csv("output.csv", index=False)
```

### Spark UDF (for large distributed batch)

```python
from pyspark.sql import SparkSession
import mlflow.pyfunc

spark = SparkSession.builder.getOrCreate()
predict_udf = mlflow.pyfunc.spark_udf(
    spark,
    "models:/WineQualityClassifier@champion",
    env_manager="conda",       # or "local"
    result_type="double",
)
df.withColumn("prediction", predict_udf("features")).show()
```

---

## 5. Custom PythonModel with `pyfunc.log_model`

```python
import mlflow.pyfunc

class PreprocessingClassifier(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # load artifacts (e.g., a fitted scaler) from context.artifacts
        import joblib
        self.scaler = joblib.load(context.artifacts["scaler"])
        self.model = mlflow.sklearn.load_model(context.artifacts["model_uri"])

    def predict(self, context, model_input, params=None):
        # model_input: pd.DataFrame
        scaled = self.scaler.transform(model_input)
        return self.model.predict(scaled)

mlflow.pyfunc.log_model(
    python_model=PreprocessingClassifier(),
    artifacts={
        "scaler": "scaler.joblib",
        "model_uri": "models:/WineQualityClassifier/3",
    },
    name="wine-classifier-with-preproc",
    input_example=X_train.head(3),
    signature=infer_signature(X_train, model.predict(X_train)),  # computed with raw X
)
```

---

## 6. `models_from_code` (Avoid Pickle)

> Pickling models has gotchas: Python version skew, class definition drift, security. MLflow 2.18+ supports `models_from_code` which loads your model class from a file in the model directory instead of unpickling.

```python
# predict.py  (saved into model_dir/code/)
import mlflow.pyfunc
import pandas as pd
import joblib

class WineClassifier(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model_path"])

    def predict(self, context, model_input: pd.DataFrame, params=None):
        return self.model.predict(model_input)

# elsewhere — log it
mlflow.pyfunc.log_model(
    python_model=None,             # not used in models_from_code
    artifacts={"model_path": "model.joblib"},
    code_paths=["predict.py"],     # <- tells MLflow to bundle this file
    name="wine-classifier",
)
```

When loading, MLflow reads `code/predict.py` and instantiates `WineClassifier` fresh. No pickle.

---

## 7. Containerize with `build-docker`

```bash
mlflow models build-docker \
  -m models:/WineQualityClassifier@champion \
  -n wine-classifier \
  --enable-mlserver   # use Seldon MLServer runtime instead of default Flask
```

Then:
```bash
docker run -p 5001:8080 wine-classifier
# or with MLServer:
docker run -p 8080:8080 -e MLSERVER_MODEL_URI='...' wine-classifier
```

MLServer (Seldon) is recommended for K8s deployments — supports multi-model, async, batch.

---

## 8. Deployment Targets

| Target | CLI / SDK | Notes |
|---|---|---|
| Local | `mlflow models serve` / `pyfunc.load_model().predict` | FastAPI server |
| Docker | `mlflow models build-docker` | Containerized local server |
| Kubernetes | `mlflow deployments` plugin (Databricks) | Via MLServer |
| AWS SageMaker | `mlflow sagemaker` | `deploy()`, `predict()` |
| Azure ML | `azureml-mlflow` plugin | |
| Modal | `modal` integration (community) | Serverless GPU |
| Custom | `mlflow deployments` plugin framework | Implement your own target |

```bash
# SageMaker example
mlflow sagemaker deploy \
  -m models:/WineQualityClassifier@champion \
  --region us-east-1 \
  --deployment-name wine-prod
```

---

## 9. CLI Matrix

```bash
# Serve
mlflow models serve -m <uri> -p 5001

# Batch predict
mlflow models predict -m <uri> -i input.csv -o output.csv -t csv

# Build Docker image
mlflow models build-docker -m <uri> -n <image-name>
mlflow models containerize -m <uri> -n <image-name>     # just generates Dockerfile

# Inspect a model locally
mlflow models predict -m <uri> -i input.json -o output.json -t json

# Custom targets (plugins)
mlflow deployments --help
mlflow sagemaker --help
```

---

## 10. Pitfalls

1. **`file:` backend + `models serve`**: fails with `No such registered model` because there's no DB to read registry from. Use sqlite/postgres or local file URI (`/path/to/mlruns`).
2. **MLflow 3.5+ tracking server missing `--allowed-hosts`**: browser clients get connection-refused. Add `--allowed-hosts "*"` for local dev.
3. **Port 5000 already in use**: mlflow ui also defaults to 5000; use 5001 for serve.
4. **Conda env rebuild slow**: first `models serve` may take 1-3 minutes to build the env. `--env-manager local` skips it.
5. **Signature mismatch**: client sends wrong columns/types → 400 error. Re-check `infer_signature(X, y)` columns.
6. **Wrong payload key**: using `"data"` or `"records"` instead of `dataframe_records` → server can't parse. Use one of the six documented formats.
7. **Workers vs timeout**: high `--workers` with small CPU = thrashing. Start with `--workers 1 --timeout 60`.
8. **`mlflow.pyfunc.spark_udf` with missing dependencies**: needs the same Python env on executors. Use `--conda-env` or pre-install.
9. **Custom PythonModel `predict()` exceptions**: not caught by the server — caller sees 500. Wrap with try/except and return error dicts.
10. **`mlflow runs compare` doesn't exist** (common typo): use `mlflow.search_runs` or the UI's Compare feature.

---

## See also

- `model-packaging.md` — signature, dependencies, code_paths deep dive
- `registry.md` — which URI to serve
- `ui.md` — visually inspecting served model in the tracking UI
- `troubleshooting.md` — "serve fails to start", "invocations returns 500"