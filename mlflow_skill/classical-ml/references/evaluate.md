# Model Evaluation Reference

Use `mlflow.models.evaluate` (NOT `mlflow.genai.evaluate`) for classical ML classification/regression tasks. Built-in metrics, SHAP, custom metrics, and threshold gates for CI/CD.

## Table of Contents
1. [`mlflow.models.evaluate` vs `mlflow.genai.evaluate`](#1-which-evaluate-api)
2. [Three Evaluation Modes](#2-three-evaluation-modes)
3. [Classification Built-in Metrics](#3-classification-built-in-metrics)
4. [Regression Built-in Metrics](#4-regression-built-in-metrics)
5. [Inspecting Results — `result.metrics / artifacts / tables`](#5-inspecting-results)
6. [Custom Metrics with `make_metric` + `MetricValue`](#6-custom-metrics)
7. [Threshold Gates — `MetricThreshold` + `validate_evaluation_results`](#7-threshold-gates)
8. [SHAP Explainability](#8-shap-explainability)
9. [Pitfalls](#9-pitfalls)

---

## 1. Which `evaluate` API?

| API | Use for | Don't use for |
|---|---|---|
| `mlflow.models.evaluate` | Tabular classification/regression on classical ML models (sklearn/XGBoost/PyTorch/etc.) | LLM/agent outputs |
| `mlflow.genai.evaluate` | GenAI outputs (traces, agent responses, RAG results) | Classical ML |
| `mlflow.evaluate` (legacy) | Deprecated; renamed and reshaped into `models.evaluate` | New code |

> ⛔ The two are **not interchangeable**. Passing a `pyfunc` LLM model to `models.evaluate` will silently produce wrong metrics; passing a sklearn classifier to `genai.evaluate` will error.

---

## 2. Three Evaluation Modes

### Mode 1: model on data (most common)

```python
import mlflow
import pandas as pd

result = mlflow.models.evaluate(
    model="models:/WineQualityClassifier@champion",  # OR runs:/<id>/<path>
    data=eval_df,                                    # features + label column
    targets="quality",                               # label column name
    model_type="classifier",                         # or "regressor"
    evaluators="default",
    extra_metrics=[my_custom_metric],                # from make_metric()
    evaluator_config={
        "log_explainer": True,                       # SHAP
        "explainer_type": "exact",                   # or "permutation", "tree"
        "pos_label": 1,
        "average": "weighted",
    },
)
print(result.metrics)
```

### Mode 2: pre-computed predictions (no re-inference)

```python
eval_df = eval_df.assign(pred=existing_predictions)
result = mlflow.models.evaluate(
    data=eval_df,
    targets="quality",
    predictions="pred",                              # column with predictions
    model_type="classifier",
)
```

### Mode 3: a function instead of a model

```python
def predict_fn(features_df):
    return loaded_model.predict(features_df)

result = mlflow.models.evaluate(
    model=predict_fn,
    data=eval_df,
    targets="quality",
    model_type="classifier",
)
```

---

## 3. Classification Built-in Metrics

For `model_type="classifier"`:

| Metric | Key in `result.metrics` |
|---|---|
| Accuracy | `accuracy_score` |
| Precision (positive class) | `precision_score` |
| Recall (positive class) | `recall_score` |
| F1 | `f1_score` |
| ROC-AUC | `roc_auc` |
| PR-AUC | `precision_recall_auc` |
| Log loss | `log_loss` |
| Brier score | `brier_score` |
| Confusion matrix (image) | `artifacts/.../confusion_matrix.png` |
| ROC curve (image) | `artifacts/.../roc_curve.png` |
| Precision-Recall curve | `artifacts/.../precision_recall_curve.png` |
| Per-class metrics table | `tables/.../classification_report_table.csv` |
| SHAP feature importance | `artifacts/.../shap/...` (if `log_explainer=True`) |

Some metrics depend on `pos_label` and `average` in `evaluator_config`.

---

## 4. Regression Built-in Metrics

For `model_type="regressor"`:

| Metric | Key in `result.metrics` |
|---|---|
| Mean Absolute Error | `mean_absolute_error` |
| Mean Squared Error | `mean_squared_error` |
| Root Mean Squared Error | `root_mean_squared_error` |
| R² | `r2_score` |
| Adjusted R² | `adjusted_r2_score` |
| Mean Absolute Percentage Error | `mean_absolute_percentage_error` |
| Residual plot | `artifacts/.../residual_plot.png` |

---

## 5. Inspecting Results

```python
result = mlflow.models.evaluate(...)

# Scalar metrics
for k, v in result.metrics.items():
    print(f"{k}: {v}")

# Artifact paths (PNG, HTML, etc.)
for path in result.artifacts:
    print(path)  # already uploaded to the eval run

# Tables (DataFrames)
for name, df in result.tables.items():
    print(name, df.shape)
```

The evaluation **creates its own MLflow run** (parent run) under the current experiment. All metrics/artifacts are logged there.

---

## 6. Custom Metrics with `make_metric`

`eval_fn` must be **pure** — no I/O, no global state, no prints, no file writes. MLflow parallelizes calls.

```python
from mlflow.models import make_metric
from mlflow.metrics.base import MetricValue

def weighted_accuracy_eval(predictions, targets, metrics):
    # predictions: pd.Series, targets: pd.Series, metrics: dict of built-in metrics
    tp = ((predictions == 1) & (targets == 1)).sum()
    fp = ((predictions == 1) & (targets == 0)).sum()
    tn = ((predictions == 0) & (targets == 0)).sum()
    fn = ((predictions == 0) & (targets == 1)).sum()
    score = (tp * 2 - fp * 1.5 + tn) / max(tp + fp + tn + fn, 1)
    return MetricValue(
        aggregate_results={"weighted_accuracy": float(score)},
        # optional: per-row judgments
        per_row_results={"weighted_accuracy_row": (predictions == targets).astype(float).tolist()},
    )

weighted_accuracy = make_metric(
    eval_fn=weighted_accuracy_eval,
    greater_is_better=True,
    name="weighted_accuracy",
)
```

Then pass to `evaluate(..., extra_metrics=[weighted_accuracy])`. The custom metric appears in `result.metrics["weighted_accuracy"]`.

---

## 7. Threshold Gates — `MetricThreshold` + `validate_evaluation_results`

```python
from mlflow.models import MetricThreshold
import mlflow

# Define thresholds
thresholds = {
    "accuracy_score": MetricThreshold(threshold=0.85, greater_is_better=True),
    "roc_auc":        MetricThreshold(threshold=0.90, greater_is_better=True),
    # Optional: require NEW model to BEAT old by at least N
    "accuracy_score": MetricThreshold(
        threshold=0.85, greater_is_better=True,
        min_absolute_change=0.02,           # ≥2% absolute improvement
        min_relative_change=None,
    ),
}

# Compare candidate vs baseline
try:
    mlflow.validate_evaluation_results(
        validation_thresholds=thresholds,
        candidate_result=result_new,
        baseline_result=result_old,           # if None, candidate vs threshold only
    )
    print("✓ New model meets all thresholds")
except mlflow.models.evaluation.ModelValidationFailedException as e:
    print(f"✗ Validation failed: {e}")
    raise
```

> ⚠️ **`min_absolute_change` must be ≥ 0**. The sign is dictated by `greater_is_better`. Negative values would let a new model be *worse* and pass.

Typical CI pattern:
```python
result_new = mlflow.models.evaluate(model="models:/New@v3", data=eval, targets="y",
                                   model_type="classifier")
result_old = mlflow.models.evaluate(model="models:/Old@v2", data=eval, targets="y",
                                   model_type="classifier")
mlflow.validate_evaluation_results(thresholds, result_new, result_old)
```

If validation fails, the registry promotion script should refuse to set `champion` alias.

---

## 8. SHAP Explainability

```python
result = mlflow.models.evaluate(
    model=model,
    data=eval_df,
    targets="label",
    model_type="classifier",
    evaluator_config={
        "log_explainer": True,
        "explainer_type": "exact",       # for tree models; "permutation" for black-box
    },
)
# SHAP artifacts land under artifacts/shap/
```

For tree models (XGBoost/LightGBM/sklearn trees), `explainer_type="exact"` is fast and accurate. For Keras/PyTorch, use `"permutation"` or `"deep"` (much slower).

---

## 9. Pitfalls

1. **Wrong `model_type`**: passing `"regressor"` to a classifier (or vice versa) returns incorrect metrics silently. Inspect `type(model)` first.
2. **Inconsistent feature columns at serve time**: if you trained with columns `["a","b","c"]` and serve with only `["a","b"]`, signature validation rejects the input. Use `infer_signature` and pass the same shape at inference.
3. **`eval_fn` side effects**: MLflow parallelizes — a `print()` or file write will fire multiple times in undefined order. Keep `eval_fn` pure.
4. **`min_absolute_change` negative**: silently lets worse models pass. Always ≥ 0.
5. **Missing baseline**: passing only `validation_thresholds` checks the candidate against the threshold value, not against the current production model. Pass `baseline_result=` for real comparisons.
6. **SHAP for non-tree models**: `explainer_type="exact"` is only valid for tree models; on Keras it raises. Pick `"permutation"` or `"deep"`.
7. **`predictions` column without `targets`**: you'll get no metrics. Always provide both.
8. **Categorical columns without encoding**: `data` must be numeric-ready; the eval API doesn't auto-encode string columns.
9. **Large `data`**: evaluation copies data into the run. For >100k rows, prefer Mode 2 (pre-computed predictions) to avoid re-inference.

---

## See also

- `tracking.md` — how to log the model being evaluated
- `registry.md` — how to promote a model that passed thresholds
- `optimize.md` — using evaluate in Optuna/GridSearchCV trials
- `troubleshooting.md` — `ModelValidationFailedException` debugging