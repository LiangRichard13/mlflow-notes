# 阶段 4 学习笔记：评估、服务与经典 ML 验证

> 对应脚本：`04_evaluate/04a_evaluate_basics.py`、`04b_evaluate_custom.py`、`04c_models_serve.sh`

## 一、`mlflow.models.evaluate()` 内置评估

```python
result = mlflow.models.evaluate(
    model="runs:/<run_id>/model",   # 或 models:/MyModel@champion
    data=eval_df,                   # 包含 features + label 列
    targets="target",               # label 列名
    model_type="classifier",        # 或 "regressor"
    evaluators=["default"],
    extra_metrics=[custom_metric],  # 自定义指标（MLflow 3 用 extra_metrics）
)
```

**自动产出**（写到 Run 的 Artifacts/eval/）：
- `confusion_matrix.png`
- `roc_curve_plot.png`
- `precision_recall_curve_plot.png`
- `calibration_curve_plot.png`
- `per_class_metrics/` — 各类指标 JSON

**内置指标**：
- `accuracy_score`, `precision_score`, `recall_score`, `f1_score`
- `log_loss`, `roc_auc`, `precision_recall_auc`
- `mean_absolute_error`, `mean_squared_error`, `r2_score`（回归）

## 二、自定义指标

```python
from mlflow.metrics import make_metric

def my_fn(predictions, targets):
    # 注意 predictions/ targets 是 pandas Series
    return float((predictions == targets).mean())

custom = make_metric(
    eval_fn=my_fn,
    greater_is_better=True,
    name="my_custom_metric",
)

result = mlflow.models.evaluate(..., extra_metrics=[custom])
```

⚠️ **MLflow 3 的参数名是 `extra_metrics`，不是 `custom_metrics`**

## 三、`validate_evaluation_results`（MLflow 3 新 API）

替代 MLflow 2 的 `baseline_model=` 参数：

```python
from mlflow.models import MetricThreshold

# 跑两次 evaluate
result_baseline = mlflow.evaluate(model_uri_a, ...)
result_candidate = mlflow.evaluate(model_uri_b, ...)

# 定义阈值
thresholds = {
    "accuracy_score": MetricThreshold(
        threshold=0.9,                  # 绝对值下限
        min_absolute_change=0.02,      # 至少比 baseline 高 0.02
        min_relative_change=0.05,      # 或至少高 5%
        greater_is_better=True,
    ),
}

# 验证
mlflow.validate_evaluation_results(
    validation_thresholds=thresholds,
    candidate_result=result_candidate,
    baseline_result=result_baseline,
)
# 通过 → 不抛异常
# 失败 → 抛 MlflowException
```

⚠️ `min_absolute_change` 必须是非负数（始终是"比 baseline 至少好多少"）

## 四、本地部署：`mlflow models serve`

```bash
# 启动 tracking server（一个终端）
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --default-artifact-root ./mlruns \
              --port 5000

# 注册 + 设置别名（另一个终端）
# python 03_registry/02a_log_model.py
# python 03_registry/02b_register_alias.py

# 部署模型（再一个终端）
mlflow models serve -m "models:/WineQualityClassifier@champion" -p 5001

# curl 推理
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  --data '{"dataframe_records": [{...}]}'
```

**两种请求格式**：
- `application/json` + `{"dataframe_records": [...]}` 或 `{"dataframe_split": {...}}`
- `text/csv` + CSV 文件

格式自动选择基于模型签名（signature）。

## 五、生产部署

```bash
# Docker 镜像
mlflow models build-docker -m models:/WineQualityClassifier@champion -n wine-classifier
docker run -p 5001:8080 wine-classifier

# 纯命令行推理
mlflow models predict -m models:/WineQualityClassifier@champion \
                      -i input.csv -o output.csv
```

## 六、常见坑

1. **Registry 必须有 backend store**：纯文件模式不支持 Model Registry
2. **MLflow 3.5+ 必须配 `--allowed-hosts`** 防 DNS rebinding
3. **predict 时格式要对**：JSON 必须用 `dataframe_records` 或 `dataframe_split`
4. **`extra_metrics` 不是 `custom_metrics`**：MLflow 3 参数改名了
5. **`min_absolute_change` 必须 ≥ 0**：表达的是"好多少"，不是"差多少"