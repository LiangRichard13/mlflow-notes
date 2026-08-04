# 阶段 4 学习笔记：评估、服务与经典 ML 验证

> 对应脚本：`04_evaluate/04a_evaluate_basics.py`、`04b_evaluate_custom.py`、`04c_models_serve.sh`
> 需要 API Key：否
> 阶段定位：Phase 1-3 你已经会"训练 + 记录 + 注册模型"了；这一阶段回答两个问题——**这个模型好不好？** 以及 **怎么把它跑起来给别人用？**

## 🎯 这篇笔记做什么

模型训完之后，真正的工作才刚开始：你得**量化它有多好**（光看 0.97 没用，得知道是哪一类的 0.97），还得**把它变成一个能 curl 的服务**让别人用。

这一阶段解决两件事：
1. **评估**：用一行 `mlflow.models.evaluate()` 自动算一堆指标 + 自动画混淆矩阵、ROC 曲线，省去手写 `matplotlib` 的痛苦。
2. **部署**：用 `mlflow models serve` 把注册好的模型变成一个本地 REST API，一条 `curl` 就能调。

> 🍳 **类比**：训练模型就像做菜——厨师（你）做完一道菜，要做两件事：
> - **试吃**（评估）：找几个食客打分、看摆盘是否好看（混淆矩阵）、味道曲线是否平滑（ROC）。MLflow 的 `evaluate` 就是帮你搞了个自动试吃团。
> - **上桌**（部署）：把菜放进窗口让客人点餐（`models serve` 起一个 HTTP 服务）。

### 你会学到什么

- 一行代码跑全套分类/回归指标 + 自动生成可视化图
- 用 `make_metric` 写自定义指标（例如"高价值客户加权 accuracy"）
- 用 `validate_evaluation_results` 对比新旧模型，自动化"上线门槛"（MLflow 3 推荐写法）
- 用 `mlflow models serve` 把模型起成 REST API
- 用 `curl` 推 JSON/CSV 到 `/invocations` 做推理

### 对应脚本清单

| 脚本 | 一句话作用 | 是否必跑 | 前置 |
|------|-----------|---------|------|
| `04a_evaluate_basics.py` | 跑一次 `mlflow.models.evaluate`，自动算所有内置指标 + 生成混淆矩阵/ROC 图 | ✓ 必跑 | Phase 2 |
| `04b_evaluate_custom.py` | 写一个"高价值客户加权"自定义指标，对比 RF 和 LR 两个模型，验证 B 是否比 A 好 | 推荐 | 跑过 04a |
| `04c_models_serve.sh` | 用 `mlflow models serve` 起本地 REST API，再用 `curl` 调 `/invocations` 推理 | 推荐 | 跑过 04b + 注册过模型 |

### 前置知识

- 已完成 Phase 1-3，会写 `mlflow.start_run()`、`mlflow.sklearn.log_model()`
- 至少注册过一个模型到 Model Registry（`mlflow.register_model()` 或 alias）
- MLflow Tracking Server 已在 `sqlite:///mlflow.db` + `./mlruns` 跑起来（04a/04b 用本地文件模式也行；04c **必须** server 模式）

### 跑完必看（UI）

1. 启动 UI：`mlflow ui --port 5000`
2. 选 experiment `04_evaluate`
3. 点开 Run `evaluate-baseline`
4. 看：
   - **Metrics 标签**：`accuracy_score`、`f1_score`、`roc_auc` 等内置指标一行行排好
   - **Artifacts → eval/**：自动生成的 `confusion_matrix.png`、`roc_curve_plot.png`、`precision_recall_curve_plot.png`
   - **Artifacts → model/**：模型本身 + `signature` + `requirements.txt`

---

## 一、核心概念：用人话讲清楚

### 1.1 `mlflow.models.evaluate()` 是干嘛的

想想你以前怎么"评估一个 sklearn 模型"：
```python
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average="macro"))
# 还得 import matplotlib, 手写混淆矩阵代码...
# 想跑回归？再 import 一堆 metrics...
```

`mlflow.models.evaluate()` 是 **"全自动评估 + 自动作图 + 自动写回 MLflow"** 的整合入口。你只需要：
- 指一个模型（`runs:/xxx/model` 或 `models:/MyModel@champion`）
- 给一份含 label 列的数据
- 说清楚是分类还是回归（`model_type="classifier"`）

它就帮你：
- 算所有内置指标（accuracy / F1 / precision / recall / ROC-AUC / log_loss...）
- 对分类自动生成：混淆矩阵图、ROC 曲线、PR 曲线、校准曲线
- 对回归自动生成：残差图、预测 vs 真实散点图
- 把所有指标和图都写回当前 Run

> 💡 这意味着：在 MLflow UI 里你点开一个 Run，**所有评估结果在同一个页面**，不需要跳到别处找图。

### 1.2 自定义指标 `make_metric`

内置指标对简单问题够用，但业务上经常要更"狡猾"的指标，例如：
- "class_0 是高价值客户，识别错了代价 5 倍" → **加权 accuracy**
- "假阳性罚 10 元，假阴性罚 100 元" → **业务损失函数**
- "top-5 推荐命中率" → **业务命中率**

`make_metric` 就是让你把这种"我的业务特殊规则"塞进去，包装成一个 MLflow 指标。它会和内置指标一起出现在 UI 的 Metrics 标签里。

### 1.3 `validate_evaluation_results`：MLflow 3 的"模型升级门槛"

业务里上线的真实流程：
1. 旧模型 A 在跑
2. 你训了新模型 B
3. 关键是：**B 必须比 A 好多少才允许替换？**

MLflow 2 时代这套规则藏在 `mlflow.evaluate(baseline_model=...)` 一个超长参数里，很难复用、很难调试。
MLflow 3 拆成了两步：
- 先分别对 A、B 各 `evaluate()` 一次（拿到两个 `EvaluationResult` 对象）
- 再用 `validate_evaluation_results(candidate=B, baseline=A, thresholds={...})` 验证 B 是否达标

好处：**阈值规则可以独立写、独立复用、独立测试**，还能塞进 CI。

### 1.4 `mlflow models serve`：把模型变成 HTTP 服务

训练完的模型本质是个文件，**别人没法直接用**（除非你把 sklearn + pickle 文件传给他）。`mlflow models serve` 把模型起成一个标准 REST API，路径统一是 `/invocations`，接受 JSON 或 CSV——这样前端、后端、别的服务都能用 `curl` 调。

它内部会装好这个模型需要的 Python 环境（用 model signature + requirements 推断），你不用管 conda。

---

## 二、代码模式：可复用的模板

### 2.1 内置评估模板（`mlflow.models.evaluate`）

```python
import mlflow

with mlflow.start_run(run_name="evaluate-baseline") as run:
    # 先把模型 log 进去（evaluate 要 model_uri）
    mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(3))
    model_uri = f"runs:/{run.info.run_id}/model"

    # 一行评估
    result = mlflow.models.evaluate(
        model=model_uri,
        data=eval_df,                       # 必须含 label 列
        targets="target",                   # label 列名
        model_type="classifier",            # 或 "regressor"
        evaluators=["default"],
    )

    # 拿指标
    print(result.metrics)                  # dict: {accuracy_score: 0.97, ...}
    # 拿可视化列表
    print(result.artifacts)                # ['confusion_matrix.png', 'roc_curve_plot.png', ...]
```

**自动产出**（写到 Run 的 `Artifacts/eval/`）：
- `confusion_matrix.png`：分类器最常看的图
- `roc_curve_plot.png`：ROC 曲线 + AUC
- `precision_recall_curve_plot.png`：不平衡数据更该看的图
- `calibration_curve_plot.png`：概率校准
- `per_class_metrics/`：每个类单独的指标 JSON

**内置指标全集**（分类）：
`accuracy_score`、`precision_score`、`recall_score`、`f1_score`、`log_loss`、`roc_auc`、`precision_recall_auc`

**内置指标全集**（回归）：
`mean_absolute_error`、`mean_squared_error`、`root_mean_squared_error`、`r2_score`、`mean_absolute_percentage_error`

> ⚠️ **MLflow 3 的参数名是 `extra_metrics`，不是 `custom_metrics`**——这是新手最容易翻车的地方，下面有专门避坑。

### 2.2 自定义指标模板（`make_metric`）

```python
from mlflow.metrics import make_metric
import numpy as np

def my_metric_fn(predictions, targets):
    """
    predictions 和 targets 都是 pandas Series,index 一一对齐。
    返回值必须是 float (或可转 float 的标量)。
    """
    preds = np.asarray(predictions)
    targs = np.asarray(targets)
    # 例：按 class_weights 加权的 accuracy
    CLASS_WEIGHTS = {0: 5.0, 1: 1.0, 2: 1.0}
    total = 0.0
    correct = 0.0
    for p, t in zip(preds, targs):
        w = CLASS_WEIGHTS.get(int(t), 1.0)
        total += w
        if p == t:
            correct += w
    return float(correct / total)

custom_metric = make_metric(
    eval_fn=my_metric_fn,
    greater_is_better=True,         # 越大越好；如果是 loss 这种，填 False
    name="weighted_accuracy_v1",    # 在 UI 显示的名字
)

# 塞进 evaluate
result = mlflow.models.evaluate(
    model=model_uri, data=eval_df, targets="target",
    model_type="classifier",
    extra_metrics=[custom_metric],   # ← 注意：MLflow 3 用 extra_metrics
)
# 现在 result.metrics 里会多一项 "weighted_accuracy_v1"
```

**`eval_fn` 签名必须遵守的规矩**：
- 入参：`predictions`（模型预测值 Series）、`targets`（真实标签 Series）
- 返回：标量（float / int / numpy scalar）
- 不要在这里面 print 或写文件——纯函数

### 2.3 `validate_evaluation_results` 模板（MLflow 3 新写法）

```python
from mlflow.models import MetricThreshold

# 假设 result_a 是旧模型、result_b 是新模型的 EvaluationResult
result_a = mlflow.models.evaluate(model=old_uri,  data=eval_df, targets="target", model_type="classifier", evaluators=["default"])
result_b = mlflow.models.evaluate(model=new_uri,  data=eval_df, targets="target", model_type="classifier", evaluators=["default"])

# 定义 candidate (新模型 B) 必须达到的门槛
thresholds = {
    "accuracy_score": MetricThreshold(
        threshold=0.90,                # 绝对值下限：B 至少要 0.90
        greater_is_better=True,
        # 可选：相对 baseline 的提升要求
        # min_absolute_change=0.02,   # B 比 A 至少高 0.02
        # min_relative_change=0.05,   # 或至少高 5%
    ),
    "f1_score": MetricThreshold(
        threshold=0.85,
        greater_is_better=True,
    ),
}

# 验证
try:
    mlflow.validate_evaluation_results(
        validation_thresholds=thresholds,
        candidate_result=result_b,     # 新模型
        baseline_result=result_a,      # 旧模型（可省略,只验绝对值）
    )
    print("✓ 通过，新模型可以替换")
except Exception as e:
    print(f"✗ 不通过：{e}")
    # MLflow 会抛 MlflowException；你可以决定是否让上线流程中断
```

**MLflow 2 vs 3 对比**：

```python
# MLflow 2（你可能在旧文档里看到）
result = mlflow.evaluate(
    model=new_uri,
    data=eval_df,
    targets="target",
    model_type="classifier",
    baseline_model=old_uri,           # ← 旧写法,所有阈值挤在一个地方
    metric_thresholds=thresholds,
)

# MLflow 3（推荐）
result_a = mlflow.evaluate(model=old_uri, ...)
result_b = mlflow.evaluate(model=new_uri, ...)
mlflow.validate_evaluation_results(   # ← 新写法：拆出来,更清晰可测
    validation_thresholds=thresholds,
    candidate_result=result_b,
    baseline_result=result_a,
)
```

> 💡 **为什么 MLflow 3 要拆开？** 因为 `validate_evaluation_results` 拿到的 `EvaluationResult` 对象本身就是可序列化、可缓存、可存数据库的——你可以在 CI 里把它的 JSON 存下来，下次复用同样的 baseline 做对比。

### 2.4 `mlflow models serve` 部署模板

```bash
# 终端 A：启 MLflow server（models serve 必须有 server）
mlflow server \
  --backend-store-uri sqlite:///$(pwd)/mlflow.db \
  --default-artifact-root $(pwd)/mlruns \
  --host 0.0.0.0 --port 5000

# 终端 B：部署 champion 模型
mlflow models serve \
  -m "models:/WineQualityClassifier@champion" \
  -p 5001
# 第一次启动会 pip install 模型依赖（conda env，要等几十秒）

# 终端 C：curl 推理（JSON 格式，推荐）
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  --data '{
    "dataframe_records": [
      {"alcohol": 13.0, "malic_acid": 1.5, "ash": 2.5, "alcalinity_of_ash": 19.0,
       "magnesium": 100, "total_phenols": 2.8, "flavanoids": 3.0,
       "nonflavanoid_phenols": 0.3, "proanthocyanins": 1.8, "color_intensity": 5.0,
       "hue": 1.0, "od280/od315_of_diluted_wines": 3.0, "proline": 1000}
    ]
  }'
```

**支持的请求格式**（格式由模型签名自动决定）：
| Content-Type | Body 格式 |
|---|---|
| `application/json` | `{"dataframe_records": [...]}` —— 每条是一个 dict（推荐） |
| `application/json` | `{"dataframe_split": {"columns": [...], "data": [[...]]}}` |
| `text/csv` | 纯 CSV 文本，第一行是列名 |

返回是 JSON：`{"predictions": [...]}` 或带 `{"predictions": [...], "probabilities": [[...]]}`。

---

## 三、实战步骤：按顺序照做

### Step 1：跑 `04a_evaluate_basics.py`

```bash
python 04_evaluate/04a_evaluate_basics.py
```

你会看到：训练完一个 RandomForest，对它跑一遍 `mlflow.models.evaluate`，打印一坨指标和图的名字。

打开 MLflow UI → experiment `04_evaluate` → Run `evaluate-baseline`：
- **Metrics 标签**：看 `accuracy_score`、`f1_score`、`roc_auc` 等
- **Artifacts → eval/**：能看到 `confusion_matrix.png`、`roc_curve_plot.png` 等自动生成的图

> ✨ 这一步完成时，你应该理解：**模型评估这件事，UI 里点开 Run 就能看到所有结果——不用自己 plot。**

### Step 2：跑 `04b_evaluate_custom.py`

```bash
python 04_evaluate/04b_evaluate_custom.py
```

这一脚本做了 5 件事：
1. 训练 RF（A）和 LR（B）两个模型
2. 写一个"高价值客户（class_0）加权 5 倍"的自定义 `weighted_accuracy_v1`
3. 对 A、B 各跑一次 `evaluate()`，塞入 `extra_metrics=[custom_metric]`
4. 用 `validate_evaluation_results` 比 B 是否比 A 好
5. 打印 MLflow 2 vs 3 的 API 差异图

**重点观察**：
- 两个 Run 的 Metrics 标签里都多了一行 `weighted_accuracy_v1`
- 脚本末尾会打印"✓ 通过"或"✗ 不通过"，看看 B 是不是真的比 A 好（业务场景下 LR 在小数据上可能不如 RF）

### Step 3：跑 `04c_models_serve.sh`

这个是 shell 脚本，需要**三个终端**：

**终端 A**（启动 MLflow server）：
```bash
bash 04_evaluate/04c_models_serve.sh   # 或手动复制里面的命令
# 或者：把 04c 拆开,先跑 terminal 1 的 server 部分
```

**终端 B**（注册并打 champion）：
```bash
# 假设你已经跑过 Phase 3 的注册脚本,模型叫 WineQualityClassifier,有 @champion 别名
python 03_registry/02a_log_model.py
python 03_registry/02b_register_alias.py
```

**终端 C**（起模型服务）：
```bash
mlflow models serve -m "models:/WineQualityClassifier@champion" -p 5001
# 第一次启动会 pip install,等几十秒到一分钟
# 看到 "Listening on http://127.0.0.1:5001" 就 ok
```

**终端 D**（curl 推理）：
```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  --data '{"dataframe_records": [{"alcohol": 13.0, ...}]}'
```

返回：
```json
{"predictions": [0]}     # 类 0（某种葡萄酒）
```

> ✨ 这一步完成时，你已经在用生产级的方式（HTTP + JSON）调用模型了——和 `requests.post(url, json=...)` 完全一样。

### Step 4：（可选）容器化部署

```bash
# 把模型打成 Docker 镜像
mlflow models build-docker -m "models:/WineQualityClassifier@champion" -n wine-classifier

# 跑容器（容器内 8080 端口对应外部 5001）
docker run -p 5001:8080 wine-classifier
```

> 这个就是你写论文/做 demo 时给评审看"我真的部署了一个 ML 服务"的证据。

---

## 四、避坑清单

### 坑 1：把 `custom_metrics` 当参数名（最常见的 API 改名）

```python
# ❌ 报错：TypeError: got unexpected keyword argument 'custom_metrics'
result = mlflow.models.evaluate(
    model=..., data=..., targets=..., model_type=...,
    custom_metrics=[custom_metric],
)

# ✓ MLflow 3 改名叫 extra_metrics
result = mlflow.models.evaluate(
    model=..., data=..., targets=..., model_type=...,
    extra_metrics=[custom_metric],
)
```

**为啥改了**：MLflow 3 把内置 evaluator（`"default"`、`"shap"` 等）和自定义 metric 统一到一个 `extra_*` 命名空间，未来再加 evaluator / metric 不会再撞名。

### 坑 2：`models serve` 用 file store 不行

```bash
# ❌ mlflow models serve 需要 server 模式
# ❌ 纯 --backend-store-uri ./mlruns 这种 fs 模式不支持 Model Registry
# ✓ 必须 sqlite / postgres / mysql
mlflow server \
  --backend-store-uri sqlite:///$(pwd)/mlflow.db \
  --default-artifact-root $(pwd)/mlruns \
  --host 0.0.0.0 --port 5000
```

错误现象：`No such registered model: WineQualityClassifier`——明明 log 了，但找不到。

**为啥**：Model Registry 是 server 的功能，纯文件模式不支持 stage/alias/registered model。

### 坑 3：predict 时 JSON 格式写错

```bash
# ❌ 报错：DataFrame column not found
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  --data '{"data": [{"alcohol": 13.0, ...}]}'   # 错！应该是 dataframe_records

# ✓ 必须用 MLflow 约定的两个 key 之一
--data '{"dataframe_records": [...]}'
--data '{"dataframe_split": {"columns": [...], "data": [[...]]}}'
```

**为啥**：MLflow 在服务侧会判断 key 名再决定怎么转 pandas。

### 坑 4：`min_absolute_change` 写了负数

```python
# ❌ 报错或行为反掉
MetricThreshold(threshold=0.9, min_absolute_change=-0.02)

# ✓ min_absolute_change 必须是 ≥ 0
# 它表达的是"candidate 比 baseline 至少好 X"——
# 如果你的指标是 greater_is_better=True,正数=更好;greater_is_better=False,正数=更小
MetricThreshold(threshold=0.9, min_absolute_change=0.02, greater_is_better=True)
```

**为啥**：`min_absolute_change` 的符号语义由 `greater_is_better` 自动决定，你写负数反而会产生"我允许新模型比旧模型差"的诡异效果。

### 坑 5：MLflow 3.5+ 必须配 `--allowed-hosts`

```bash
# ❌ 浏览器打 mlflow ui 报 "Invalid Host header"
mlflow ui --port 5000

# ✓ MLflow 3.5 默认拒绝非 localhost 的 Host header（防 DNS rebinding）
mlflow server --host 127.0.0.1 --port 5000 --allowed-hosts "*"
# 或在 config.toml 里设
```

### 坑 6：`make_metric` 的 `eval_fn` 不是纯函数

```python
# ❌ 报错或结果不稳定
def my_metric(predictions, targets):
    print(len(predictions))                     # 不允许
    open("/tmp/log.txt", "a").write("hi\n")    # 不允许
    global SOME_GLOBAL                           # 不允许
    SOME_GLOBAL += 1
    return float((predictions == targets).mean())

# ✓ 纯函数：只读入参,返回标量
def my_metric(predictions, targets):
    return float((predictions == targets).mean())
```

**为啥**：MLflow 在某些场景会并行调用 `eval_fn`，副作用会乱序或丢。

### 坑 7：`mlflow.evaluate` 的入参名也变了

```python
# MLflow 2.16+: 入参 keyword 从 model_type 改成 model_type (兼容)
# 但有些签名细节在变,看最新官方文档
# 经验法则:跟着 mlflow.models.evaluate 走,不要 import mlflow.evaluate 然后传 model=...
result = mlflow.models.evaluate(   # ✓ 用 models 命名空间下的
    model=..., data=..., targets=..., model_type="classifier",
)
```

---

## 五、小结：5 个 take-aways

1. **`mlflow.models.evaluate` 是你的"一站式评估员"**：给模型 + 数据 + 类型，它吐一整套指标和图——不用再手写 `matplotlib`。
2. **MLflow 3 的自定义指标参数叫 `extra_metrics`**：`custom_metrics` 是 MLflow 2 的命名，新代码不要用。
3. **自定义指标用 `make_metric(eval_fn=..., greater_is_better=..., name=...)`**：`eval_fn` 必须是纯函数，接收 `predictions`/`targets` Series，return float。
4. **模型对比用 `validate_evaluation_results`（MLflow 3 新写法）**：拆成"对每个模型 evaluate 一次" + "集中验证 candidate vs baseline"，不再用 MLflow 2 的 `baseline_model=` 一锅炖。
5. **`mlflow models serve` 把模型变 REST API**：必须 sqlite/postgres 等数据库后端，配 `models:/Name@alias` 起服务，`/invocations` 收 `dataframe_records` JSON 或 `text/csv`——和 `curl`/`requests.post` 完全一样。

---

## 六、和前几阶段的衔接

| 阶段 | 关注点 | 阶段 4 加什么 |
|---|---|---|
| Phase 1-2 | 训练 + tracking | 加 `mlflow.models.evaluate` 自动算全套指标 + 自动出图 |
| Phase 3 | 模型注册 + alias | 用 `validate_evaluation_results` 做 "B 是否真的比 @champion 好" |
| Phase 4 | **评估 + 部署** | 起 `models serve` 把 @champion 变成 HTTP API 给别人用 |
| Phase 5+ | （看你想往哪走） | 可以学 Docker / KServe / Databricks Model Serving |

> 🎓 **学完这一阶段，你已经是"能交付 ML 模型"的人了**：训练 → 记录 → 注册 → 评估 → 部署上线。整个 MLOps 闭环的基础部分你已经能跑通。后续阶段会带你看更复杂的实验管理、超参调优、CI/CD 等。
