# 阶段 1 学习笔记：入门与追踪基础

> 对应脚本：`scripts/01_basics/01_hello_mlflow.py`、`scripts/01_basics/01b_sklearn_basics.py`
> 需要 API Key：否

## 🎯 这篇笔记做什么

这一阶段带你跨进 MLflow 的大门——搞清楚"跑实验时到底该怎么把每次训练的关键信息记下来"。你可能已经有过这样的痛苦经历：调了一晚上超参数，第二天回想"上次那组 lr=0.01、batch_size=32 的效果到底是多少来着？模型文件扔哪了？训练曲线截图存哪了？"。MLflow 就是来解决这个问题的——它像一个**实验记账本**，每次训练都自动帮你把参数、指标、模型文件、备注标签统统归档。

类比：如果你把每一次模型训练当成"做一道菜"，那 MLflow 就是你的**厨房日志**——记录每次用了什么食材（Param）、出品评分多少（Metric）、成品照片存在哪（Artifact）、备注标签比如"辣/不辣/试做"（Tag）。几个相关实验放在一起就是一道"实验项目"（Experiment）。

### 你会学到什么

- 用最基础的 3 个 API（`log_param` / `log_metric` / `log_artifact`）手动记录一次训练
- 用 `mlflow.sklearn.autolog()` 一行代码自动记录 sklearn 训练全过程
- 理解 MLflow 的 4 个核心对象：Experiment / Run / Param / Metric / Artifact / Tag
- 启动 `mlflow ui`，用 Compare 功能对比多个 Run 的效果

### 对应脚本清单

| 脚本 | 一句话作用 | 是否必跑 | 前置 |
|------|-----------|---------|------|
| `01_hello_mlflow.py` | 纯手动 `log_param`/`log_metric`/`log_artifact`，最基础的 demo | ✓ 必跑 | 无 |
| `01b_sklearn_basics.py` | 用 `mlflow.sklearn.autolog()` 自动记录，4 个 sklearn 模型对比 | 推荐 | 跑过 01 |

### 前置知识

- 会用 Python 和 scikit-learn（至少能看懂 `fit` / `predict` / `train_test_split`）
- 懂"超参数"、"训练集/测试集"、"accuracy/f1"这些基本概念
- 本阶段**不要求**任何 MLOps 经验，也不需要 API Key
- 已安装 `mlflow` 和 `scikit-learn`（`pip install mlflow scikit-learn`）

### 跑完必看（UI）

1. 启动 UI：`mlflow ui --port 5000`
2. 浏览器打开 `http://localhost:5000`
3. 选 experiment `01_basics_demo`：看 3 个 Run 的参数对比表
4. 选 experiment `01_sklearn_iris`：勾选 4 个 Run，点 **Compare**，并排看 Param/Metric
5. 点开任一 Run，看 Artifacts → model/ 下的 `model.pkl`、`MLmodel`、`conda.yaml`

---

## 一、核心概念：用人话讲清楚

### 1. Experiment（实验）

一组相关 Run 的容器。比如"iris 分类项目"是一个 Experiment，里面跑的所有模型都是它的 Run。

> 类比：就像一个项目文件夹，里面放着你这次课题跑过的所有实验记录。

### 2. Run（一次运行）

单次训练的执行过程。每次 `start_run()` 就会产生一个 Run，有唯一的 `run_id`。

> 类比：相当于项目文件夹里的一篇"实验日记"。

### 3. Param（参数）

字符串型配置，比如 `learning_rate=0.01`、`batch_size=32`、`optimizer="adam"`。**同一个 Param key 只能记一次**，所以适合记那些不会变的超参。

### 4. Metric（指标）

数值型效果，比如 `loss=0.35`、`accuracy=0.92`。**可以带 step**，所以能记录每个 epoch 的 loss，画出训练曲线。

### 5. Artifact（产物）

任意文件：模型文件、配置文件、训练曲线图、混淆矩阵图……MLflow 会把它们统一收在每个 Run 的 `Artifacts` 文件夹下。

> 类比：实验日记里贴的截图、附件、模型快照。

### 6. Tag（标签）

任意备注文本，比如 `status=completed`、`dataset=iris`、`notes=第3次试做`。主要用于 UI 里过滤/搜索。

### 7. autolog（自动记录）

很多框架（sklearn / pytorch / xgboost / lightgbm）MLflow 都提供了 `mlflow.<框架>.autolog()`，**一行代码**就帮你自动记录参数、指标、模型文件、签名等等。

---

## 二、代码模式：可复用的模板

### 模式 1：手动记录（最小可用）

```python
import mlflow
from mlflow import log_param, log_metric, log_artifact, set_experiment

# 1. 选/创建实验
set_experiment("my_project")

# 2. 启动一次 Run（用 with 自动关闭）
with mlflow.start_run(run_name="trial-1") as run:
    log_param("lr", 0.01)              # 记参数
    log_metric("loss", 0.5, step=0)     # 记指标（带 step 画曲线）
    log_metric("loss", 0.3, step=1)
    log_artifact("model.pkl")           # 记文件
    # 异常会自动 end_run，不用手动管
```

**什么时候用**：调小实验、想精确控制记录什么；或者框架不在 autolog 覆盖范围时。

### 模式 2：批量记录

```python
mlflow.log_params({"lr": 0.01, "batch_size": 32})
mlflow.log_metrics({"loss": 0.3, "accuracy": 0.92})
```

**什么时候用**：参数/指标很多时，比一行行写更整洁。

### 模式 3：sklearn 自动记录

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

mlflow.sklearn.autolog()  # 一行开启

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    # 自动记录：超参、训练指标、模型文件、签名
```

**什么时候用**：跑 sklearn / xgboost / lightgbm 等常见框架，想省事。

### 模式 4：手动记模型 + 签名

```python
from mlflow.models import infer_signature

signature = infer_signature(X_train, model.predict(X_train))
input_example = X_train[:3]

mlflow.sklearn.log_model(
    model,
    artifact_path="model",
    signature=signature,
    input_example=input_example,
)
```

**什么时候用**：要部署模型时，签名让 MLflow 知道输入输出长什么样。

### 模式 5：打 Tag 做标注

```python
mlflow.set_tag("dataset", "iris")
mlflow.set_tag("status", "completed")
mlflow.set_tag("notes", "第一次试做，lr 太大")
```

**什么时候用**：事后想筛选/搜索 Run，比如"找出所有用 iris 数据集的 Run"。

---

## 三、实战步骤：按顺序照做

### Step 1：环境准备

```bash
conda activate mlflow          # 或你的虚拟环境
pip install mlflow scikit-learn
cd <project-root>
```

### Step 2：跑第一个 demo（手动记录）

```bash
python scripts/01_basics/01_hello_mlflow.py
```

这个脚本会：
- 创建 experiment `01_basics_demo`
- 跑 3 个 Run，每个模拟不同 `learning_rate` / `batch_size`
- 记录 10 个 epoch 的 loss/accuracy 曲线
- 写两个 artifact 文件（config.txt、summary.md）

跑完会看到 "所有 Run 已记录！下一步" 的提示。

### Step 3：跑第二个 demo（autolog + 真实 sklearn）

```bash
python scripts/01_basics/01b_sklearn_basics.py
```

这个脚本会：
- 创建 experiment `01_sklearn_iris`
- 跑 4 个 sklearn 模型（逻辑回归强/弱正则、随机森林深/浅）
- 每个都记录 accuracy、f1、模型文件（带签名）

### Step 4：启动 UI 看结果

另开一个终端：

```bash
cd <project-root>
mlflow ui --port 5000
```

浏览器打开 `http://localhost:5000`，按"跑完必看"那节的清单点一遍。

---

## 四、避坑清单

### 坑 1：Run 没关程序就崩了

**症状**：UI 里 Run 状态一直是 `RUNNING`，数据没完整记录。

**原因**：手动调 `start_run()` 但忘了 `end_run()`，程序异常退出。

**解决**：永远用 `with mlflow.start_run() as run:`，异常会自动结束 Run。

### 坑 2：`log_param` 同一个 key 第二次会警告

**症状**：控制台警告 "Changing param .. is not allowed"。

**原因**：Param 设计为"一次定型"，同名 key 不让覆盖。

**解决**：如果是每次会变的值（比如运行中的状态），改用 `set_tag`。

### 坑 3：`log_metric` 同 key 不同 step 不会冲突

**说明**：这正是 Metric 的设计——同名 key 加 step 就能画曲线。

**示例**：
```python
for epoch in range(10):
    log_metric("loss", compute_loss(), step=epoch)
```
这样 UI 会自动画一张 loss 随 epoch 下降的图。

### 坑 4：Artifact 路径写错

**症状**：`log_artifact("/tmp/xxx")` 报路径错误，或者文件没出现在 UI 里。

**解决**：传相对当前工作目录的路径，或者直接传文件名（会在当前目录找）。文件会被**复制**到该 Run 的 artifact 目录，不是软链。

### 坑 5：SQLite 不适合高并发

**症状**：多进程同时写 `mlflow.db` 时偶发 `database is locked`。

**原因**：默认的 SQLite 是单写锁。

**解决**：本地开发无所谓；多人协同请用 PostgreSQL/MySQL。

---

## 五、小结：3-5 个 take-aways

1. **MLflow 解决的核心问题是"实验追溯"**：每次训练都自动归档参数、指标、模型、备注，再也不会忘。
2. **4 个核心对象用一句话记牢**：Experiment = 项目，Run = 一次训练，Param/Metric = 配置和效果，Artifact = 文件，Tag = 备注。
3. **`with mlflow.start_run()` 是黄金范式**：自动管理 Run 生命周期，忘了 `end_run()` 也不会留烂尾。
4. **`autolog()` 能省 90% 的样板代码**：跑 sklearn/xgboost/lightgbm 时优先用它，手动 log 只补它没记到的。
5. **UI 的 Compare 是杀手锏**：多 Run 并排看 Param/Metric 一目了然，比翻日志快 100 倍。