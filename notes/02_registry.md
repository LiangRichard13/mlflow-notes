# 阶段 2 学习笔记：模型格式与注册表

> 对应脚本：`02_registry/02a_log_model.py`、`02b_register_alias.py`、`02c_load_predict.py`
> 需要 API Key：否

## 🎯 这篇笔记做什么

Phase 1 我们学会了"记录"——把参数、指标写进 MLflow，训练完能回头查。但那时候还有个大问题没解决：**训练出来的模型本身去哪了？** 你把 `model.pkl` 用 pickle 存到本地，三个月后同事问你"线上跑的是哪个模型、当时用什么数据训的、输入要几列",你多半答不上来。更糟的是，你想换个更好的模型上线，得手动改代码里的文件路径，还得重启服务。

这一阶段就是解决这些问题的。MLflow 提供了两样东西：**MLflow Model 格式**（把模型 + 依赖 + 输入输出说明打包成一个自描述的目录）和 **Model Registry**（模型的"版本仓库"，像 Git 之于代码）。

打个比方：如果 Phase 1 的 Run 是"实验日记"，那 Registry 就是"产品货架"。日记里有几百次实验，货架上只放你精挑细选、贴好标签的那几个。而 **Alias（别名）** 就是货架上的标签牌——"champion"（现役冠军）这块牌子今天挂在 v1 上，明天可以挂到 v3 上，所有来取货的人（加载模型的服务）自动拿到新版本，**不用改一行代码、不用重启**。

**产出物**：跑完三个脚本，你会得到一个名为 `WineQualityClassifier` 的注册模型，它有 v1 版本、带完整签名（输入 13 列 float、输出 int）、挂着 `champion` 别名，并且能用一行 `mlflow.sklearn.load_model("models:/WineQualityClassifier@champion")` 在任何地方加载出来直接推理。

### 你会学到什么

- 读懂 MLflow Model 目录结构，尤其是 `MLmodel` 这个 YAML 元数据文件在说什么
- 用 `infer_signature()` 自动推断模型的输入输出 schema，让部署服务能自动校验请求格式
- 把 Run 里的模型注册（`register_model`）到 Model Registry，理解版本号是怎么自动累加的
- 用 Alias（`champion` / `challenger`）管理"哪个版本在生产用"，并解释清楚为什么它取代了已废弃的 Stage
- 掌握三种 model URI 写法，知道什么场景该用哪一种

### 对应脚本清单

| 脚本 | 一句话作用 | 是否必跑 | 前置 |
|------|-----------|---------|------|
| `02a_log_model.py` | 训练 sklearn Pipeline，推断签名 + input_example，把模型作为 artifact 记录到 Run | ✓ 必跑 | Phase 1 |
| `02b_register_alias.py` | 把最新 Run 里的模型注册为 `WineQualityClassifier`，设 `champion` 别名并加描述 | ✓ 必跑 | 跑过 02a |
| `02c_load_predict.py` | 用 `models:/name@champion` 加载模型并推理，演示别名热切换 | 推荐 | 跑过 02b |

### 前置知识

- 已完成 Phase 1，理解 Experiment / Run / Param / Metric / Artifact 这几个对象
- 会用 sklearn 的 `Pipeline`、`train_test_split`（脚本里用的是内置 Wine 数据集，178 样本 / 13 特征 / 3 类）
- 已安装：`mlflow`（3.x）、`scikit-learn`、`pandas`、`numpy`
- **关键环境要求**：Model Registry **必须**有数据库后端。本项目统一用 `mlflow.set_tracking_uri("sqlite:///mlflow.db")`，纯文件系统（`file:./mlruns`）**不支持** Registry

### 跑完必看（UI）

1. 启动 UI：`mlflow ui --port 5000`（如果用的是 sqlite，加参数 `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`）
2. 左侧导航栏点 `Models` → 看到 `WineQualityClassifier`
3. 点进去看 **Version 1**，重点看三处：
   - **Aliases** 那一栏显示 `champion`
   - **Description** 是 02b 里 `update_model_version` 写进去的那段文字
   - **Source Run** 链接能跳回 02a 的那次 Run
4. 再从 `Experiments` → `02_model_registry` → 打开 Run → `Artifacts` tab → 点开 `wine-classifier` 目录，**亲眼看一遍 `MLmodel` 文件的内容**，这是理解模型格式最直接的方式

---

## 一、核心概念：用人话讲清楚

### 1.1 MLflow Model 不是一个文件，是一个目录

新手最容易误解的一点：MLflow 保存的"模型"不是 `model.pkl` 那一个文件，而是**一整个自描述的目录**。

```
wine-classifier/
├── MLmodel              # 主元数据（YAML）—— 灵魂所在
├── model.pkl            # 序列化后的模型本体
├── python_env.yaml      # Python 版本 + 依赖
├── conda.yaml           # Conda 环境描述
├── requirements.txt     # pip 依赖清单
└── input_example.json   # 输入示例（log_model 传了 input_example 才有）
```

为什么要这么麻烦？因为**光有 pkl 是没法部署的**。别人拿到你的 pkl，不知道该用哪个 Python 版本、要装哪些包、输入要传几列什么类型。MLflow 把这些"上下文"全打包进去，于是这个目录可以直接丢给 `mlflow models serve` 起一个 REST 服务，或者打成 Docker 镜像。

### 1.2 MLmodel 文件：模型的"身份证"

`MLmodel` 是 YAML，跑完 02a 后你能在 UI 的 Artifacts 里看到它，长这样：

```yaml
artifact_path: wine-classifier
flavors:
  python_function:                # 通用 flavor：任何语言/框架都能用的统一入口
    env: conda.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    predict_fn: predict
  sklearn:                        # 原生 flavor：还原成真正的 sklearn 对象
    code: null
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: 1.5.0
mlflow_version: 3.x
model_size_bytes: 1234
run_id: abc123...
signature:                        # 输入输出 schema
  inputs: '[{"name": "alcohol", "type": "double"}, ...]'
  outputs: '[{"type": "long"}]'
```

**"flavors"（风味）是这里最值得理解的概念**。同一个模型可以有多种"读法"：
- `sklearn` flavor：加载后你拿到的是**真正的 sklearn Pipeline 对象**，可以访问 `.steps`、`.feature_importances_` 这些原生属性
- `python_function` flavor：加载后你拿到的是一个**统一的 `predict()` 接口**，不管底层是 sklearn、PyTorch 还是 XGBoost，用法完全一样

部署工具（比如 `mlflow models serve`）只认 `python_function`，所以它能一视同仁地服务任何框架的模型。这就是 flavor 设计的价值：**训练侧自由选框架，部署侧只需要一套代码**。

### 1.3 Signature（签名）：模型的"接口文档"

签名记录了模型**输入要什么、输出是什么**。它有两个实实在在的好处：

1. **部署时自动校验**：请求少传一列、类型传错了，服务会直接报清晰的错误，而不是在模型内部炸出一个看不懂的堆栈
2. **给人看的文档**：三个月后你自己回来看，UI 上直接列出 13 个特征名和类型，不用翻训练代码

推断签名只要一行——把训练输入和模型输出丢给 `infer_signature`，它自己去看列名和 dtype：

```python
from mlflow.models import infer_signature
signature = infer_signature(X_train, pipe.predict(X_train))
```

`input_example` 是签名的好搭档：存几行真实输入样本进去。它既能在你忘记传 signature 时帮 MLflow 自动推断，也能让 UI 直接显示"请求长什么样"，还能让部署后的冒烟测试有现成数据用。

### 1.4 Model Registry：模型的"Git 仓库"

Run 里的模型是**实验产物**——你可能跑了 200 次，其中 199 次都是垃圾。Registry 是**发布通道**——你从那 200 次里挑出好的，给它起个正式名字（`WineQualityClassifier`），它就有了 v1、v2、v3 的版本序列。

对照理解：

| 概念 | 类比 | 特点 |
|------|------|------|
| Run 里的模型 | 本地的一次 commit | 数量多，随手产生，用 `runs:/<run_id>/<name>` 引用 |
| Registered Model | 一个 Git 仓库 | 有名字，是一个逻辑上的"产品线" |
| Model Version | 打的 tag（v1、v2） | 注册一次自动 +1，不可变 |
| Alias | 指向某个 tag 的分支指针 | 可以随时改指向，如 `champion` → v3 |

### 1.5 ⭐ Alias 为什么取代了 Stage（新手最困惑的点）

**先说结论：MLflow 2 时代的 Stage（`None` / `Staging` / `Production` / `Archived`）在 MLflow 3 里已经废弃，取而代之的是 Alias。** 如果你在网上搜到 `transition_model_version_stage(...)` 的教程，那是旧写法，别学。

**Stage 的三个硬伤：**

1. **写死的四个值，改不了**。现实里团队的流程五花八门：有人要 `dev` / `qa` / `canary` / `prod` 四级，有人做 A/B 测试要同时上两个模型，有人还要区分"华东区在用"和"华南区在用"。Stage 只给你四个固定选项，全都塞不下。Alias 是**自定义字符串**，你想叫什么叫什么。

2. **一个 stage 只能挂一个版本，一个版本只能有一个 stage**。这个 1 对 1 的死限制让 A/B 测试特别难做——你没法说"v2 和 v3 同时是生产模型"。Alias 是**多对多**的：一个版本可以同时挂 `champion` 和 `stable`，你也可以再加 `challenger` 挂到 v3 上做灰度。

3. **语义模糊，容易误会**。"Production" 到底是"正在生产环境跑"还是"通过了测试可以上生产"？不同团队理解不一样，还得靠口头约定。Alias 强迫你自己命名，反而更明确。

**Alias 的核心好处——热切换（这是最实用的部分）：**

生产服务里你的加载代码写死一行：

```python
model = mlflow.sklearn.load_model("models:/WineQualityClassifier@champion")
```

新模型 v2 上线时，你**不改代码、不重启服务**，只要执行：

```python
client.set_registered_model_alias("WineQualityClassifier", "champion", version=2)
```

下一次加载就自动是 v2 了。要回滚？把别名指回 v1，一秒钟的事。这个切换是**原子操作**，不存在"改到一半"的中间状态。

常用的别名约定（社区惯例，非强制）：

| 别名 | 含义 |
|------|------|
| `champion` | 当前生产在用的冠军模型 |
| `challenger` | 正在评测、准备挑战冠军的候选 |
| `baseline` | 用于对比的基准模型 |
| `archived` | 已下线但保留，方便回溯 |

---

## 二、代码模式：可复用的模板

### 2.1 记录带签名的模型（02a 的核心）

```python
import mlflow
from mlflow.models import infer_signature

mlflow.set_tracking_uri("sqlite:///mlflow.db")   # Registry 必须要数据库后端
mlflow.set_experiment("02_model_registry")

with mlflow.start_run(run_name="wine-rf-v1") as run:
    mlflow.log_params({"n_estimators": 200, "max_depth": 10})
    mlflow.log_metrics({"accuracy": acc, "f1": f1})

    signature = infer_signature(X_train, pipe.predict(X_train))

    mlflow.sklearn.log_model(
        pipe,
        name="wine-classifier",        # ⚠️ MLflow 3 用 name，不是 artifact_path
        signature=signature,
        input_example=X_train.head(3),
    )

    print(f"模型 URI: runs:/{run.info.run_id}/wine-classifier")
```

**唯一容易踩的坑**：MLflow 3 里 `log_model()` 的参数从 `artifact_path=` 改成了 `name=`。旧教程里全是 `artifact_path`，会触发 deprecation 警告或直接报错。

### 2.2 注册 + 设别名（02b 的核心）

```python
import mlflow
from mlflow import MlflowClient

client = MlflowClient()
REGISTERED_NAME = "WineQualityClassifier"

# 找到最近一次 Run
runs = mlflow.search_runs(
    experiment_names=["02_model_registry"],
    order_by=["start_time DESC"],
    max_results=1,
)
model_uri = f"runs:/{runs.iloc[0].run_id}/wine-classifier"

# 注册：同名重复注册 = 自动创建新版本
result = mlflow.register_model(model_uri, REGISTERED_NAME)
print(f"v{result.version}, source={result.source}, run_id={result.run_id}")

# 设别名（替代已废弃的 stage）
client.set_registered_model_alias(REGISTERED_NAME, "champion", version=result.version)

# 补充描述，UI 上能看到
client.update_model_version(
    name=REGISTERED_NAME,
    version=result.version,
    description="StandardScaler + RandomForest(n=200, depth=10) on Wine dataset",
)
```

### 2.3 查询版本与别名

```python
# 列出所有版本
for v in client.search_model_versions(f"name='{REGISTERED_NAME}'"):
    aliases = [a.alias for a in v.aliases] if hasattr(v, "aliases") else []
    print(f"v{v.version} | run={v.run_id[:8]} | aliases={aliases} | status={v.status}")

# 拿到"别名 → 版本号"的映射
print(client.get_registered_model(REGISTERED_NAME).aliases)

# 直接按别名拿版本对象
mv = client.get_model_version_by_alias(REGISTERED_NAME, "champion")
print(mv.version)

# 删除别名（不删版本）
# client.delete_registered_model_alias(REGISTERED_NAME, "challenger")
```

### 2.4 三种加载方式（02c 的核心）

```python
# 1) 从 Run 加载 —— 调试期用，URI 里的 run_id 很长且不好记
model = mlflow.sklearn.load_model(f"runs:/{run_id}/wine-classifier")

# 2) 用别名加载 —— ⭐ 生产推荐，代码写死不用改
model = mlflow.sklearn.load_model("models:/WineQualityClassifier@champion")

# 3) 用版本号加载 —— 需要复现某个确定版本时用（如 debug 线上问题）
model = mlflow.sklearn.load_model("models:/WineQualityClassifier/1")
```

加载回来的是**完整的 Pipeline**，StandardScaler 和 RandomForest 两步都在，直接喂原始数据即可：

```python
print([s[0] for s in model.steps])   # ['scaler', 'clf']
preds = model.predict(X_test.head(5))
```

**`mlflow.sklearn.load_model` 和 `mlflow.pyfunc.load_model` 的区别**：前者还给你原生 sklearn 对象（能访问 `.steps`、`.predict_proba`），后者给你统一的 `pyfunc` 包装（只有 `.predict()`，但不依赖 sklearn，写通用部署代码时用）。

---

## 三、实战步骤：按顺序照做

```bash
conda activate mlflow
cd /home/sstl/lcd/MLFlowLearning
```

**Step 1 — 训练并记录模型**

```bash
python 02_registry/02a_log_model.py
```

预期输出里有 `模型性能: accuracy=1.0000, f1=1.0000`（Wine 数据集很简单，满分正常）和一行 `模型 URI: runs:/<run_id>/wine-classifier`。

**Step 2 — 注册并设别名**

```bash
python 02_registry/02b_register_alias.py
```

预期看到 `✓ 已注册为 WineQualityClassifier v1` 和 `✓ 已设置 champion alias → v1`，最后打印出版本列表和别名映射。

**Step 3 — 用别名加载并推理**

```bash
python 02_registry/02c_load_predict.py
```

预期看到 `Pipeline steps: ['scaler', 'clf']`，以及 5 个预测标签和真实标签的对比。

**Step 4 — 去 UI 验证（别跳过）**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

打开 `http://localhost:5000` → `Models` → `WineQualityClassifier`，确认版本、别名、描述、签名都在。

**Step 5 — 自己做个小实验（强烈推荐）**

把 02a 里的 `n_estimators` 改成 50，重跑 02a 和 02b，你会看到自动生成了 **v2**，并且 `champion` 别名**自动跳到了 v2**（因为 02b 每次都把别名设到刚注册的版本上）。然后手动执行下面这段，把 champion 切回 v1，再跑一次 02c，看看预测有没有变——这就完整体验了一次"上线新版本 + 回滚"：

```python
from mlflow import MlflowClient
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
MlflowClient().set_registered_model_alias("WineQualityClassifier", "champion", version=1)
```

---

## 四、避坑清单

1. **Registry 必须有数据库后端**。用 `file:./mlruns` 时调 `register_model` 会直接报错。本项目统一 `sqlite:///mlflow.db`，生产环境用 PostgreSQL/MySQL。另外注意：脚本是相对路径打开 sqlite，**必须在项目根目录运行**，否则会在别处生成一个空的 `mlflow.db`，然后你会困惑"为什么 UI 里什么都没有"。

2. **`artifact_path` 已改名为 `name`**。MLflow 3 的 `log_model(model, name="...")`，网上大量旧教程还在用 `artifact_path=`。

3. **Stage 相关 API 全部废弃**。`transition_model_version_stage()`、`stage="Production"` 这类写法不要再用，一律换成 `set_registered_model_alias()`。UI 里也已经看不到 Stage 下拉框了。

4. **版本号只增不减，删了也不会复用**。删掉 v2 之后，下次注册是 v3 而不是补上 v2。所以版本号可以放心当唯一标识用。

5. **02b 用 `search_runs` 取"最近一次 Run"，有隐患**。如果你在跑完 02a 之后又在 `02_model_registry` 这个实验里跑了别的 Run，02b 就会注册错的那个。稳妥做法是显式指定 run_id，或者在过滤条件里加上 run_name：

   ```python
   runs = mlflow.search_runs(
       experiment_names=["02_model_registry"],
       filter_string="attributes.run_name = 'wine-rf-v1'",
       order_by=["start_time DESC"], max_results=1,
   )
   ```

6. **model URI 里的路径名必须和 `log_model` 的 `name` 完全一致**。02a 写的是 `name="wine-classifier"`，02b 就必须拼 `runs:/{run_id}/wine-classifier`。写错一个字符就是 `RESOURCE_DOES_NOT_EXIST`，而错误信息不会告诉你"你是不是拼错了名字"。

7. **别名区分大小写，且不能用作纯数字**。`Champion` 和 `champion` 是两个不同的别名；别名也不能起成 `1`、`2` 这种，会和版本号语法冲突。

8. **02b/02c 的文件头注释里写的是 `python 03_registry/...`，这是笔误**，正确目录是 `02_registry/`。

9. **签名太严格也会咬人**。如果推断签名时用的是 DataFrame（有列名），那推理时也必须传 DataFrame，传 numpy 数组会因为缺列名而校验失败。保持训练和推理的数据形态一致。

---

## 五、小结：5 个 take-aways

1. **MLflow Model 是一个自描述的目录，不是一个 pkl 文件**。`MLmodel` YAML 记录了 flavors（多种加载方式）、依赖和签名，这让模型能脱离训练代码独立部署。

2. **Signature 是模型的接口契约**，一行 `infer_signature(X_train, model.predict(X_train))` 就能生成，配上 `input_example` 后部署时能自动校验请求，也是给未来的自己看的文档。

3. **Run 里的模型是实验产物，Registry 里的才是发布产物**。`mlflow.register_model()` 把前者提升为后者，同名重复注册自动累加版本号。

4. **Alias 取代 Stage，因为 Stage 是写死的四个值 + 1 对 1 绑定，装不下真实团队的发布流程**。Alias 是自定义字符串 + 多对多，还支持原子热切换：改别名指向就等于换生产模型，代码不用动、服务不用重启、回滚一秒钟。

5. **生产加载一律用 `models:/<name>@<alias>`**。`runs:/<run_id>/<name>` 留给调试，`models:/<name>/<version>` 留给需要精确复现的场景。
