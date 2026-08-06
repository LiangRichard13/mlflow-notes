# 阶段 3 学习笔记：追踪服务器与数据集血缘

> 对应脚本：`scripts/03_tracking/03a_start_server.sh`、`scripts/03_tracking/03b_dataset_lineage.py`、`scripts/03_tracking/03c_search_logged_models.py`

## 🎯 这篇笔记做什么

阶段 2 我们已经会用 `mlflow.log_metric`、`mlflow.log_param` 记参数和指标。但生产环境不是单机一个人跑——团队里张三、李四、CI 流水线、调度平台都可能往 MLflow 写数据。这时就要有一个**中心化的「追踪服务器」**来汇集所有实验。阶段 3 的第一件事，就是把"本地一坨文件"升级成"团队共享的 Tracking Server"。

第二件事是**数据集血缘**。模型出问题时，老板问"这个模型用了什么数据训练的？"——你脱口答得出来吗？MLflow 让每个 Run 都跟输入的数据集绑死，谁、什么时间、用哪份数据、跑出多少分，全留痕。这就是数据治理的"出生证明"。

> **类比**：阶段 2 的 MLflow 像你一个人用笔记本写实验记录；阶段 3 的 MLflow 像公司给你配了 **GitLab**（追踪服务器）+ **数据血缘**（像 Git 里每行代码都带作者和 commit）。

跑完这一阶段，你会拿到：一个 SQLite 后端的 Tracking Server（团队共享雏形）、一个把数据集与 Run 绑死的训练脚本、以及一个能跨实验用 SQL 搜索模型的工具。

### 你会学到什么

- 能启动一个**跟踪服务器（Tracking Server）**，让多个客户端往同一个 SQLite 库里写
- 能区分 **Backend Store**（元数据） 和 **Artifact Store**（文件）
- 能用 `mlflow.data.from_pandas` + `mlflow.log_input` 给 Run 关联训练数据并理解 digest 的意义
- 能用 MLflow 3 的 `search_logged_models` 跨实验、跨参数、跨指标筛选模型
- 知道 MLflow 3 与 2 在模型 URI、`log_model` 参数、UI 入口上的关键差异

### 对应脚本清单

| 脚本 | 一句话作用 | 是否必跑 | 前置 |
|------|-----------|---------|------|
| `03a_start_server.sh` | 启动 SQLite 后端的 Tracking Server | 必跑 | 无 |
| `03b_dataset_lineage.py` | 用 `mlflow.data.from_pandas` 记录训练数据血缘 | 必跑 | `03a` 先启动（或用本地 sqlite） |
| `03c_search_logged_models.py` | 用 MLflow 3 的 `search_logged_models` SQL 搜索 | 推荐 | `03b` |

### 前置知识

- 已完成阶段 2：会用 `mlflow.log_param`、`mlflow.log_metric`、`mlflow.sklearn.log_model`
- 装好 mlflow + scikit-learn + pandas（环境：`conda activate mlflow`）
- **不需要** API key（没有 LLM 调用，全是本地 sklearn 训练）
- 数据库知识：你听过 SQLite 就够（就是个文件）
- 命令行基础：在哪个目录跑命令、怎么 `export`

### 跑完必看（UI）

1. 启动 UI：
   - 如果你跑了 `03a_start_server.sh`：直接打开 `http://localhost:5000`
   - 否则本地跑：`mlflow ui --port 5000`
2. 左侧导航栏的 **`Logged Models`**（这是 MLflow 3 新增的栏目）→ 看跨实验的所有 LoggedModel 卡片
3. 选 experiment **`03_dataset_lineage`** → 点开 Run `rf-with-dataset`（训练那个）
4. 看：
   - **Metrics 标签**：只有一行 `accuracy`，代表测试集准确率
   - **Artifacts 标签**：`classifier/` 模型目录（注意 MLflow 3 不在 Run 页直接展示模型，只展示别的 artifact）
   - **Datasets 标签**（关键新栏目）：列出 `wine_dataset (training)` 和 `wine_test_split (testing)` 两行，每行带 digest
   - **Tags 标签**：`data_snapshot = <digest>` 我们用 `set_tag` 留了一份快照
5. 进 experiment `03_search_demo`（`03c` 跑完才有）→ 左边 `Logged Models` → 看 5 个模型按 accuracy 排序的卡片

## 一、核心概念：用人话讲清楚

### 1. Tracking Server：团队协作的"实验中心"

之前的脚本都是直接写本地文件（`./mlruns/` 目录）。一旦团队大于 1 人，就要一个 **HTTP 服务** 让所有人往同一个地方写。这就是 **Tracking Server**：一个 FastAPI 进程，对外暴露 REST API 和 UI。

启动后，所有 `mlflow.log_*` 调用都自动走 HTTP，根本不用改业务代码——只要环境变量 `MLFLOW_TRACKING_URI` 指过去就行。

### 2. Backend Store vs Artifact Store：账本 vs 仓库

Tracking Server 自己不存数据，它把数据分两家：

| 概念 | 类比 | 存什么 | 选型 |
|------|------|--------|------|
| **Backend Store** | 餐厅的点菜单（数据库） | experiments、runs、metrics、params、tags | SQLite（小团队）、PostgreSQL（生产） |
| **Artifact Store** | 餐厅的仓库（文件系统） | 模型文件、图片、配置、特征文件 | 本地路径（学习）、S3/MinIO（生产） |

为什么要分？因为数据库擅长"频繁小写入"（每次 metric 都写一行），文件系统擅长"大文件顺序读写"（模型权重几百 MB）。混在一起两者都做不好。

### 3. 数据集血缘：给训练数据发"身份证"

每次训练时，不光要记录参数和指标，还要告诉 MLflow "我用了这份数据"。`mlflow.data.from_pandas(df, source=..., name=..., targets=...)` 创建一个 `Dataset` 对象，里面带四样东西：

- **source**（来源）：数据来自哪个文件 / URL / 库。复现靠它。
- **name**（名字）：你给数据集起的名，比如 `"wine_dataset"`。
- **digest**（摘要）：**数据集内容的哈希**。一模一样的数据 → 同一个 digest；只要改一个字节 → digest 全变。这是"防偷偷换数据"的关键。
- **schema**（模式）：列名 + 类型。部署时能校验线上输入是否符合训练时的格式。

`mlflow.log_input(dataset, context="training")` 把这个"身份证"挂在 Run 上，从此 Run 知道自己吃了什么数据长大。

### 4. LoggedModel：MLflow 3 把"模型"独立成一等公民

MLflow 2 里，模型是 Run 下的一个 artifact（`runs:/<run_id>/model`）。MLflow 3 把模型抽出来当独立对象，叫 **LoggedModel**，有自己的 `model_id`，可以跨 Run 跨 Experiment 引用（`models:/<model_id>`）。

为什么这事重要？因为**搜索模型** 不该受 Run 的束缚。一个超参搜索可能产生几百个 Run，但模型本身才是你关心的产物。`search_logged_models` 就是为了这个——SQL 风格筛选，秒级返回。

## 二、代码模式：可复用的模板

### 启动 Tracking Server

```python
# mlflow.set_tracking_uri 不需要！环境变量 MLFLOW_TRACKING_URI 已设
# 但如果直接 python 跑：
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
```
**什么时候用**：你的脚本要连远端 server，而不是本地写文件。

### 关联数据集到 Run

```python
dataset = mlflow.data.from_pandas(df, source="data.csv", name="train", targets="label")
with mlflow.start_run(run_name="rf-1"):
    model.fit(X_train, y_train)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_input(dataset, context="training")  # 关键
```
**什么时候用**：任何用 DataFrame 训模型的脚本都应该写这一行，便宜但责任重大。

### 反向查"这个 Run 用了什么数据"

```python
client = mlflow.MlflowClient()
run = client.get_run(run_id)
for ds_input in run.inputs.dataset_inputs:
    ds = ds_input.dataset
    print(ds.name, ds.digest, ds.source)
```
**什么时候用**：排错 / 审计 / 复现报告。

### 跨实验搜模型（MLflow 3）

```python
df = mlflow.search_logged_models(
    experiment_ids=[exp_id],                              # 传 id 不是 name
    filter_string="metrics.accuracy > 0.9 AND params.n_estimators = '50'",
    order_by=[{"field_name": "metrics.accuracy", "ascending": False}],
    max_results=10,
    output_format="list",                                  # 或 "pandas"
)
```
**什么时候用**：从一堆历史模型里挑冠军、挑符合线上指标的版本。

### 用 LoggedModel 加载模型

```python
best = mlflow.search_logged_models(...)[0]
loaded = mlflow.sklearn.load_model(f"models:/{best.model_id}")
```
**什么时候用**：脚本里挑出最佳模型直接部署。

## 三、实战步骤：按顺序照做

### Step 1：启动 Tracking Server（新开一个终端）

```bash
conda activate mlflow
cd <project-root>
bash scripts/03_tracking/03a_start_server.sh
```

它会在当前目录建两个东西：`mlflow.db`（SQLite 数据库）和 `mlruns/`（artifact 文件夹）。看到 `Listening at: http://0.0.0.0:5000` 就 OK。

> MLflow 3.5+ 需加 `--allowed-hosts "localhost,127.0.0.1"`，不然浏览器可能打不开。

### Step 2：让客户端连上 server（再开一个终端）

```bash
conda activate mlflow
export MLFLOW_TRACKING_URI=http://localhost:5000
cd <project-root>
```

不设这个变量，脚本会默认写本地 `./mlruns`，跟 server 不通。

### Step 3：跑数据集血缘脚本

```bash
python scripts/03_tracking/03b_dataset_lineage.py
```

它用本地 sqlite（不依赖 server），跑完你会看到：
- 控制台打印 `dataset.digest`（一长串哈希）
- 数据库 `mlflow.db` 里多了个 experiment `03_dataset_lineage` 和 run `rf-with-dataset`

### Step 4：跑搜索 LoggedModel 脚本

```bash
python scripts/03_tracking/03c_search_logged_models.py
```

它训练 5 个不同模型（不同 C、深度），然后用 `search_logged_models` 筛三种条件。控制台会打印三张表，对应"accuracy > 0.95"、"name 含 classifier"、"n_estimators=50"。

### Step 5：UI 验证

浏览器开 `http://localhost:5000`，按上文"跑完必看（UI）"一项一项对。

## 四、避坑清单

- ⚠️ **忘了 `export MLFLOW_TRACKING_URI`** → 写本地 mlruns，server 看不到。脚本里看不到就在 Python 里加 `mlflow.set_tracking_uri("http://localhost:5000")`。
- ⚠️ **想用 Model Registry 但用的是文件 backend** → 必须先 `mlflow db upgrade sqlite:///mlflow.db`，否则注册会失败。
- ⚠️ **`search_logged_models` 用 `params.lr <= 0.01`** → 报错。`params` 是字符串，只支持 `=`、`!=`、`LIKE`、`IN`；数值比较只对 `metrics` 有效。
- ⚠️ **`experiment_ids=["03_search_demo"]` 传名字** → 返回空。必须用 `mlflow.get_experiment_by_name(...).experiment_id` 拿到 id。
- ⚠️ **改了数据但忘了 digest 检测** → 每次跑新实验 MLflow 都会重算 digest，变了就说明数据被改过——别绕开它。

## 五、小结

- **Tracking Server** = 团队共享的实验中心；Backend Store 管账本，Artifact Store 管仓库，分开配。
- **数据集血缘** = `mlflow.data.from_pandas` + `mlflow.log_input`；`digest` 是数据指纹，改了就报警。
- **MLflow 3 的 LoggedModel** 是独立对象，能跨实验用 SQL 风格搜索，比 MLflow 2 强大得多。
- 不管 server 启不启，`MLFLOW_TRACKING_URI` 决定写哪里；本地学习不启 server 也 OK，但脚本结构要先对齐。
- `params` 在 `filter_string` 里当字符串处理，数值比较一律用 `metrics` 字段。
