# 阶段 3 学习笔记：追踪服务器与数据集血缘

## 一、MLflow 后端架构

```
MLflow = Tracking Server（API） + Backend Store（元数据） + Artifact Store（文件）

            POST /api/...                SQLite/Postgres         本地/S3/MinIO
客户端 ───────► Tracking Server ─────► metadata ─────► model files
   │           （FastAPI）             experiments              artifacts
   │                                  runs                     
   │                                  metrics/params          
   ▼
  UI (React) <─── 同一端口 ──────
```

- **Backend Store**：存元数据（experiments、runs、metrics、params、tags）。SQLite / PostgreSQL / MySQL
- **Artifact Store**：存大文件（模型、图像、配置）。本地文件系统 / S3 / Azure / GCS / HDFS
- **Tracking Server**：FastAPI 包装，提供 REST API 和 UI

## 二、三种启动方式

| 方式 | 适用 | 命令 |
|------|------|------|
| 纯本地（无 server） | 一次性脚本 | `python script.py`（用 `sqlite:///mlflow.db`） |
| 本地 SQLite + Server | 学习/小团队 | `mlflow server --backend-store-uri sqlite:///mlflow.db ...` |
| 生产 PostgreSQL + S3 | 生产 | `mlflow server --backend-store-uri postgresql://... --default-artifact-root s3://...` |

⚠️ MLflow 3.5+ 必须配 `--allowed-hosts` 防 DNS rebinding。

## 三、Model Registry 需要 Backend Store

Model Registry 必须有 database backend，不能用纯文件模式。
第一次启用时跑一次：`mlflow db upgrade sqlite:///mlflow.db`

## 四、数据集血缘

```python
dataset = mlflow.data.from_pandas(df, source="data.csv", name="train", targets="label")
with mlflow.start_run():
    mlflow.log_input(dataset, context="training")

# 反向查询（MLflow 3：通过 Client API）
client = mlflow.MlflowClient()
run = client.get_run(run_id)
for ds_input in run.inputs.dataset_inputs:
    print(ds_input.dataset.name, ds_input.dataset.digest)
```

**核心概念**：
- **digest**：数据集内容的哈希。变了说明数据被改了（防"偷偷换数据"）
- **source**：数据来源（文件路径、URL、库名），用于复现
- **schema**：列名 + 类型，部署时校验输入格式
- **context**：training / testing / validation，标记用途

## 五、MLflow 3 新 API：`search_logged_models`

跨实验、按 metrics/params 搜索 LoggedModel：

```python
df = mlflow.search_logged_models(
    experiment_ids=["1"],                              # 传 id 不是 name
    filter_string="metrics.accuracy > 0.9 AND params.n_estimators = '50'",  # SQL 风格
    order_by=[{"field_name": "metrics.accuracy", "ascending": False}],       # list[dict]
    max_results=10,
    output_format="list",   # 或 "pandas"
)
```

**关键约束**：
- `params` 是字符串，**只能用 `=` `!=` `LIKE` `IN`**，不能 `<= >`
- `metrics` 是数值，可以用 `>` `<` `>=` `<=` `=` `!=`
- `order_by` 是 `list[dict]`，不是 `list[str]`

## 六、MLflow 2 vs 3 关键区别

| 场景 | MLflow 2 | MLflow 3 |
|------|---------|---------|
| 模型位置 | `runs:/<run_id>/model` | `models:/<model_id>`（独立于 Run）|
| 存储路径 | `experiments/<exp>/<run>/artifacts/` | `experiments/<exp>/models/<model_id>/artifacts/` |
| log_model 参数 | `artifact_path=` | `name=` |
| 跨实验搜模型 | 需遍历 run 自己筛 | `search_logged_models()` SQL 风格 |
| Run 页面 Artifacts | 包含模型 | 不再显示模型，去 Logged Models 页 |