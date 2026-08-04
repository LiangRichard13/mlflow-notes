# 阶段 1 学习笔记：MLflow 核心概念

## 一、4 个核心对象

```
Experiment（实验）
└── Run（运行） ── 一次执行/一次实验
    ├── Param      字符串型配置（lr、batch_size 等）
    ├── Metric     数值型指标（loss、accuracy 等），可带 step
    ├── Artifact   任意文件（模型、图表、日志、配置）
    └── Tag        任意标签，用于过滤/搜索
```

类比：
- **Experiment** = 项目文件夹
- **Run** = 一次训练/实验
- **Param** = 这次训练用了什么配置
- **Metric** = 这次训练效果怎么样
- **Artifact** = 这次训练产出了什么文件
- **Tag** = 这次训练的备注/状态

## 二、5 个最常用 API

| API | 作用 | 示例 |
|-----|------|------|
| `mlflow.set_experiment(name)` | 切换/创建实验 | `set_experiment("iris")` |
| `mlflow.start_run(run_name=...)` | 启动一个 Run | `with mlflow.start_run() as run:` |
| `mlflow.log_param(k, v)` | 记一个参数 | `log_param("lr", 0.01)` |
| `mlflow.log_metric(k, v, step=...)` | 记一个指标 | `log_metric("loss", 0.5, step=epoch)` |
| `mlflow.log_artifact(path)` | 记一个文件 | `log_artifact("model.pkl")` |

更便捷的批量版本：
- `log_params({...})` 一次记多个 param
- `log_metrics({...})` 一次记多个 metric

## 三、追踪后端（Tracking Backend）

MLflow 默认把数据存到文件系统 `./mlruns/` + SQLite `mlflow.db`。两种 URI：

```python
# 本地文件（默认）
mlflow.set_tracking_uri("file:./mlruns")

# SQLite（推荐，支持并发）
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 远程 Tracking Server
mlflow.set_tracking_uri("http://server:5000")

# 完整部署（PostgreSQL + S3）
mlflow.set_tracking_uri("postgresql://user:pass@host/db")
```

> MLflow 3 默认会创建 sqlite 数据库。在生产环境应该用 PostgreSQL/MySQL。

## 四、Artifact 存储（Artifact Store）

Artifact 是大文件（模型、图表），可以独立配置存储位置：

| URI Scheme | 存储 | 场景 |
|-----------|------|------|
| `./mlruns` | 本地文件系统 | 开发 |
| `file:///path` | 指定本地路径 | 单机 |
| `s3://bucket/path` | AWS S3 | 云端 |
| `gs://bucket/path` | GCS | GCP |
| `azure://...` | Azure Blob | Azure |
| `hdfs://...` | HDFS | 大数据 |

## 五、UI 看什么

跑完 `mlflow ui --port 5000` 后浏览器打开 `http://localhost:5000`：

- **Experiments 列表**：所有实验
- **实验详情页**：这个实验下所有 Run 的对比表
  - 顶部表头可点击排序/筛选
  - "Compare" 按钮可选择多个 Run 做并排对比
- **单 Run 详情页**：
  - Parameters / Metrics / Tags / Artifacts 4 个 tab
  - Metric 曲线图（点 metric 名旁的 eye 图标）
  - Artifacts 文件树（可下载）

## 六、跑完阶段 1 必看

跑完 `01_hello_mlflow.py` 和 `01b_sklearn_basics.py` 后，在 UI 里确认：

1. 看到 2 个实验：`01_basics_demo`（3 个 Run）和 `01_sklearn_iris`（4 个 Run）
2. 在 `01_sklearn_iris` 里点 Compare，选中 4 个 Run，看 Param/Metric 对比表
3. 点开任一 Run，看 Artifacts → model/ 下的：
   - `model.pkl`：序列化的 sklearn 模型
   - `MLmodel`：MLflow 自带的元数据（包含 signature、input example）
   - `conda.yaml`、`python_env.yaml`：环境依赖（部署用）

## 七、常见坑

1. **Run 没关就崩了** → 用 `with mlflow.start_run() as run:` 让异常自动 end_run
2. **log_param 只能记一次**（同一个 key 第二次会警告）。同 key 想记多次用 `set_tag`
3. **log_metric 同 key 不同 step 不会冲突**（自动画曲线）
4. **Artifact 路径不能用绝对路径以外的写法** → 传文件路径或目录路径，会复制到当前 Run 的 artifact 目录
5. **数据库锁**：SQLite 不适合高并发写，生产用 PostgreSQL

## 八、深入阅读

- 官方文档：https://mlflow.org/docs/latest/ml/tracking/
- Tracking API Reference：https://mlflow.org/docs/latest/python_api/mlflow.html
