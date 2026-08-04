# 阶段 9 学习笔记：部署到云与生产可观测性

> 对应脚本：`09_deployment/09a_sampling_redaction.py`、`09b_prod_infra.sh`、`09c_hardware_monitor.py`
> 需要 API Key：是（仅 `09a` 需要；`09b`、`09c` 不需要）

## 🎯 这篇笔记做什么

把 LLM 应用从"笔记本里的 demo"推到"真正上线的服务"，会遇到三类以前完全不用想的问题：**存哪里、要不要全记、机器扛不扛得住**。这一阶段就解决这三大问题。

**类比**：前面 8 个阶段相当于在自己家里做饭，现在要开一家小餐馆。要把厨房整套搬到店里（Postgres + S3），客人太多不能每桌都点一本菜单（采样 + 脱敏），还要随时知道炉子火多大、冰箱满不满（硬件监控）。

**产出物**：跑完 `09a` 能直观看到"原始 vs 脱敏"两份 trace 对比；看完 `09b` 你手里有一份能直接 `docker compose up` 的生产 MLflow 配置；跑完 `09c` 你会在 UI 里看到一条漂亮的 CPU 飙到 100% 的曲线。

### 你会学到什么

- 能看懂 MLflow 生产环境的"三层架构"（Tracking Server + Backend Store + Artifact Store）
- 能用 `docker-compose` 一键起 MLflow + Postgres + MinIO
- 能给 trace 加采样，把存储成本压到原来的 1/10
- 能写一个 PII 脱敏函数，在数据"进 trace 之前"就把邮箱、手机、身份证洗掉
- 能用 psutil + MLflow 把 CPU/内存/磁盘/网络指标画成曲线
- 能区分 MLflow 能做什么、不能做什么，并知道何时引入 Prometheus + Grafana

### 对应脚本清单

| 脚本 | 一句话作用 | 是否必跑 | 前置 |
|------|-----------|---------|------|
| `09a_sampling_redaction.py` | 演示 trace 采样 + PII 脱敏，对比 raw vs redacted 两条 Run | ✓ 必跑 | `OPENAI_API_KEY` |
| `09b_prod_infra.sh` | docker-compose 参考配置（Postgres + MinIO + MLflow） | 推荐 | Docker 环境 |
| `09c_hardware_monitor.py` | 后台采样 CPU/内存/磁盘/网络 30 秒，画到 UI 上 | 推荐 | 无（不需要 API key） |

### 前置知识

- 已完成前面 8 个阶段（至少跑过 `04_tracing` 的例子，知道 trace 是什么）
- 装好 `mlflow`、`psutil`、`openai`：`pip install mlflow psutil openai`
- 需要 `OPENAI_API_KEY`（仅 `09a` 需要，`09c` 不需要 LLM）
- 本阶段假设你懂：什么是 LLM trace、什么是 Postgres/S3；不懂 docker-compose 也能跑 `09a` 和 `09c`
- ⚠️ 不需要真买云服务，`09b` 的 docker-compose 在本地就能起

### 跑完必看（UI）

1. 启动 UI：`mlflow ui --port 5000`
2. 选 experiment `09_sampling_pii` → 对比 Run `raw-no-redaction` 和 Run `redacted`
3. 点开任一 Run → 看 **Traces** 标签，对比两个 Run 的 `trace_inputs`：
   - `raw-no-redaction` 应该能看到 `zhangsan@example.com`、`13812345678`、`110101199001011234`
   - `redacted` 同样的输入会变成 `[EMAIL]`、`[PHONE]`、`[ID_CARD]`
4. 选 experiment `09_hardware_monitor` → 点开 Run `hardware-monitor-demo`
5. 看 **Metrics** 标签的曲线：
   - `cpu_percent` 应该在 5-10s 期间飙到接近 100%（CPU 密集负载）
   - `mem_percent` 应该在 10-15s 和 20-25s 期间明显上升（内存分配负载）
   - `disk_read_mb` / `disk_write_mb` 是磁盘读写字节数
6. 看 **Tags**：`monitoring_type=hardware_resources`、`interval_seconds=1`

---

## 一、核心概念：用人话讲清楚

### 1.1 三层架构

生产环境的 MLflow 不是单机文件，而是一套"三个角色"：

```
┌──────────────────────────────────────────────────┐
│  Client (Web/Mobile/API)                         │
└────────────────┬─────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────┐
│  MLflow Tracking Server (FastAPI, 端口 5000)    │
│  --backend-store-uri  → PostgreSQL (元数据)       │
│  --default-artifact-root → S3/MinIO (大文件)      │
└────────────────┬─────────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼────────────┐    ┌───────▼────────────┐
│ Model Service   │    │ Agent Server (3.6+) │
│ (REST /invocations) │ │  (@invoke/@stream)  │
└────────────────┘    └────────────────────┘
```

**Backend Store（后端存储）**：存"元数据"——Run 的名字、参数、指标、谁跑了什么。这是**结构化数据**，适合放 Postgres 这种关系型数据库。**绝对不要用 SQLite 上生产**，并发写会锁表。

**Artifact Store（构件存储）**：存"大文件"——模型权重、trace JSON、图片、CSV。这是**二进制文件**，适合放 S3（或本地用 MinIO 替代）。**绝对不要存到本地磁盘**，容器一重启就没了。

**类比**：Backend Store 是"图书馆的目录卡"（告诉你每本书在哪），Artifact Store 是"书库本身"（真正放书的地方）。

### 1.2 Trace 采样（控制成本）

生产环境最大的隐性成本是 **trace 存储**。每个 trace 平均 ~50KB，1 亿条 = 50TB。100 QPS 全量记 trace 一个月 8.6 亿条，**采样 10% 直接省 90% 存储**。

**核心思路**：不是每个请求都记 trace，而是按比例抽一部分。

| 场景 | 采样率 |
|------|--------|
| 调试 / PoC | 100% |
| 一般生产 | 10-20% |
| 高流量 (>1k QPS) | 1-5% |

### 1.3 PII 脱敏（隐私合规）

**PII = Personally Identifiable Information**，个人信息。邮箱、手机、身份证、信用卡、姓名、IP 都算。

**关键原则**：在 **trace 边界**（也就是函数入口）就要把 PII 洗掉，而不是先 trace 再洗——因为一旦进入 trace 存储，**就泄漏了**。

**类比**：快递单上不要写真实手机号，写成 "**** 1234"。同理，trace 里存的应该是 "[EMAIL]"而不是 `zhangsan@example.com`。

### 1.4 硬件监控

MLflow autolog 对 PyTorch/TensorFlow 会自动记 GPU 利用率，但 **CPU/内存的历史曲线它不管**。要补这块，需要用 `psutil` 手动采样，再 `mlflow.log_metric` 写进 Run。

### 1.5 MLflow 的边界（什么时候用它、什么时候换工具）

MLflow 不是万能工具。下面这张表告诉你哪些功能它能做、哪些要靠别的工具：

| 需求 | MLflow 能做？ | 推荐工具 |
|------|--------------|----------|
| 实验追踪（参数/指标/artifact） | ✓ 完美 | MLflow |
| Trace 记录和回放 | ✓ 完美 | MLflow |
| 模型注册和版本管理 | ✓ 完美 | MLflow |
| 模型部署 | ✓ 可以 | MLflow + Docker |
| 模型服务监控 | 部分（只能看 Run 内） | Prometheus + Grafana |
| 阈值告警（CPU > 90% 报警） | ✗ 不能 | Prometheus Alertmanager |
| 分布式 tracing（多服务） | 部分 | OpenTelemetry + Jaeger |
| APM（应用性能管理） | ✗ 不能 | Datadog / New Relic |

---

## 二、代码模式：可复用的模板

### 2.1 PII 脱敏函数（什么时候用：任何要把用户输入进 trace 的函数）

```python
import re

def redact_pii(data):
    """递归脱敏 dict/list/str，去除邮箱、手机、身份证、信用卡、姓名"""
    if isinstance(data, dict):
        return {k: redact_pii(v) for k, v in data.items()}
    if isinstance(data, list):
        return [redact_pii(item) for item in data]
    if isinstance(data, str):
        data = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL]", data)
        data = re.sub(r"1[3-9]\d{9}", "[PHONE]", data)
        data = re.sub(r"\d{17}[\dXx]", "[ID_CARD]", data)
        data = re.sub(r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}", "[CARD]", data)
        data = re.sub(r"我叫[一-龥]{2,4}", "我叫[NAME]", data)
        return data
    return data
```

### 2.2 采样装饰器（什么时候用：函数被高频调用，不想每个调用都 trace）

```python
import random, functools

def sampled_trace(rate: float = 0.1):
    """只对 rate 比例的调用进行 trace"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 注：脚本里简化成"都执行"，生产要做 trace 上下文切换
            return func(*args, **kwargs)
        return wrapper
    return decorator

@mlflow.trace(span_type="AGENT")
@sampled_trace(rate=0.1)
def my_agent(q):
    ...
```

MLflow 3 也提供 OpenTelemetry 风格 sampler：

```python
from mlflow.tracing.sampling import TraceIdRatioBased
```

### 2.3 安全的 Handler（什么时候用：写任何处理用户输入的 trace 函数）

```python
@mlflow.trace(span_type="SUPPORT_AGENT")
def safe_handler(user_message: str, user_id: str) -> str:
    # 在 trace 入口立即脱敏（关键！）
    safe_msg = redact_pii(user_message)
    safe_uid = redact_pii(user_id)
    # 后续 LLM 调用只用 safe_* 版本
    return call_llm(safe_msg, safe_uid)
```

### 2.4 硬件采样片段（什么时候用：想把 CPU/内存画成曲线）

```python
import psutil, mlflow, time

# 第一次必须调用一次，否则第一次返回 0
psutil.cpu_percent(interval=None)

while True:
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    mlflow.log_metric("cpu_percent", cpu, step=int(time.time()))
    mlflow.log_metric("mem_percent", mem.percent, step=int(time.time()))
    time.sleep(1)
```

可采集的指标：
- `psutil.cpu_percent()`：CPU 总体利用率
- `psutil.virtual_memory()`：内存（used / available / percent）
- `psutil.disk_usage('/')`：磁盘用量
- `psutil.disk_io_counters()`：磁盘读写字节数
- `psutil.net_io_counters()`：网络收发字节数

### 2.5 模型打包为 Docker（什么时候用：要把 MLflow Model 部署成独立服务）

```bash
mlflow models build-docker \
  -m models:/my-model@champion \
  -n my-model:v1 \
  --env-manager conda

docker run -p 5001:8080 my-model:v1
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  --data '{"inputs": [{"question": "..."}]}'
```

### 2.6 完整的 docker-compose 模板（什么时候用：要在本地或服务器一键起 MLflow）

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
      POSTGRES_DB: mlflow
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: minio123
    volumes:
      - miniodata:/data
    ports:
      - "9000:9000"   # API
      - "9001:9001"   # Web UI

  createbuckets:
    image: minio/mc
    depends_on:
      - minio
    entrypoint: >
      /bin/sh -c "
      mc alias set local http://minio:9000 minio minio123;
      mc mb -p local/mlflow-artifacts;
      mc anonymous set download local/mlflow-artifacts;
      exit 0;
      "

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    depends_on:
      - postgres
      - minio
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: minio
      AWS_SECRET_ACCESS_KEY: minio123
    command: >
      mlflow server
        --host 0.0.0.0
        --port 5000
        --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow
        --default-artifact-root s3://mlflow-artifacts/
    ports:
      - "5000:5000"

volumes:
  pgdata:
  miniodata:
```

---

## 三、实战步骤：按顺序照做

### 3.1 跑 PII 脱敏 + 采样对比

```bash
# 1. 激活环境
conda activate mlflow

# 2. 设 API key（脚本要调真实 LLM）
export OPENAI_API_KEY=sk-xxx
export OPENAI_API_BASE=https://api.deepseek.com
export DEEPSEEK_MODEL=deepseek-chat

# 3. 跑脚本
python 09_deployment/09a_sampling_redaction.py
```

脚本会跑两遍同一组 5 个客服请求：一次"原始版"（含 PII），一次"脱敏版"。脚本最后会打印每个 trace 的 `trace_inputs` 前 80 字符，并标出是否还残留邮箱/手机/身份证/信用卡。

**在终端会看到**：
- 第 [A] 段：5 条 trace 的输入里都能看到 `@example.com`、`13812345678` 等
- 第 [B] 段：同样的 5 条输入被替换成 `[EMAIL]`、`[PHONE]`、`[ID_CARD]`

### 3.2 启动生产 MLflow（用 docker-compose）

把 `09b_prod_infra.sh` 里那段 YAML heredoc 单独保存为 `docker-compose.yml`，然后：

```bash
docker compose up -d
# 访问 http://localhost:5000
```

**第一次启动会做的事**：
1. 启动 Postgres 并初始化 mlflow 数据库
2. 启动 MinIO 并创建一个叫 `mlflow-artifacts` 的 bucket
3. 启动 MLflow Tracking Server，连上 Postgres 和 MinIO

**客户端连接**：

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

**容器化已有模型**：

```bash
mlflow models build-docker -m models:/my-model@champion -n my-model:v1
docker run -p 5001:8080 my-model:v1
```

### 3.3 跑硬件监控

```bash
conda activate mlflow
python 09_deployment/09c_hardware_monitor.py
```

脚本会后台起采样线程（每秒采一次），同时跑 30 秒"负载剧本"：0-5s 空闲、5-10s CPU 密集、10-15s 分配内存、15-20s CPU 密集、20-25s 分配更多内存、25-30s 空闲。

另开终端看曲线：

```bash
mlflow ui --port 5000
```

选 experiment `09_hardware_monitor` → Run `hardware-monitor-demo` → **Metrics** 标签看曲线。

### 3.4 （选跑）Agent Server 框架（MLflow ≥ 3.6）

```python
from mlflow.genai.agent_server import invoke, stream, AgentServer

agent_server = AgentServer()

@invoke()
async def non_stream_endpoint(request):
    return await my_agent.run(request)

@stream()
async def stream_endpoint(request):
    async for chunk in my_agent.stream(request):
        yield chunk
```

### 3.5 故障排查速查

| 现象 | 原因 | 解决 |
|------|------|------|
| `mlflow.db is locked` | SQLite 并发写 | 换 Postgres |
| MinIO bucket 不存在 | 没跑 `createbuckets` | 手动 `mc mb local/mlflow-artifacts` |
| `mlflow server` 启动后 502 | Postgres 还没起来 | `depends_on` 加 `condition: service_healthy` |
| Trace 只写一半 | 采样没配置 | 检查 `mlflow.tracing.sampling` 设置 |
| `psutil.cpu_percent()` 一直返回 0 | 第一次没设基线 | 在循环外调一次 `psutil.cpu_percent(interval=None)` |

---

## 四、避坑清单

- ⚠️ **后端存了 SQLite** → 生产并发写会锁表。换 Postgres：`--backend-store-uri postgresql://...`
- ⚠️ **构件存在本地磁盘** → 容器重启数据丢失。换 S3/MinIO：`--default-artifact-root s3://...`
- ⚠️ **PII 进了 trace 才发现要洗** → 已经泄漏，永远清不干净。**在 trace 入口函数第一行就脱敏**
- ⚠️ **trace 全量 100% 记录** → 100 QPS 一个月 50TB 存储。立刻降到 10-20%
- ⚠️ **`psutil.cpu_percent()` 第一次返回 0** → 这是 psutil 的"坑"，第一次调用只设基线不返回值。必须先空调一次再进入循环
- ⚠️ **MLflow 没配 `--allowed-hosts`（< 3.5.0）** → 有 DNS rebinding 漏洞，至少升到 3.5 并显式配置
- ⚠️ **用 MLflow 当专业监控** → MLflow 没有阈值告警、没有历史回溯，工业标准是 **Prometheus + Grafana**（或 OpenTelemetry + Datadog）。MLflow 只适合"在 Run 里附带硬件 snapshot"
- ⚠️ **没有 retention policy** → 数据库和 S3 会无限增长。定期跑 `mlflow gc --backend-store-uri $POSTGRES_URI`
- ⚠️ **用明文密码写在 docker-compose** → 生产换 Kubernetes Secret 或 Vault
- ⚠️ **没有 liveness probe** → MLflow server 挂了 K8s 不会重启。要加 `healthcheck`

---

## 五、小结：3-5 个 take-aways

- 生产 MLflow = **Tracking Server + Postgres（后端）+ S3/MinIO（构件）**，缺一不可
- Trace 成本爆炸靠 **采样** 控制，10% 采样省 90% 存储是高 ROI 操作
- **PII 必须在 trace 边界（函数入口）就脱敏**，泄漏之后再洗已经晚了
- MLflow 不是监控工具，要做硬件告警用 **Prometheus + Grafana**，MLflow 只做"实验 Run 内的硬件 snapshot"
- 任何 MLflow Model（含 sklearn / PyTorch / ResponsesAgent）都能用 `mlflow models build-docker` 一键打成 Docker 镜像

---

## 六、生产 Checklist

### 6.1 必须做（上线前 24 小时）

- [ ] Backend store 用 PostgreSQL（不要 SQLite）
- [ ] Artifact store 用 S3/MinIO（不要本地文件系统）
- [ ] MLflow ≥ 3.5 配 `--allowed-hosts` 防 DNS rebinding
- [ ] 启用 HTTPS（nginx 反代）
- [ ] Trace 采样 10-20%
- [ ] PII 脱敏在 trace 边界
- [ ] 定期 `mlflow gc` 清理过期数据
- [ ] Postgres + S3 跨区备份
- [ ] MLflow server liveness probe
- [ ] Registry 权限控制（3.5+ 有 auth）

### 6.2 强烈推荐（上线第一周内）

- [ ] 接入 Prometheus exporter，把 MLflow server 的请求数、错误率、延迟暴露出去
- [ ] Grafana 配 dashboard：API 延迟、错误率、磁盘使用、Postgres 连接数
- [ ] 配置告警规则（Alertmanager）：
  - API 5xx > 1% 持续 5 分钟 → 飞书/Slack
  - Postgres 磁盘 > 80% → 邮件
  - MinIO 磁盘 > 80% → 邮件
- [ ] Trace 采样率写成可配置（环境变量），方便调整
- [ ] 备份策略演练（恢复一次试试）

### 6.3 进阶（成熟期）

- [ ] MLflow → OpenTelemetry → Jaeger/Tempo 做分布式 tracing
- [ ] 模型推理走独立 GPU 节点 + DCGM exporter
- [ ] 影子流量（shadow traffic）回放历史请求验证新模型
- [ ] PII 检测升级到 Presidio（不只是正则）
- [ ] 审计日志：谁、什么时候、改了什么

### 6.4 成本估算参考

下面是一个粗略的月度成本参考（具体取决于云厂商和配置）：

| 组件 | 规格 | 云厂商估价 | 备注 |
|------|------|-----------|------|
| Postgres | db.r6g.large，100GB | ~$200/月 | 存元数据 |
| S3 / MinIO | 1TB 标准存储 | ~$25/月 | 存 trace + 模型 |
| MLflow Server | 2x t3.medium | ~$70/月 | 计算节点 |
| 带宽 | 100GB 出 | ~$10/月 | UI + API 调用 |
| 监控（Prometheus + Grafana） | 托管服务 | ~$50/月 | 注意：Prometheus 比 MLflow 本身更贵是正常的 |
| **合计** | | **~$355/月** | 中小规模 |

**省钱的技巧**：
- Trace 采样 10% 而不是 100%：S3 费用降 90%
- `mlflow gc` 定期清理：避免无限增长
- 选 S3 IA / Glacier 存超过 30 天的模型：降 60% 存储成本

---

## 附录：生产 Checklist（精简版）

- [ ] Backend store: PostgreSQL（不要用 SQLite）
- [ ] Artifact store: S3 / MinIO（不要用本地文件系统）
- [ ] MLflow 3.5+: 配置 `--allowed-hosts` 防 DNS rebinding
- [ ] 配置 HTTPS（nginx 反向代理）
- [ ] 启用 trace 采样（10-20%）
- [ ] 启用 PII 脱敏
- [ ] 设置 retention policy（定期清理过期数据）—— `mlflow gc --backend-store-uri $POSTGRES_URI`
- [ ] 配置 backup（Postgres + S3 cross-region）
- [ ] 监控 MLflow server 自身健康（liveness probe）
- [ ] 注册表权限控制（MLflow 3.5+ 有 auth）
- [ ] 引入 Prometheus + Grafana 做硬件和应用监控（MLflow 不擅长）
