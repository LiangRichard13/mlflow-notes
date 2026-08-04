# 阶段 9 学习笔记：部署到云与生产可观测性

## 一、生产环境三层架构

```
┌──────────────────────────────────────────────────┐
│  Client (Web/Mobile/API)                         │
└────────────────┬─────────────────────────────────┘
                 │
┌────────────────▼─────────────────────────────────┐
│  MLflow Tracking Server (FastAPI)                │
│  --backend-store-uri  → PostgreSQL (元数据)       │
│  --default-artifact-root → S3/MinIO (大文件)      │
│  --allowed-hosts (防 DNS rebinding, ≥3.5.0)     │
└────────────────┬─────────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼────────────┐    ┌───────▼────────────┐
│ Model Service   │    │ Agent Server (3.6+) │
│ (REST /invocations) │ │  (@invoke/@stream)  │
└────────────────┘    └────────────────────┘
```

## 二、最小 docker-compose.yml

```yaml
services:
  postgres:    # 元数据
    image: postgres:15
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
      POSTGRES_DB: mlflow

  minio:       # 构件存储（S3 兼容）
    image: minio/minio
    command: server /data --console-address ":9001"

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    command: >
      mlflow server --host 0.0.0.0 --port 5000
        --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow
        --default-artifact-root s3://mlflow-artifacts/
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: minio
      AWS_SECRET_ACCESS_KEY: minio123
```

详见 `08_project/09b_prod_infra.sh`。

## 三、Trace 采样（控制成本）

生产环境不能 100% 记录 trace（成本爆炸），需要采样：

```python
import random
import functools

def sampled_trace(rate: float = 0.1):
    """只对 rate 比例的调用记录 trace"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if random.random() < rate:
                return func(*args, **kwargs)   # 命中采样
            return func(*args, **kwargs)        # 未命中，不 trace
        return wrapper
    return decorator

@mlflow.trace(span_type="AGENT")
@sampled_trace(rate=0.1)
def my_agent(q): ...
```

**推荐采样率**：
| 场景 | 采样率 |
|------|--------|
| 调试/PoC | 100% |
| 一般生产 | 10-20% |
| 高流量 (>1k QPS) | 1-5% |

MLflow 3 也提供 OpenTelemetry 风格 sampler：
```python
from mlflow.tracing.sampling import TraceIdRatioBased
# 配置 sampler 的细节见 mlflow.tracing.sampling
```

## 四、PII 脱敏（隐私合规）

⚠️ **关键原则：在 trace 边界脱敏，不要泄漏了再洗**

```python
def redact_pii(data):
    """递归脱敏邮箱、手机、身份证、信用卡"""
    if isinstance(data, dict):
        return {k: redact_pii(v) for k, v in data.items()}
    if isinstance(data, list):
        return [redact_pii(item) for item in data]
    if isinstance(data, str):
        import re
        data = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL]", data)
        data = re.sub(r"1[3-9]\d{9}", "[PHONE]", data)
        data = re.sub(r"\d{17}[\dXx]", "[ID_CARD]", data)
        data = re.sub(r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}", "[CARD]", data)
        return data
    return data


@mlflow.trace(span_type="AGENT")
def safe_handler(message, user_id):
    # 在 trace 入口立即脱敏
    safe_msg = redact_pii(message)
    safe_uid = redact_pii(user_id)
    # 后续 LLM 调用只用 safe_* 版本
    return call_llm(safe_msg, safe_uid)
```

## 五、模型打包为 Docker

```bash
# 任何 MLflow 模型（含 ResponsesAgent、sklearn、PyTorch 等）
mlflow models build-docker \
  -m models:/my-model@champion \
  -n my-model:v1 \
  --env-manager conda

docker run -p 5001:8080 my-model:v1
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  --data '{"inputs": [{"question": "..."}]}'
```

## 六、Agent Server（≥3.6.0）

FastAPI 风格的 Agent 托管框架，支持流式：

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

## 七、生产 Checklist

- [ ] Backend store 用 PostgreSQL（不要 SQLite）
- [ ] Artifact store 用 S3/MinIO（不要本地文件系统）
- [ ] MLflow ≥3.5 配 `--allowed-hosts`
- [ ] 启用 HTTPS（nginx 反代）
- [ ] Trace 采样 10-20%
- [ ] PII 脱敏在 trace 边界
- [ ] 定期 `mlflow gc` 清理过期数据
- [ ] Postgres + S3 跨区备份
- [ ] MLflow server liveness probe
- [ ] Registry 权限控制（3.5+ 有 auth）

## 八、成本估算参考

| 流量 | Trace 全量 | Trace 10% | 节省 |
|------|----------|----------|------|
| 100 QPS | 8.6 亿/月 | 8600 万/月 | 90% |
| 1000 QPS | 86 亿/月 | 8.6 亿/月 | 90% |
| 10000 QPS | 860 亿/月 | 86 亿/月 | 90% |

Trace 存储成本主要来自 spans（每次 LLM 调用 ~10 spans）。
每个 trace 平均 ~50KB，1 亿条 ~50TB → 采样是必须的。