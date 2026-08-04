# MLflow 3 关键变化（vs MLflow 2）

> 从 MLflow 2 升级到 MLflow 3 的必读——很多 API 改了名字或语义
> 参考：https://mlflow.org.cn/docs/latest/ml/mlflow-3/

## 一、模型（Models）

### LoggedModel 升级为一等公民

- **MLflow 2**：模型从属于 Run（`runs:/<run_id>/model`）
- **MLflow 3**：模型独立于 Run（`models:/<model_id>`），可以在 Run 之间共享
- 制品存储位置变了：
  - v2：`experiments/<exp>/<run>/artifacts/`
  - v3：`experiments/<exp>/models/<model_id>/artifacts/`

### API 改名

| MLflow 2 | MLflow 3 | 原因 |
|----------|----------|------|
| `mlflow.<flavor>.log_model(..., artifact_path="model")` | `mlflow.<flavor>.log_model(..., name="model")` | `name` 更准确 |
| `mlflow.sklearn.log_model()` 需要 `start_run()` 包裹 | 不再需要 | LoggedModel 独立 |
| `MlflowClient.list_artifacts()` 返回模型目录 | 模型不再在 artifacts 下 | 路径改了 |

### 新增 API

```python
# SQL 风格跨实验搜索模型（新增）
mlflow.search_logged_models(
    experiment_ids=["1"],
    filter_string="metrics.accuracy > 0.9 AND params.lr < 0.01"
)

# 记录外部模型（不在 MLflow 训练，但注册管理）
mlflow.create_external_model(name="my-external-llm")
```

## 二、Run 标识

- `run_info.run_uuid` **移除**，统一使用 `run_id`
- Git 标签 `mlflow.gitBranchName` / `mlflow.gitRepoURL` **移除**（改用 `setup_mlflow_git_based_version_tracking`）
- Run 页面的 Artifacts 选项卡**不再显示模型**，改在独立的 Logged Models 页面管理

## 三、Flavor 变化

| 变化 | 说明 |
|------|------|
| **移除** MLflow Recipes | 改用普通 mlflow API |
| **移除** fastai / mleap / diviner / gluon flavor | 框架过时 |
| **移除** 旧的 deployment server 与 `start-server` CLI | 改用 `mlflow models serve` 或容器化 |
| TensorFlow autolog | 移除 `every_n_iter` 参数 |
| PyTorch flavor | 移除 `requirements_file`，改用 `pip_requirements` / `extra_pip_requirements` |
| 模型 save/log API | 移除 `example_no_conversion` / `code_path` / `inference_config` 等参数 |

## 四、ModelInfo 变化

- `signature_dict` **移除**，统一用 `signature`（`MlflowSignatureDict` 没了）

## 五、评估 API

| MLflow 2 | MLflow 3 |
|----------|----------|
| `mlflow.evaluate(baseline_model=...)` | 用 `mlflow.validate_evaluation_results(new, baseline)` 显式比较 |

## 六、追踪服务器安全（≥3.5.0）

新版本默认开启 DNS rebinding 防护和 CORS 安全中间件，需要配置 `--allowed-hosts`：

```bash
mlflow server --allowed-hosts "*.example.com" --port 5000
```

## 七、GenAI 新增核心能力

### LoggedModel 活跃上下文

```python
mlflow.set_active_model(name="my-agent-v1")  # 后续所有追踪自动关联
```

### ResponsesAgent（替代 ChatAgent）

```python
class MyAgent(mlflow.pyfunc.ResponsesAgent):  # 新基类
    def predict(self, request): ...            # 兼容 OpenAI Responses API
    def predict_stream(self, request): ...     # 流式响应
```

### Agent Server（≥3.6.0）

FastAPI 风格的 Agent 托管：

```python
from mlflow.genai.agent_server import invoke, stream

@invoke()    # 同步/非流
async def endpoint(request): ...

@stream()    # 流式
async def stream_endpoint(request): ...
```

### Prompt Optimization（≥3.5.0）

自动数据驱动的提示词优化：

```python
from mlflow.genai.optimize import GepaPromptOptimizer, MetaPromptOptimizer

result = mlflow.genai.optimize_prompts(
    predict_fn=...,
    train_data=...,
    prompt_uris=["prompts:/qa/1"],
    optimizer=GepaPromptOptimizer(reflection_model="openai:/gpt-4o-mini"),
    scorers=[Correctness()],
)
```

### MCP Server（≥3.5.1）

让 AI 助手通过 MCP 协议操作 MLflow（这个版本可以用 Claude Code 直接调 MLflow！）

### AI Insights（CLEARS 框架）

自动多阶段 AI 流水线分析 trace，检测生产问题。

### Prompt Registry 新能力

- `PromptModelConfig`：把模型参数（temperature/max_tokens）和提示词一起存
- `response_format`：绑定 Pydantic / JSON schema（结构化输出）
- Jinja2 模板：条件、循环、过滤器
- 基于版本无限 TTL / 基于别名 60s TTL 的内存缓存

### 自定义 Judge

```python
from mlflow.genai.judges import make_judge

tone = make_judge(
    name="brand_tone",
    instructions="Score 1-5 based on...",
    model="openai:/gpt-4o-mini",
)
```

### Span 直接存数据库（≥3.3.0）

提升查询性能。

### `mlflow-tracing` 精简包

生产环境可用，只包含追踪功能，体积比 `mlflow` 小 95%。

## 八、迁移清单

如果你从 MLflow 2 项目迁移到 3：

1. **全局替换** `artifact_path=` → `name=`
2. 检查 `mlflow.evaluate(baseline_model=...)` → 改用 `validate_evaluation_results()`
3. 检查 `from mlflow.entities.model_signature import MlflowSignatureDict` → 改用 `ModelSignature`
4. `run_info.run_uuid` → `run.info.run_id`
5. 不再依赖 Run 页面看模型 → 用 Logged Models 页面
6. 部署命令从 `mlflow models serve -m ...` 仍然兼容，但旧的 deployment server / `start-server` 没了
7. （≥3.5）生产 server 加 `--allowed-hosts`
8. （GenAI）旧 `ChatAgent` → 改用 `ResponsesAgent`