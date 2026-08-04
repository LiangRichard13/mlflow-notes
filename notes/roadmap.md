# MLflow 3 完整学习路线图（机器学习 + LLM/Agent）

> 基于 mlflow.org.cn 官方文档站（`/ml/` + `/genai/`）系统性调研产出
> 调研时间：2026-08-04 · MLflow 版本：3.15.1

## 整体设计：螺旋式学习

```
Phase 1 ─→ 2 ─→ 3 ─→ 4       ── 传统 ML 基础（无需 API key）
            │
            └─→ 5 ─→ 6 ─→ 7 ─→ 8 ─→ 9   ── LLM/Agent 进阶
                    │                   （需要 OpenAI/Anthropic key）
                    └─────────────────────┘
                    （用同一套 MLflow 概念）
```

**理念**：先用 scikit-learn 跑通 MLflow 核心概念（实验、模型、注册表、评估、部署），同样的概念在 Phase 5+ 直接套用到 LLM/Agent。每个 ML 概念都会在 GenAI 阶段复现一次，加深印象。

---

## 阶段总览（9 个阶段）

| # | 中文 | 英文 | 时长 | API Key | 目录 |
|---|------|------|------|---------|------|
| 1 | 入门与追踪基础 | Onboarding & Tracking Fundamentals | 2-3h | ❌ | `01_basics/` |
| 2 | 模型格式与注册表 | Model Format & Registry | 2-3h | ❌ | `02_tracking/03_registry/` |
| 3 | 追踪服务器与数据集血缘 | Tracking Server & Dataset Lineage | 2-3h | ❌ | `02_tracking/` |
| 4 | 评估、服务与经典 ML 验证 | Evaluation, Serving & Classic ML Validation | 2-3h | ❌ | `02_tracking/04_serving/` |
| 5 | GenAI 入门与追踪 | GenAI Onboarding & Tracing | 3-4h | ✅ | `04_tracing/` |
| 6 | 提示词注册表与框架 Flavors | Prompt Registry & Framework Flavors | 3h | ✅ | `05_prompts/` |
| 7 | GenAI 评估与自定义评分器 | GenAI Evaluation & Custom Scorers | 3h | ✅ | `06_evaluation/` |
| 8 | 版本追踪、提示词优化与 ResponsesAgent | Version Tracking, Prompt Optimization & ResponsesAgent | 4h | ✅ | `07_agents/` |
| 9 | 部署到云与生产可观测性 | Cloud Deployment & Production Observability | 3h | ✅ | `08_project/` |

---

## Phase 1：入门与追踪基础（2-3h）

**目标**：理解 MLflow 四大支柱，跑通第一个实验

**关键能力**：
- 启动 Tracking Server
- 用 `mlflow.sklearn.autolog()` 一键记录参数/指标/模型/signature
- 区分手动记录 vs 自动记录
- 在 UI 中对比 Run

**阅读文档**：
- https://mlflow.org.cn/docs/latest/ml/getting-started/
- https://mlflow.org.cn/docs/latest/ml/tracking/quickstart/
- https://mlflow.org.cn/docs/latest/ml/tracking/tracking-api/
- https://mlflow.org.cn/docs/latest/ml/tracking/autolog/
- https://mlflow.org.cn/docs/latest/genai/

**必学代码模式**：
```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLflow Quickstart")
mlflow.sklearn.autolog()                       # ← 一行开启自动记录
with mlflow.start_run() as run:
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
# 启动 server: mlflow server --port 5000
```

**动手任务**：用 sklearn 在 Iris 数据集上训练两个 LogisticRegression 模型（不同 C 值），用 `mlflow.sklearn.autolog()` 记录，启 UI 对比 accuracy。

**当前进度**：✅ 已完成（见 `01_basics/01_hello_mlflow.py` 和 `01b_sklearn_basics.py`）

---

## Phase 2：模型格式与注册表（2-3h）

**目标**：理解 MLflow Model 格式，掌握注册表与别名

**关键能力**：
- 理解 MLmodel YAML + flavors 的格式
- 用 `infer_signature()` 记录模型签名
- 注册模型 + 版本管理
- **用别名（champion/challenger）替代已废弃的 stage**
- 用 `models:/name@alias` 加载模型

**阅读文档**：
- https://mlflow.org.cn/docs/latest/ml/model/
- https://mlflow.org.cn/docs/latest/ml/traditional-ml/sklearn/
- https://mlflow.org.cn/docs/latest/ml/model-registry/
- https://mlflow.org.cn/docs/latest/ml/model-registry/workflow/
- https://mlflow.org.cn/docs/latest/ml/model-registry/stages-aliases/

**必学代码模式**：
```python
from mlflow.models import infer_signature
signature = infer_signature(X_train, model.predict(X_train))
mlflow.sklearn.log_model(model, name="model", signature=signature, input_example=X_train.head())
mlflow.register_model(f"runs:/{run.info.run_id}/model", "WineRF")
client.set_registered_model_alias("WineRF", "champion", version=1)
model = mlflow.sklearn.load_model("models:/WineRF@champion")
```

**动手任务**：在 Wine 数据集上构建 sklearn Pipeline，记录签名、注册、设置 champion 别名、写加载脚本。

---

## Phase 3：追踪服务器、构件存储与数据集血缘（2-3h）

**目标**：掌握生产级 Tracking Server 配置与数据集血缘

**关键能力**：
- 配置 sqlite/postgres backend store
- 配置 S3/MinIO artifact store
- 用 `mlflow.data.from_pandas()` 追踪数据集血缘
- **MLflow 3 新增**：`mlflow.search_logged_models()` SQL 风格搜索
- 跑 `mlflow db upgrade` 启用注册表
- 用 `mlflow gc` 清理

**阅读文档**：
- https://mlflow.org.cn/docs/latest/ml/tracking/server/
- https://mlflow.org.cn/docs/latest/ml/tracking/backend-stores/
- https://mlflow.org.cn/docs/latest/ml/tracking/artifact-stores/
- https://mlflow.org.cn/docs/latest/ml/dataset/
- https://mlflow.org.cn/docs/latest/ml/mlflow-3/

**必学代码模式**：
```bash
# 启动生产级 server
mlflow server \
  --backend-store-uri sqlite:///mydb.sqlite \
  --default-artifact-root ./artifacts \
  --host 0.0.0.0 --port 5000
mlflow db upgrade sqlite:///mydb.sqlite
```
```python
dataset = mlflow.data.from_pandas(df, source="wine.csv", name="wine", targets="quality")
with mlflow.start_run():
    mlflow.log_input(dataset, context="training")

# MLflow 3 新增的跨实验搜索
mlflow.search_logged_models(experiment_ids=["1"],
    filter_string="metrics.accuracy > 0.9")
```

**动手任务**：用 sqlite 启动 server，记录一个 sklearn 实验并追踪数据集，然后用 `search_logged_models` 找出 accuracy > 0.95 的模型。

---

## Phase 4：评估、服务与经典 ML 验证（2-3h）

**目标**：评估模型 + 本地部署

**关键能力**：
- 用 `mlflow.models.evaluate()` 计算内置指标 + 可视化
- 用 `make_metric()` 写自定义指标
- **MLflow 3 改动**：用 `mlflow.validate_evaluation_results()` 替代旧的 baseline_model
- `mlflow models serve` 启动本地服务
- 用 curl 调 `/invocations`

**阅读文档**：
- https://mlflow.org.cn/docs/latest/ml/evaluation/
- https://mlflow.org.cn/docs/latest/ml/evaluation/validation/
- https://mlflow.org.cn/docs/latest/ml/deployment/deploy-model-locally/

**必学代码模式**：
```python
result = mlflow.models.evaluate(
    "models:/WineRF@champion", eval_df,
    targets="label", model_type="classifier"
)
mlflow.validate_evaluation_results(new_result, baseline_result)
```
```bash
mlflow models serve -m models:/WineRF@champion -p 5001
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  --data '{"dataframe_records": [...]}'
```

**动手任务**：评估两个 sklearn 模型、写一个自定义业务指标（如"利润加权 accuracy"）、验证新模型更优、用 `models serve` 部署冠军模型并 curl 测 3 个样本。

---

## Phase 5：GenAI 入门与追踪（3-4h）

**目标**：进入 LLM 世界，用追踪观察 LLM 行为

**关键能力**：
- `mlflow.openai.autolog()` / `mlflow.anthropic.autolog()` 一行开启追踪
- 用 `@mlflow.trace` 装饰自定义函数
- 理解 Trace/Span 数据模型（嵌套的调用树）
- 用 `mlflow.update_current_trace()` 附加 user/session 元数据
- 用 `mlflow.search_traces()` 搜索

**阅读文档**：
- https://mlflow.org.cn/docs/latest/genai/getting-started/connect-environment/
- https://mlflow.org.cn/docs/latest/genai/tracing/
- https://mlflow.org.cn/docs/latest/genai/tracing/quickstart/

**必学代码模式**：
```python
import mlflow
mlflow.openai.autolog()    # ← 一行开启 OpenAI 追踪
mlflow.anthropic.autolog()

@mlflow.trace(span_type=SpanType.LLM)
def summarize(text: str) -> str:
    return client.messages.create(...).content[0].text

mlflow.update_current_trace(metadata={
    "mlflow.trace.user": uid,
    "mlflow.trace.session": sid,
})
traces = mlflow.search_traces(filter_string="status=OK", max_results=10)
```

**动手任务**：写一个小 CLI Q&A bot，调 OpenAI/Anthropic，开启 autolog，问 5 个问题，在 UI 看 trace 瀑布图，加 user/session 元数据把 5 次调用归到同一 session。

---

## Phase 6：提示词注册表与框架 Flavors（3h）

**目标**：管理提示词生命周期，集成 LangChain

**关键能力**：
- `mlflow.genai.register_prompt()` 注册版本化提示词
- 用 Jinja2 模板
- 用别名（production/staging/latest）管理生命周期
- `PromptModelConfig` 绑定模型参数（temperature/max_tokens）
- `response_format` 绑定 Pydantic/JSON schema（结构化输出）
- `mlflow.langchain.autolog()` 追踪 LangChain
- `prompts:/name@alias` 在 app 中加载提示词

**阅读文档**：
- https://mlflow.org.cn/docs/latest/genai/prompt-registry/
- https://mlflow.org.cn/docs/latest/genai/prompt-registry/manage-prompt-lifecycles-with-aliases/
- https://mlflow.org.cn/docs/latest/genai/prompt-registry/use-prompts-in-apps/
- https://mlflow.org.cn/docs/latest/genai/flavors/
- https://mlflow.org.cn/docs/latest/genai/flavors/langchain/

**必学代码模式**：
```python
mlflow.genai.register_prompt(
    name="qa-prompt",
    template="Answer: {{question}}",
    commit_message="v1",
)
mlflow.genai.set_prompt_alias("qa-prompt", alias="production", version=2)
mlflow.genai.set_prompt_model_config(
    "qa-prompt", PromptModelConfig(model_name="gpt-4o-mini", temperature=0.2)
)
prompt = mlflow.genai.load_prompt("prompts:/qa-prompt@production")
formatted = prompt.format(question=q)
```

**动手任务**：注册两个版本的客服提示词（简洁版 vs 详细版），分别打 `@production`/`@staging`，写 LangChain RAG chain，启动时从 `prompts:/support@production` 加载。

---

## Phase 7：GenAI 评估与自定义评分器（3h）

**目标**：系统化评估 LLM 输出

**关键能力**：
- `mlflow.genai.evaluate()` 用内置 judges（Correctness/Guidelines/Safety/Relevance）
- 用 `@scorer` 装饰器写自定义评分器
- `make_judge()` 构建 LLM-as-judge（自定义 rubric）
- 跨提示词版本对比评估结果
- 把生产 trace 收集成 eval 数据集（在线→评估闭环）

**阅读文档**：
- https://mlflow.org.cn/docs/latest/genai/eval-monitor/
- https://mlflow.org.cn/docs/latest/genai/eval-monitor/quickstart/
- https://mlflow.org.cn/docs/latest/genai/prompt-registry/evaluate-prompts/

**必学代码模式**：
```python
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import Correctness, Safety

tone_judge = make_judge(
    name="brand_tone",
    instructions="Score 1-5 based on whether the response matches our friendly tone...",
    model="openai:/gpt-4o-mini",
)

@scorer
def has_citation(outputs: str) -> bool:
    return "[source:" in outputs

results = mlflow.genai.evaluate(
    data=eval_df,
    predict_fn=lambda row: chain.invoke({"q": row["q"]}),
    scorers=[Correctness(), Safety(), tone_judge, has_citation],
)
```

**动手任务**：构建 30 行金色 eval 集，写 3 个 scorer（Correctness + 自定义 has_citation + LLM-judge 评估 tone），评估 `@production` 和 `@staging` 两个提示词，在 UI 选赢家。

---

## Phase 8：版本追踪、提示词优化与 ResponsesAgent（4h）

**目标**：MLflow 3 新核心能力——把整个 LLM 应用当一等公民管理

**关键能力**：
- `mlflow.set_active_model()` 关联后续追踪到具体版本
- `setup_mlflow_git_based_version_tracking()` 每个 git commit 一个 LoggedModel
- `mlflow.genai.optimize_prompts()` 用 GepaPromptOptimizer/MetaPromptOptimizer 自动优化
- 自定义 `ResponsesAgent` 子类 + `@mlflow.trace`
- `mlflow.pyfunc.log_model()` 包装 Agent
- 在 quality/cost/latency 上对比 Agent 版本

**阅读文档**：
- https://mlflow.org.cn/docs/latest/genai/version-tracking/
- https://mlflow.org.cn/docs/latest/genai/prompt-registry/optimize-prompts/
- https://mlflow.org.cn/docs/latest/genai/flavors/responses-agent-intro/
- https://mlflow.org.cn/docs/latest/genai/flavors/custom-pyfunc-for-llms/

**必学代码模式**：
```python
from mlflow.genai.optimize import GepaPromptOptimizer

# Git 版本追踪
setup_mlflow_git_based_version_tracking()
v = get_git_commit(".")[:8]
mlflow.set_active_model(name=f"agent-{v}")

# 自动优化提示词
result = mlflow.genai.optimize_prompts(
    predict_fn=my_predict,
    train_data=train_df,
    prompt_uris=["prompts:/qa/1"],
    optimizer=GepaPromptOptimizer(reflection_model="openai:/gpt-4o-mini"),
    scorers=[Correctness()],
)

# 自定义 ResponsesAgent
class MyAgent(mlflow.pyfunc.ResponsesAgent):
    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request):
        ...

mlflow.pyfunc.log_model(python_model=MyAgent(), name="agent", input_example=example)
```

**动手任务**：把项目初始化为 git 仓库 → 开启 git 版本追踪 → 用 GepaPromptOptimizer 优化 QA 提示词 → 写一个带 tool-call 的 ResponsesAgent → 用 pyfunc 打包 → `models serve` 本地测试 → 在 UI 看不同 LoggedModel 版本的 trace。

---

## Phase 9：部署到云与生产可观测性（3h）

**目标**：把模型/Agent 部署到生产并监控

**关键能力**：
- `mlflow models build-docker` 打 Docker 镜像
- 配生产 Tracking Server（Postgres + MinIO）
- Trace 采样 + PII redact
- AI Insights 自动问题检测（CLEARS 框架）
- **Agent Server（≥3.6.0）**：FastAPI 风格 Agent 托管，`@invoke` / `@stream` 装饰器

**阅读文档**：
- https://mlflow.org.cn/docs/latest/ml/deployment/deploy-model-locally/
- https://mlflow.org.cn/docs/latest/ml/deployment/deploy-model-to-sagemaker/
- https://mlflow.org.cn/docs/latest/genai/serving/agent-server/
- https://mlflow.org.cn/docs/latest/genai/eval-monitor/ai-insights/detect-issues/

**必学代码模式**：
```bash
mlflow models build-docker -m models:/WineRF@champion -n wine-classifier
docker compose up -d  # MLflow + Postgres + MinIO
```
```python
from mlflow.genai.agent_server import invoke, stream, AgentServer

@invoke()
async def non_stream_endpoint(request):
    return await my_agent.run(request)

@stream()
async def stream_endpoint(request):
    async for chunk in my_agent.stream(request):
        yield chunk

# Trace 采样与 PII redact
mlflow.tracing.set_span_processor(SamplingSpanProcessor(rate=0.1))
PiiRedactionSpanProcessor(...)
```

**动手任务**：用 `build-docker` 容器化 sklearn 冠军 → docker-compose 起生产 Tracking Server → 流式 100 条 trace（10% 采样 + PII 脱敏） → 触发 AI Insights 自动检测问题 → 用 Agent Server 部署 Agent。

---

## 核心交叉对照：ML 概念 ↔ GenAI 应用

| ML 概念 | GenAI 应用 | 示例 |
|---------|-----------|------|
| **Experiment & Run** | 同样用于 LLM 调用序列（Trace → Span 嵌套） | sklearn 训练 run 包含 params/metrics/model；OpenAI Agent run 包含 chat span + tool call span + 子 LLM span |
| **Model Registry + Aliases** | Prompt Registry 用同样的别名模式 | `set_registered_model_alias('m','champion',2)` ≡ `set_prompt_alias('qa','production',2)` |
| **`mlflow.models.evaluate`** | **`mlflow.genai.evaluate`**（换 scorer 为 LLM judge） | `model_type='classifier'` 算 accuracy/F1 ↔ `scorers=[Correctness(),Safety()]` 算质量分 |
| **`mlflow models serve`** | 同样可服务 ResponsesAgent LoggedModel | `mlflow models serve -m models:/MyAgent@champion` 对两类模型都生效 |
| **数据集血缘 `mlflow.data`** | 同样记录 eval 集，让 prompt 版本对比可追溯 | 同一 eval CSV 给多个 prompt 版本评估 |
| **`search_logged_models`**（MLflow 3 新）| 同样适用于 LLM LoggedModel（prompt+code+config）| 跨团队模型发现统一接口 |

---

## 毕业项目：SupportPilot

**一个生产级 GenAI 客服 Copilot，结合传统 ML 安全网**

需求描述：
- LangChain RAG Agent（GenAI）回答客户问题
- sklearn 意图分类器（传统 ML）过滤越界请求
- 每个 git commit → LoggedModel（通过 `setup_mlflow_git_based_version_tracking`）
- 提示词在 Prompt Registry 中版本化，`@production` / `@staging` 别名管理
- 100 行金色 eval 集跑 `mlflow.genai.evaluate`，3 个 scorer：Correctness + 自定义 has_citation + LLM-judge 评 brand tone
- `mlflow.validate_evaluation_results` 对比每个新版本与 baseline（quality + cost + latency）
- 冠军用 `mlflow models serve` 或 Agent Server（≥3.6.0）部署
- 生产 trace 采样 + PII 脱敏 + AI Insights 自动问题检测

**涉及全部 MLflow 3 能力**：
- Tracking Server (sqlite/postgres + MinIO)
- `mlflow.sklearn.autolog` + `mlflow.langchain.autolog`
- Model Registry + 别名 (champion/challenger)
- Prompt Registry + response_format + PromptModelConfig
- `mlflow.data.from_pandas` 记录 eval 集血缘
- `mlflow.search_logged_models` 跨版本发现
- `mlflow.models.evaluate` (sklearn) + `mlflow.genai.evaluate` (LLM)
- `@scorer` 和 `make_judge` 自定义 scorer
- `mlflow.validate_evaluation_results`
- LoggedModel 版本追踪（set_active_model + git 集成）
- `mlflow.genai.optimize_prompts` (GepaPromptOptimizer)
- ResponsesAgent 子类 + `mlflow.pyfunc.log_model`
- 本地服务 `mlflow models serve`
- Trace 采样 + PII redact + AI Insights

**预计时长**：1-2 周（part-time）

---

## 关键文档链接速查

**机器学习文档**（https://mlflow.org.cn/docs/latest/ml/）：
- 入门：https://mlflow.org.cn/docs/latest/ml/getting-started/
- Tracking：https://mlflow.org.cn/docs/latest/ml/tracking/quickstart/
- Autologging：https://mlflow.org.cn/docs/latest/ml/tracking/autolog/
- Models：https://mlflow.org.cn/docs/latest/ml/model/
- Registry：https://mlflow.org.cn/docs/latest/ml/model-registry/
- Deployment：https://mlflow.org.cn/docs/latest/ml/deployment/
- Evaluation：https://mlflow.org.cn/docs/latest/ml/evaluation/
- scikit-learn 集成：https://mlflow.org.cn/docs/latest/ml/traditional-ml/sklearn/
- Tracking Server：https://mlflow.org.cn/docs/latest/ml/tracking/server/
- 数据集：https://mlflow.org.cn/docs/latest/ml/dataset/
- MLflow 3 迁移：https://mlflow.org.cn/docs/latest/ml/mlflow-3/

**GenAI 文档**（https://mlflow.org.cn/docs/latest/genai/）：
- GenAI 入门：https://mlflow.org.cn/docs/latest/genai/
- Tracing：https://mlflow.org.cn/docs/latest/genai/tracing/
- Prompt Registry：https://mlflow.org.cn/docs/latest/genai/prompt-registry/
- Prompt Optimization：https://mlflow.org.cn/docs/latest/genai/prompt-registry/optimize-prompts/
- GenAI Evaluation：https://mlflow.org.cn/docs/latest/genai/eval-monitor/
- 版本追踪：https://mlflow.org.cn/docs/latest/genai/version-tracking/
- Flavors：https://mlflow.org.cn/docs/latest/genai/flavors/
- Agent Server：https://mlflow.org.cn/docs/latest/genai/serving/agent-server/