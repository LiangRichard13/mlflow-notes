# 阶段 8 学习笔记：版本追踪、提示词优化与 ResponsesAgent

> 对应脚本：`08_agents/08a_active_model.py`、`08b_prompt_optimize.py`、`08c_responses_agent.py`、`simple_qa_agent.py`

## 一、`set_active_model` - 版本化追踪

```python
import mlflow

with mlflow.start_run(run_name="agent-v1-batch"):
    mlflow.set_active_model(name="agent-v1")   # ← 后续 trace 自动关联到 'agent-v1'
    for q in questions:
        agent_v1(q)

with mlflow.start_run(run_name="agent-v2-batch"):
    mlflow.set_active_model(name="agent-v2")   # 关联到 'agent-v2'
    for q in questions:
        agent_v2(q)
```

**实战用法**：每个 git commit → 一个 LoggedModel。

```python
from mlflow.version_tracking import setup_mlflow_git_based_version_tracking
from mlflow.utils.git_utils import get_git_commit

setup_mlflow_git_based_version_tracking()
commit = get_git_commit(".")[:8]
mlflow.set_active_model(name=f"agent-{commit}")
```

## 二、`mlflow.genai.optimize_prompts` - 自动优化提示词

```python
from mlflow.genai.optimize import MetaPromptOptimizer, GepaPromptOptimizer

result = mlflow.genai.optimize_prompts(
    predict_fn=lambda question: my_llm(question),
    train_data=df,                       # {"inputs": ..., "expectations": ...}
    prompt_uris=[f"prompts:/name/{version}"],
    optimizer=MetaPromptOptimizer(
        reflection_model="openai:/deepseek-v4-flash",
    ),
    scorers=[Correctness(model="openai:/deepseek-v4-flash")],
)
```

**两个 optimizer 对比**：
| Optimizer | 依赖 | 速度 | 智能程度 |
|-----------|------|------|---------|
| `MetaPromptOptimizer` | 内置 | 快 | 中 |
| `GepaPromptOptimizer` | 需 `pip install gepa` | 慢 | 高 |

⚠️ 国内服务商兼容性：GEPA 在 DeepSeek 上偶尔报 reflection 调用错误；MetaPrompt 更稳定。

**每个改写版本自动注册**：v1 → v2 → v3... → 最终优化版。

## 三、ResponsesAgent - 自定义 LLM 应用基类

`ResponsesAgent` 是 MLflow 3 的新基类，**兼容 OpenAI Responses API**。

```python
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
)
from mlflow.entities.span import SpanType
import mlflow

class MyAgent(ResponsesAgent):
    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        # 业务逻辑：拼 prompt、调 LLM
        messages = [{"role": m.role, "content": m.content} for m in request.input]
        resp = openai_client.chat.completions.create(...)
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=resp..., id=resp.id)],
            custom_outputs=None,
        )
```

## 四、Models-from-code 打包（MLflow 3 推荐方式）

⚠️ MLflow 3 要求 `python_model` 必须是**可 import 的类**，不能直接传实例。

### 步骤 1：把 Agent 类放到独立 .py 文件
```python
# my_agent.py
from mlflow.models import set_model

class MyAgent(ResponsesAgent):
    def predict(self, request):
        ...

# 必须调用 set_model() 告诉 MLflow 哪个是模型类
set_model(MyAgent())
```

### 步骤 2：log_model 时传文件路径
```python
mlflow.pyfunc.log_model(
    python_model="path/to/my_agent.py",   # ← 字符串！
    name="my-agent",
    pip_requirements=["openai", "mlflow>=3.0"],
)
```

## 五、加载并推理

```python
loaded = mlflow.pyfunc.load_model("models:/m-<model_id>")

# ⚠️ PyFuncModel.predict() 接收 dict-like（不是 ResponsesAgentRequest 对象）
api_request = {
    "input": [{"role": "user", "content": "..."}],
    "temperature": 0.3,
}
result = loaded.predict(api_request)
```

## 六、关键避坑清单

| 坑 | 解决 |
|----|------|
| `Failed to serialize Python model` | 用 Models-from-code：传文件路径字符串 + 在文件里 `set_model(YourClass())` |
| `predict()` 返回空 text | 检查 `request.input` 里 `msg.content` 是字符串还是 list |
| PyFuncModel schema 校验失败 | 传 dict 而不是 ResponsesAgentRequest 对象 |
| 优化器在 DeepSeek 上 reflection 失败 | 用 MetaPromptOptimizer，或手写评估循环替代 |
| set_active_model 没生效 | 必须在 `@mlflow.trace` 装饰的函数里调用，或在 trace 上下文里 |

## 七、MLflow 3 一等公民模型架构

```
LoggedModel（独立实体，不再附属 Run）
├── name: "agent-v2-abc123"
├── model_id: m-xxx
├── model_type: agent / classifier / ...
├── source_run_id: <来源 Run>
├── aliases: [@champion, @challenger]
├── Traces 标签：所有用此版本的 trace
└── Artifacts: 模型文件
```

每个 LoggedModel 可以：
- 关联多个 Run
- 关联多个 Trace
- 注册到 Registry 设别名
- 跨实验搜索（search_logged_models）

## 八、生产部署入口

```bash
# 标准 MLflow 模型 serve（任何模型类型）
mlflow models serve -m models:/my-agent@champion -p 5001

# Agent Server（>=3.6.0，FastAPI 风格）
from mlflow.genai.agent_server import invoke, stream, AgentServer

@invoke()
async def endpoint(request):
    return await my_agent.run(request)

@stream()
async def stream_endpoint(request):
    async for chunk in my_agent.stream(request):
        yield chunk
```