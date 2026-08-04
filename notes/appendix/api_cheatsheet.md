# MLflow 3 API 速查表

> 按用途分类，速查用

## 1. Tracking 核心

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")    # 设置追踪 URI
mlflow.set_experiment("exp_name")                   # 切换/创建实验
mlflow.set_experiment_tags({"team": "ml"})          # 给实验打标签

with mlflow.start_run(run_name="trial-1") as run:   # 启动 Run
    mlflow.log_param("lr", 0.01)                     # 记参数
    mlflow.log_params({"batch": 32, "epochs": 10})   # 批量记
    mlflow.log_metric("loss", 0.5, step=epoch)       # 记指标（带 step）
    mlflow.log_metrics({"acc": 0.9, "f1": 0.85})
    mlflow.log_artifact("model.pkl")                 # 记文件
    mlflow.log_artifacts("./outputs", artifact_path="results")
    mlflow.set_tag("stage", "dev")
    mlflow.set_tags({"env": "prod", "owner": "alice"})

# 自动记录（最常用！）
mlflow.sklearn.autolog()
mlflow.pytorch.autolog()
mlflow.langchain.autolog()
mlflow.openai.autolog()
mlflow.anthropic.autolog()

# 搜索
mlflow.search_runs(experiment_names=["exp"], filter_string="metrics.acc > 0.9")
```

## 2. Models 与 Registry

```python
from mlflow.models import infer_signature

# 推断签名（部署必备）
signature = infer_signature(X_train, model.predict(X_train))

# 记录模型（MLflow 3 新写法：name 而非 artifact_path）
mlflow.sklearn.log_model(model, name="model",
                          signature=signature, input_example=X_train.head())

# 加载模型
model = mlflow.sklearn.load_model("runs:/<run_id>/model")
model = mlflow.sklearn.load_model("models:/MyModel@champion")  # 用别名
model = mlflow.sklearn.load_model("models:/MyModel/3")          # 用版本号

# 注册与版本
mlflow.register_model("runs:/<id>/model", "MyModel")
client = mlflow.MlflowClient()
client.set_registered_model_alias("MyModel", "champion", version=2)

# MLflow 3 新增：跨实验搜索 LoggedModel
mlflow.search_logged_models(
    experiment_ids=["1"],
    filter_string="metrics.accuracy > 0.9"
)
```

## 3. 数据集血缘

```python
# 记录数据集
dataset = mlflow.data.from_pandas(df, source="data.csv",
                                   name="train", targets="label")
dataset = mlflow.data.from_spark(spark_df, source="s3://...")
dataset = mlflow.data.from_huggingface(hf_dataset)

with mlflow.start_run():
    mlflow.log_input(dataset, context="training")

# 反查
src = mlflow.data.get_source(dataset)
```

## 4. 评估（传统 ML）

```python
import mlflow

# 模型评估
result = mlflow.models.evaluate(
    model_uri="models:/MyModel@champion",
    data=eval_df,
    targets="label",
    model_type="classifier",   # 或 "regressor"
)

# 自定义指标
from mlflow.metrics import make_metric
import numpy as np

def profit_weighted_acc(preds, targets):
    weights = np.where(targets == 1, 10, 1)   # 正样本权重更高
    return float((preds == targets).dot(weights) / weights.sum())

custom = make_metric(eval_fn=profit_weighted_acc,
                     greater_is_better=True, name="profit_acc")

# 验证新模型 vs baseline
mlflow.validate_evaluation_results(new_result, baseline_result)
```

## 5. 部署

```bash
# 本地服务
mlflow models serve -m models:/MyModel@champion -p 5001

# Docker
mlflow models build-docker -m models:/MyModel@champion -n my-model

# 推理
mlflow models predict -m models:/MyModel@champion -i input.csv
```

```bash
# curl 调用 /invocations
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  --data '{"dataframe_records": [{"f1": 1.0, "f2": 2.0}]}'

# 或 csv 格式
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: text/csv" \
  --data-binary @input.csv
```

## 6. GenAI Tracing

```python
import mlflow
from mlflow.entities.span import SpanType

mlflow.openai.autolog()           # OpenAI 一行开启追踪
mlflow.anthropic.autolog()        # Anthropic 一行开启追踪
mlflow.langchain.autolog()        # LangChain 一行开启追踪

# 自定义追踪
@mlflow.trace(span_type=SpanType.LLM)
def call_llm(prompt: str) -> str:
    return client.messages.create(...).content[0].text

@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve_docs(query: str, k: int = 3):
    return vectorstore.similarity_search(query, k=k)

@mlflow.trace(span_type=SpanType.TOOL)
def my_tool(x: int) -> int:
    return x * 2

# 给当前 trace 加元数据
mlflow.update_current_trace(metadata={
    "mlflow.trace.user": "alice",
    "mlflow.trace.session": "sess-123",
})

# 搜索 trace
traces = mlflow.search_traces(
    filter_string="status=OK",
    max_results=10,
)

# 给 trace 加 tag
mlflow.set_trace_tag(trace_id, "feedback", "upvote")
```

## 7. Prompt Registry

```python
import mlflow
from mlflow.entities.model_config import PromptModelConfig

# 注册提示词
mlflow.genai.register_prompt(
    name="qa-prompt",
    template="You are a helpful assistant. Answer: {{question}}",
    commit_message="v1: initial version",
)

# 用别名管理生命周期
mlflow.genai.set_prompt_alias("qa-prompt", alias="production", version=2)
mlflow.genai.set_prompt_alias("qa-prompt", alias="staging", version=3)

# 加载提示词
prompt = mlflow.genai.load_prompt("prompts:/qa-prompt@production")
response = prompt.format(question="What is MLflow?")

# 绑定模型参数
mlflow.genai.set_prompt_model_config(
    "qa-prompt",
    PromptModelConfig(model_name="gpt-4o-mini", temperature=0.2),
)

# 删除别名
mlflow.genai.delete_prompt_alias("qa-prompt", "staging")
```

## 8. GenAI Evaluation

```python
import mlflow
from mlflow.genai.scorers import Correctness, Safety, RelevanceToQuery
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import scorer

# 内置 scorer
results = mlflow.genai.evaluate(
    data=eval_df,
    predict_fn=lambda row: my_chain.invoke({"q": row["q"]}),
    scorers=[Correctness(), Safety(), RelevanceToQuery()],
)

# 自定义 Python scorer
@scorer
def has_citation(outputs: str) -> bool:
    return "[source:" in outputs

# LLM-as-judge
tone = make_judge(
    name="brand_tone",
    instructions="""Score 1-5 based on whether the response is friendly,
                   professional, and matches our brand voice.""",
    model="openai:/gpt-4o-mini",
)

results = mlflow.genai.evaluate(
    data=eval_df,
    predict_fn=predict,
    scorers=[Correctness(), tone, has_citation],
)
```

## 9. 版本追踪与提示词优化（MLflow 3 核心新特性）

```python
# 设置活跃模型（后续追踪自动关联到这个版本）
mlflow.set_active_model(name="my-agent-v1")

# Git 版本追踪（每个 commit → 一个 LoggedModel）
from mlflow.version_tracking import setup_mlflow_git_based_version_tracking
from mlflow.utils.git_utils import get_git_commit

setup_mlflow_git_based_version_tracking()
commit = get_git_commit(".")[:8]
mlflow.set_active_model(name=f"my-agent-{commit}")

# 提示词自动优化
from mlflow.genai.optimize import GepaPromptOptimizer, MetaPromptOptimizer

result = mlflow.genai.optimize_prompts(
    predict_fn=predict,
    train_data=train_df,
    prompt_uris=["prompts:/qa/1"],
    optimizer=GepaPromptOptimizer(reflection_model="openai:/gpt-4o-mini"),
    scorers=[Correctness()],
)
```

## 10. Agent 与服务

```python
import mlflow
from mlflow.entities.span import SpanType

# 自定义 ResponsesAgent
class MyAgent(mlflow.pyfunc.ResponsesAgent):
    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request):
        # ... agent 逻辑
        return responses

# 打包 Agent
mlflow.pyfunc.log_model(python_model=MyAgent(), name="agent",
                        input_example=example_request)

# LangChain flavor
import mlflow.langchain
mlflow.langchain.autolog()
mlflow.langchain.log_model(chain, name="chain")

# Agent Server (≥3.6.0)
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

## 11. 服务端管理

```bash
# 启动 tracking server
mlflow server \
  --backend-store-uri sqlite:///mydb.sqlite \
  --default-artifact-root ./artifacts \
  --host 0.0.0.0 --port 5000

# 启用注册表（必须跑一次）
mlflow db upgrade sqlite:///mydb.sqlite

# 清理过期数据
mlflow gc --backend-store-uri sqlite:///mydb.sqlite
```

```python
# MLflow 3.5+ 安全配置
# server 端需要配置 --allowed-hosts 防 DNS rebinding
```