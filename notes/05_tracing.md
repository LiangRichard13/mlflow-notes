# 阶段 5 学习笔记：GenAI 入门与追踪

> 对应脚本：`05_tracing/env_bootstrap.py`、`05a_env_test.py`、`05b_basic_tracing.py`、`05c_custom_decorator.py`、`05d_metadata_search.py`

## 一、一行开启追踪

```python
import mlflow
mlflow.openai.autolog()       # OpenAI 客户端
mlflow.anthropic.autolog()    # Anthropic 客户端
mlflow.langchain.autolog()    # LangChain
```

之后该库的所有调用自动产生 Trace，**无需改业务代码**。

## 二、数据模型：Trace / Span

```
Trace（一次完整请求，可能跨多 Span）
├── Span A（root）
│   ├── Span B（child）
│   └── Span C（child）
└── ...
```

| 概念 | 含义 |
|------|------|
| Trace | 一次完整的调用链，由 trace_id 唯一标识 |
| Span | 单个操作（LLM 调用 / 检索 / 工具 / 自定义函数） |
| SpanType | LLM / RETRIEVER / TOOL / CHAIN / AGENT / AGENT (枚举值或字符串) |
| Attributes | Span 的元数据（model、temperature、k、...） |
| Events | Span 内的事件流 |
| trace_metadata | 整个 Trace 级别的元数据（cost、tokenUsage、user、session） |
| tags | Trace 级别可变标签（feedback、status、...） |

## 三、自定义 Span

```python
from mlflow.entities.span import SpanType
import mlflow

@mlflow.trace(span_type=SpanType.RETRIEVER, name="retrieve_docs")
def my_retriever(query):
    return [...]

@mlflow.trace(span_type="CUSTOM_TYPE", name="my_step")
def my_step(...):
    ...

# 函数嵌套 → Span 嵌套
@mlflow.trace(span_type=SpanType.CHAIN)
def chain(q):
    docs = my_retriever(q)         # 子 Span
    return my_step(docs)            # 子 Span
```

## 四、Trace 元数据（关键：专用参数）

```python
mlflow.update_current_trace(
    user=user_id,                 # → metadata.mlflow.trace.user
    session_id=sess_id,           # → metadata.mlflow.trace.session
    tags={"feedback": "upvote"},
    metadata={"git_commit": "abc123"},   # 不可变
)
```

⚠️ **不要用** `metadata={"mlflow.trace.user": ...}` 这种 key——它们是保留字段，必须用专用参数 `user=` / `session_id=`。

⚠️ `update_current_trace` 必须在**已激活的 trace 上下文**里调用：
- 函数被 `@mlflow.trace` 装饰
- 或在 `with mlflow.start_span(...)` 内

## 五、搜索 Trace

```python
# 基本搜索
traces = mlflow.search_traces(
    experiment_ids=["1"],                     # ⚠️ 即将废弃，用 locations
    filter_string="status = 'OK'",
    max_results=100,
)

# 按用户/会话过滤
traces = mlflow.search_traces(
    filter_string="metadata.`mlflow.trace.user` = 'alice'",
)

# 排序（⚠️ 字符串 list，不是 list[dict]）
traces = mlflow.search_traces(
    order_by=["execution_time_ms DESC"],      # 慢的在前
    max_results=5,
)
```

**关键约束**：
- `experiment_ids` → 即将废弃，用 `locations=[exp_id]`
- `order_by` → `list[str]`，字段名是 `execution_time_ms`（不是 `execution_duration`）
- 返回 DataFrame，列：`trace_id, trace_metadata, tags, execution_duration, request_time, state, ...`

## 六、Trace 反馈闭环

```python
mlflow.set_trace_tag(trace_id, "user_feedback", "upvote")
mlflow.set_trace_tag(trace_id, "rating", "5")

# 收集后的 trace 可以导入 eval 数据集
# 用于离线评估 / 训练奖励模型 / few-shot 示例
```

## 七、国内 LLM 桥接模式

`env_bootstrap.py` 自动做的事：

```
DEEPSEEK_API_KEY  ─┐
ZHIPU_API_KEY     ─┤
DASHSCOPE_API_KEY ─┼──→  OPENAI_API_KEY
MOONSHOT_API_KEY  ─┤     OPENAI_API_BASE
YI_API_KEY        ─┘
```

然后 `mlflow.openai.autolog()` 直接工作（DeepSeek/智谱/百炼/Moonshot 都兼容 OpenAI 协议）。

## 八、实战技巧

1. **trace 与 run 的关系**：autolog 自动把当前 run 作为 trace 的关联（用 `metadata.mlflow.sourceRun`）
2. **多轮对话**：用 `with mlflow.start_run()` 包起来，多次 LLM 调用都归到这个 run，每个调用是一个独立 trace
3. **成本追踪**：autolog 自动算 `metadata.mlflow.trace.cost`（基于 token 数 × 单价）
4. **session 聚合**：用 `session_id` 参数，同一会话的多次调用就能搜出来
5. **debug 多步 Agent**：所有中间步骤都被记录，能精确看到哪步出错