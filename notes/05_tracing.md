# 阶段 5 学习笔记：GenAI 入门与追踪

> 对应脚本：`05_tracing/env_bootstrap.py`、`05a_env_test.py`、`05b_basic_tracing.py`、`05c_custom_decorator.py`、`05d_metadata_search.py`

## 🎯 这篇笔记做什么

在前面几个阶段里你可能一直在训传统 ML 模型（scikit-learn 那种）。从 Phase 5 开始我们进入 GenAI / LLM 应用的世界。LLM 调用跟传统模型最大的不同在于：**一次"推理"往往是一连串步骤**——查向量库、组装 prompt、调大模型、解析输出、可能还调工具。出了 bug 你根本不知道是哪一步慢、哪一步错。

MLflow 的 **Tracing（追踪）** 就是为了解决这个问题。它能把你代码里**每一次调用**（不管是 LLM、检索、还是你自己写的函数）都自动记下来，形成一棵"调用树"（Span 树）。出问题的时候点开 UI 就能看到底是哪一步出了错、花了多少时间、花了多少 token。

> 类比：传统 ML 像做一道菜（洗菜 → 切菜 → 炒 → 装盘），spans 就是每个步骤的"工序记录"，trace 就是这一整份工序单。LLM 应用比做菜复杂得多（可能要十几步、还可能循环/重试），所以这套"工序记录"特别重要。

### 你会学到什么

- 能用 `mlflow.openai.autolog()` 一行开启追踪，所有 LLM 调用自动留痕
- 能用 `@mlflow.trace` 装饰自己的函数，把任何 Python 步骤也纳入追踪
- 能给 trace 打 user / session / 业务元数据，做到"按用户查历史""按会话聚合"
- 能用 `mlflow.search_traces()` 像查数据库一样程序化搜索历史 trace

### 对应脚本清单

| 脚本 | 一句话作用 | 是否必跑 | 前置 |
|------|-----------|---------|------|
| `env_bootstrap.py` | 自动把国内 LLM（DeepSeek 等）的 key 桥接成 OpenAI 协议 | 必跑（其他脚本都依赖它） | 无 |
| `05a_env_test.py` | 验证 MLflow 能联通 DeepSeek，发一次最简单的请求 | 必跑 | 跑过 env_bootstrap |
| `05b_basic_tracing.py` | `mlflow.openai.autolog()` 实战 + 多轮对话追踪 | 必跑 | 跑过 05a |
| `05c_custom_decorator.py` | `@mlflow.trace` 自定义 Span，搭一个 RAG 链看嵌套 Span 树 | 推荐 | 跑过 05b |
| `05d_metadata_search.py` | 给 trace 打 user/session + `search_traces` 查询实战 | 推荐 | 跑过 05b |

### 前置知识

- **环境**：MLflow 已装好（`conda activate mlflow`），`.env` 里写好 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY`
- **包**：`pip install mlflow openai python-dotenv`
- **API key**：需要 DeepSeek / OpenAI 中任意一个 key（推荐 DeepSeek，国内直连）
- **前置笔记**：Phase 1（必须，至少会 `mlflow ui` 和 `mlflow.set_experiment`）
- **懂什么**：会用 Python 装饰器最基础的概念（`@something` 加在函数上）
- **不懂什么**：不用了解 OpenTelemetry、OpenInference 这些行业追踪标准——MLflow 都包好了

### 跑完必看（UI）

1. 启动 UI：`mlflow ui --port 5000`
2. 选 experiment **`05_custom_tracing`**（最丰富的脚本是 05c，能看到嵌套 Span 树）
3. 点开 Run **`rag-q1`**（05c 的第一个 Run）
4. 看 **Traces 标签**：
   - 应该看到 **1 个 Trace**，点开后是 5 层嵌套 Span：
     - `rag_chain` (CHAIN) → `retrieve_docs` (RETRIEVER) → `rerank` (TOOL) → `build_prompt` → `generate_answer` (LLM) → 最底层 `chat completions` 自动 span
   - 每个 Span 点开看 **Inputs / Outputs / Latency / Attributes**（model、temperature 等参数）
5. 切到 **`05_metadata`** experiment → Run `multi-user-sim` → Traces 标签：
   - 看到 15 条 trace（3 用户 × 5 轮）
   - 用左侧过滤框输入 `metadata.mlflow.trace.user = 'alice'` 试试——只剩 alice 的 trace

## 一、核心概念：用人话讲清楚

### 1. Trace（一次完整的调用链）

Trace 就是"用户从发起请求到拿到结果"这一次完整的过程。对聊天应用来说，一次"对话"或一次"问答"就是一个 Trace。Trace 有一个独一无二的 `trace_id`，UI 上每条 Trace 都对应一行。

### 2. Span（一个具体操作）

一个 Trace 内部会被拆成若干个 Span。每个 Span 代表**一个具体做的事**，比如：调一次 LLM、查一次向量库、跑一次重排函数。

Span 是嵌套的——比如 `rag_chain` 这个父 Span 里，调了 `retrieve_docs`、`rerank`、`generate_answer`，那后三者就是它的子 Span。**最外层那个 Span 叫 root span**。

### 3. SpanType（Span 的种类）

Span 有个"类型"标签，告诉 MLflow 这个 Span 在干什么。常用值：

| SpanType | 含义 | 用在哪儿 |
|----------|------|----------|
| `LLM` | 大模型调用 | 包装 LLM 函数 |
| `RETRIEVER` | 检索 | 包装向量查询 |
| `TOOL` | 工具调用 | 包装外部 API / 函数 |
| `CHAIN` | 编排链 | 包装整个流程 |
| `AGENT` | Agent 决策 | 包装 Agent 主循环 |
| `CUSTOM_TYPE` 或任意字符串 | 自定义 | 你自己随便起名 |

```python
from mlflow.entities.span import SpanType

@mlflow.trace(span_type=SpanType.RETRIEVER, name="retrieve_docs")
def my_retriever(query):
    ...
```

UI 上不同类型会显示不同图标，**类型不影响功能**，主要方便你一眼看出 Span 在干什么。

### 4. Attributes / Metadata / Tags 三兄弟

这三个都是 Span 或 Trace 上挂的"小标签"，新手容易混，记住下面就行：

| 字段 | 挂在哪 | 能不能改 | 用途 |
|------|--------|---------|------|
| `attributes` | 单个 Span | 只读 | Span 自己的输入输出参数（model 名、温度、检索 k 等，**自动填充**） |
| `metadata` | 整个 Trace | **不可变** | 业务关键标识（user_id、session_id、git_commit 等） |
| `tags` | 整个 Trace | **可变** | 可变的状态标记（feedback、status 之类） |

记法：**metadata 是一次性写死的，tags 是后来可以补的**。给 trace 加 user 必须用 `update_current_trace(user=...)`，自动落到 `metadata.mlflow.trace.user`。

### 5. autolog：零侵入的追踪

`mlflow.openai.autolog()`（或 `mlflow.anthropic.autolog()`、`mlflow.langchain.autolog()`）调用一次后，**所有对应库的调用都会自动记录**——你**不需要改业务代码**，也不用 `@mlflow.trace`。这是最快的上手方式。

## 二、代码模式：可复用的模板

### 模式 1：一行开启追踪（OpenAI / 国内 LLM 都通用）

```python
import mlflow
mlflow.openai.autolog()   # OpenAI 客户端
# 之后 client.chat.completions.create() 每次都会被追踪
```

什么时候用：调国内 LLM（DeepSeek/智谱/百炼）也走 OpenAI 协议，只要已经通过 `env_bootstrap` 桥接好，就可以这么用。

### 模式 2：装饰自己的函数

```python
from mlflow.entities.span import SpanType

@mlflow.trace(span_type=SpanType.CHAIN, name="rag_chain")
def rag_chain(question):
    docs = retrieve(question)   # 子 Span
    return llm(docs)            # 子 Span
```

什么时候用：你想追踪自己写的业务函数（不是 LLM SDK），比如数据预处理、检索、重排、格式转换等。

### 模式 3：给 Trace 打 metadata（user / session）

```python
@mlflow.trace(span_type="CHAT")
def chat(user_id, session_id, question):
    mlflow.update_current_trace(
        user=user_id,           # 必须是专用 user 参数，不是 metadata={...}
        session_id=session_id,  # 专用 session_id 参数
        tags={"user_segment": "premium"},
    )
    return call_llm(question)
```

什么时候用：聊天/客服/Agent 这类"多用户多会话"场景，事后要按 user 或 session 拉历史。

### 模式 4：搜索历史 trace

```python
import mlflow

# 按用户搜
traces = mlflow.search_traces(
    experiment_ids=["1"],                       # 即将废弃，3.0 后用 locations
    filter_string="metadata.`mlflow.trace.user` = 'alice'",
)

# 按延迟排序找最慢的
traces = mlflow.search_traces(
    experiment_ids=["1"],
    order_by=["execution_time_ms DESC"],        # ⚠️ 字符串 list，且字段叫 execution_time_ms
    max_results=10,
)
```

什么时候用：离线分析、画报表、debug 线上问题。

## 三、实战步骤：按顺序照做

### Step 0：准备 .env

在项目根目录：

```bash
cp .env.example .env
# 编辑 .env，至少填一个：
# DEEPSEEK_API_KEY=sk-xxxxxxxx
# DEEPSEEK_MODEL=deepseek-chat
```

### Step 1：跑连通性测试

```bash
cd 05_tracing
python 05a_env_test.py
```

应该看到 `✓ Phase 5 环境就绪！`。没看到这个就回头检查 `.env`。

### Step 2：跑自动追踪 demo

```bash
python 05b_basic_tracing.py
```

跑完另开终端：

```bash
mlflow ui --port 5000
```

浏览器开 `http://localhost:5000` → 选 `05_basic_tracing` experiment → 点 `multi-turn-chat` Run → 看 **Traces** 标签 → 能看到 3 条 trace（每次 LLM 调用一条）。

### Step 3：跑自定义 Span demo（RAG）

```bash
python 05c_custom_decorator.py
```

回 UI → experiment `05_custom_tracing` → Run `rag-q1` → Traces → 点开 trace → 应该看到**5 层 Span 嵌套**（CHAIN → RETRIEVER → TOOL → PROMPT_TEMPLATE → LLM）。

### Step 4：跑元数据 + 搜索 demo

```bash
python 05d_metadata_search.py
```

跑完会直接打印搜索结果（不需要进 UI 看）。回 UI 后看 `05_metadata` experiment，Traces 标签左侧的过滤框可以试试用 `metadata.mlflow.trace.user = 'alice'` 过滤。

### Step 5（选跑）：打开 mlflow.db 自己玩玩

```bash
sqlite3 mlflow.db "SELECT * FROM traces LIMIT 3;"
# 或者用 mlflow.search_traces() 在 Python 里查
```

## 四、避坑清单

- ⚠️ **`update_current_trace` 用错地方** → 必须在 `@mlflow.trace` 装饰的函数内、或 `with mlflow.start_span(...)` 代码块内调用，普通函数里调用会报"no active trace"。
- ⚠️ **想给 trace 加 user 却用了 `metadata={"mlflow.trace.user": ...}`** → ❌ 这是保留字段，必须用专用关键字 `mlflow.update_current_trace(user=user_id, session_id=sid)`。
- ⚠️ **`search_traces` 的 `experiment_ids` 参数** → 在 MLflow 2.x 还能用，3.0 起会废弃，建议尽快迁移到 `locations=[exp_id]`。
- ⚠️ **`search_traces` 的 `order_by`** → 是 `list[str]`，不是 `list[dict]`；字段名是 `execution_time_ms`（不是 `execution_duration`，Duration 是返回 DataFrame 里的列名，别搞混）。
- ⚠️ **没看到 trace** → 检查是不是开了 autolog 但没在 UI 里选对 experiment；还有些包需要先 `pip install`，比如 `mlflow[genai]`。
- ⚠️ **国内连 OpenAI 直连超时** → 用 `env_bootstrap.py` 桥接到 DeepSeek/智谱/百炼，OpenAI SDK 通过 `OPENAI_API_BASE` 走兼容协议。

## 五、小结：5 个 take-aways

- **Tracing = 让 LLM 应用的每一步都"看得见"**，debug 和性能优化都靠它
- **`mlflow.openai.autolog()` 一行开启**，业务代码不用改；想追踪自己写的函数就用 `@mlflow.trace` 装饰
- **Span 是嵌套的**，函数调用嵌套 → Span 嵌套；UI 上看 span 树就像看调用栈
- **user / session 必须用专用参数**（`update_current_trace(user=..., session_id=...)`），不能用普通 metadata dict 写
- **`search_traces` 是结构化查询入口**，可以用 `metadata.\`mlflow.trace.user\`` 过滤、按 `execution_time_ms` 排序，调试时找特定用户或最慢调用特别管用
