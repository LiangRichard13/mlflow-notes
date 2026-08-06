# 阶段 10 学习笔记：SupportPilot 端到端客服 Copilot（毕业项目）

> 对应脚本：`scripts/capstone/capstone_support_pilot.py`
> 需要 API Key：是（DeepSeek / OpenAI 兼容接口）

## 🎯 这篇笔记做什么

**SupportPilot** 是一个生产级 GenAI 客服 Copilot 完整示例。它的特殊之处在于：**真实生产环境的 AI 应用从来不是单一模型就能搞定的**——通常需要"传统 ML 当安全网 + LLM 当主力"组合拳。这篇笔记会带你把这个真实场景跑一遍，并把前面 9 个 phase 学过的能力全部串起来。

打个比方：SupportPilot 就像一家银行的"客服中心"——门口站着一位传统保安（sklearn gate）先识别你是不是来办正事的，正事才会被领进大厅交给专业的 AI 柜员（LangChain RAG Agent）。这样既快又稳，还方便审计。

**跑完你能得到：**
- 一个用 sklearn 训练、注册到 Model Registry 的"意图分类器"（gate）
- 一个用 Prompt Registry 管理的客服 prompt
- 一个用 LangChain + DeepSeek 搭建的 RAG agent
- 8 个真实场景的端到端 trace 树（pipeline → agent → search_kb → LLM 四层 Span）
- 一份 `mlflow.genai.evaluate` 自动生成的评估报告

### 你会学到什么

- 把 ML + LLM 组合成生产级应用的双层架构
- 跨所有 phase 能力的整合（autolog、Model Registry、Prompt Registry、Trace、GenAI Eval）
- 端到端 trace 树是什么样、Span 怎么嵌套
- LLM-as-judge 评估和自定义 scorer 怎么写
- 把所有能力"串成一个项目"是怎么组织的

### 对应脚本清单

| 脚本 | 一句话作用 | 是否必跑 | 前置 |
|------|-----------|---------|------|
| `scripts/capstone/capstone_support_pilot.py` | 端到端：sklearn gate + LangChain RAG + Prompt Registry + GenAI eval | ✓ 必跑 | Phase 1-7 全部完成 |

### 前置知识

- **必须完成**：Phase 1-7（建议 Phase 8-9 也浏览过）
- **本阶段假设你懂**：
  - `mlflow.sklearn.autolog` / `mlflow.sklearn.log_model`（Phase 2）
  - Model Registry + Aliases（Phase 2）
  - Prompt Registry（Phase 8）
  - `@mlflow.trace` + SpanType（Phase 4）
  - `mlflow.genai.evaluate` + `@scorer`（Phase 7）
- **本阶段不假设你懂**：所有能力怎么"组合"——这就是这一阶段要展示的
- **需要的依赖**：`mlflow`、`scikit-learn`、`pandas`、`langchain`、`langchain-openai`、`openai`
- **需要的环境变量**：`OPENAI_API_KEY`、`OPENAI_API_BASE`、`DEEPSEEK_MODEL`（参考 `env_bootstrap.py`）

### 跑完必看（UI）

1. 启动 UI：`mlflow ui --port 5000`
2. 选 experiment `capstone_support_pilot`
3. 看到两个 Run：
   - `intent-gate`：sklearn 训练详情（autolog 的 Pipeline、签名、注册版本）
   - `capstone-demo`：**重点**——8 次端到端 trace 树
4. **点开 `capstone-demo` 的 Traces 标签**，任选一条 trace，会看到 4 层 Span 嵌套：
   ```
   support_pilot_pipeline  (PIPELINE)
     └─ support_pilot      (AGENT)
          ├─ search_kb      (RETRIEVER)
          └─ ChatOpenAI     (LLM, autolog 自动捕获)
   ```
5. 左侧栏浏览：
   - **Logged Models** → `support-pilot-v1`（演示 set_active_model）
   - **Prompts** → `support-pilot-prompt` 看到 v1, v2（演示版本管理）
   - **Models** → `IntentGate` → `champion` 别名（演示 Model Registry）

---

## 一、核心概念：用人话讲清楚

### 1. 为什么生产级 GenAI 应用需要"双层"架构？

直接让 LLM 回答所有用户问题，看着简单，但有三个隐患：

1. **不可控**：用户问"写首唐诗"或者"推荐餐厅"——这些跟你的客服业务无关，LLM 会瞎答
2. **浪费钱**：每次"唐诗"请求都要调一次 LLM，还要写 trace
3. **难审计**：合规要求知道"哪些问题被过滤掉了"

所以加一层**传统 ML gate**：一个超轻量的 sklearn 分类器，毫秒级判定 in_scope / out_of_scope，out_of_scope 直接挡掉。这层虽然精度有限，但便宜、快、可解释——是"粗筛"的最佳选择。

### 2. 整体架构图

```
         用户问题
            │
            ▼
   ┌─────────────────────┐
   │ sklearn gate        │  ← TF-IDF + 逻辑回归 Pipeline
   │ IntentGate@champion │     mlflow.sklearn.autolog 自动记录
   └─────────┬───────────┘     Model Registry 注册 + 别名
             │
        intent ∈ {in_scope, out_of_scope}
             │
      ┌──────┴──────┐
      ▼             ▼
   reject       ┌────────────────────────┐
 "无法回答"     │ LangChain RAG Agent    │
                │                        │
                │ ① search_kb (RETRIEVER)│  ← @mlflow.trace
                │    关键词检索 KB         │
                │                        │
                │ ② 拼 prompt (从 Registry│  ← mlflow.genai.load_prompt
                │    加载 @production 版本)│
                │                        │
                │ ③ ChatOpenAI (LLM)     │  ← mlflow.openai.autolog
                │    DeepSeek 生成回答     │
                └────────────┬───────────┘
                             │
                             ▼
                       答案 + [source:xxx] 引用
```

**关键设计点**：
- gate 的判定会被 trace 完整记录（你可以审计"哪些问题被拒了"）
- prompt 不写死在代码里，全在 Prompt Registry 里——升级 prompt 不用改代码
- LLM 调用自带 autolog，连 token 数都自动记下来

### 3. SupportPilot 用到的 MLflow 能力全清单

| 能力 | 用在哪里 | 对应 phase |
|------|---------|-----------|
| `mlflow.sklearn.autolog` | IntentGate 训练时自动记参数/指标/模型 | Phase 2 |
| `mlflow.sklearn.log_model` + `infer_signature` | 把 gate 模型带签名注册 | Phase 2 |
| Model Registry + Aliases | `IntentGate@champion` 固定生产版本 | Phase 2 |
| Prompt Registry | `support-pilot-prompt@production` | Phase 8 |
| `mlflow.langchain` 风格 trace | LangChain chain 调用追踪 | Phase 4 |
| `mlflow.openai.autolog` | DeepSeek（OpenAI 兼容）调用自动追踪 | Phase 4 |
| `@mlflow.trace(span_type=...)` | 自定义 Span（RETRIEVER/AGENT/PIPELINE）| Phase 4 |
| `mlflow.set_active_model` | 关联 trace 到 LoggedModel `support-pilot-v1` | Phase 8 |
| `mlflow.genai.evaluate` | 系统化评估整个 pipeline | Phase 7 |
| `Correctness` 内置 judge | LLM-as-judge 打分 | Phase 7 |
| `@scorer` 自定义 | `has_citation` 检查 `[source:xxx]` 引用 | Phase 7 |

---

## 二、代码模式：可复用的模板

### 模板 1：训练 sklearn 模型 + 注册到 Model Registry + 设别名

```python
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

mlflow.set_experiment("my_experiment")
mlflow.sklearn.autolog()  # 自动记参数/指标/模型

with mlflow.start_run(run_name="train_v1") as run:
    pipe = Pipeline([("tfidf", ...), ("clf", ...)])
    pipe.fit(X_train, y_train)
    
    # 带签名 + input_example 注册
    signature = infer_signature(X_train.head(3), pipe.predict(X_train.head(3)))
    mlflow.sklearn.log_model(pipe, name="model", signature=signature,
                             input_example=X_train.head(1))
    
    # 注册到 Registry 并设别名
    result = mlflow.register_model(f"runs:/{run.info.run_id}/model", "MyModel")
    MlflowClient().set_registered_model_alias("MyModel", "champion", result.version)
```

**什么时候用**：任何 sklearn 模型想上线时——autolog + 签名 + 别名是三件套。

### 模板 2：注册 prompt 到 Prompt Registry 并设 production

```python
# 注册（每次调用都创建新版本）
mlflow.genai.register_prompt(
    name="my-prompt",
    template=[{"role": "system", "content": "..."}, ...],
    commit_message="v1: 初始版本"
)

# 设别名
client = MlflowClient()
client.set_prompt_alias(name="my-prompt", alias="production", version=1)

# 加载
prompt_obj = mlflow.genai.load_prompt("prompts:/my-prompt@production")
```

**什么时候用**：prompt 要在生产稳定使用、又允许快速迭代——Registry 是答案。

### 模板 3：自定义 Span 标记关键组件

```python
@mlflow.trace(span_type="RETRIEVER", name="search_kb")
def search_kb(question: str) -> str:
    """检索 KB——Span 类型标记为 RETRIEVER"""
    ...

@mlflow.trace(span_type="AGENT", name="support_pilot")
def support_pilot(question: str) -> dict:
    """主 agent——Span 类型标记为 AGENT"""
    ctx = search_kb(question)
    answer = llm.invoke(messages)
    return {"answer": answer}

@mlflow.trace(span_type="PIPELINE", name="support_pilot_pipeline")
def pipeline(question, gate_model, agent):
    """端到端——Span 类型标记为 PIPELINE"""
    intent = gate_model.predict([question])[0]
    if intent == "out_of_scope":
        return {"answer": "拒绝"}
    return agent(question)
```

**什么时候用**：trace 树要清晰时——SpanType 是 trace 的"语义标签"，UI 上能直接分类过滤。

### 模板 4：自定义 scorer

```python
from mlflow.genai.scorers import scorer

@scorer(name="has_citation")
def has_citation(outputs: str) -> bool:
    """检查回答是否带 [source:xxx] 引用"""
    return bool(re.search(r"\[source:[^\]]+\]", outputs))

# 用法
mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=lambda row: my_predict(row["question"]),
    scorers=[Correctness(model=judge), has_citation],  # 内置 + 自定义混合
)
```

**什么时候用**：内置 scorer 不够用时——比如业务要"必须有引用"、"必须用礼貌用语"等自定义规则。

---

## 三、实战步骤：按顺序照做

### 第 1 步：确认前置 phase 都跑通过

```bash
# 至少跑过 Phase 1-7 的核心脚本，建立基础认知
ls notes/   # 应该能看到 phase_1.md ... phase_9.md
```

### 第 2 步：配置 API Key

确认 `env_bootstrap.py` 或环境变量里设置了：

```bash
export OPENAI_API_KEY="sk-xxx"
export OPENAI_API_BASE="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-chat"
```

### 第 3 步：跑 capstone 脚本

```bash
cd /path/to/MLFlowLearning
conda activate mlflow
python scripts/capstone/capstone_support_pilot.py
```

预期输出（节选）：
```
[1/5] 训练意图分类器（sklearn gate）...
  ✓ 意图分类器训练完成 (acc=0.75, f1=0.75), 注册为 v1
[2/5] 注册 SupportPilot prompts...
  ✓ 已注册 2 个 prompt 版本（最新 v2）
  ✓ production → v2（最新版）
[3/5] 构造 SupportPilot agent（用 production prompt）...
[4/5] 端到端跑 8 个真实场景...
  [1] Q: 退款怎么操作？
      intent=in_scope (expected=in_scope) status=answered
      A: 您好！购买后 7 天内可全额退款... [source:kb1]
  ...
[5/5] mlflow.genai.evaluate 评估...
  聚合指标:
    has_citation/mean: 0.750
    correctness/mean: 0.500
```

### 第 4 步：打开 UI 查看

```bash
mlflow ui --port 5000
```

浏览器打开 `http://localhost:5000`，依次看：
1. experiment `capstone_support_pilot`
2. Run `intent-gate` → 看 sklearn autolog 的参数、指标、签名
3. Run `capstone-demo` → 看 8 条 trace，每条都是 4 层 Span 树
4. 左侧 **Logged Models** → `support-pilot-v1`
5. 左侧 **Prompts** → `support-pilot-prompt` v1, v2
6. 左侧 **Models** → `IntentGate` → `champion` 别名

---

## 四、避坑清单

- ⚠️ **prompt 渲染的坑**：`mlflow.genai.PromptVersion.format(**kwargs)` 对缺失变量会报错。
  → 解决：调 `format()` 之前先看 `prompt_obj.variables`，把缺失的补齐；或者直接用 `f"{{ var }}"` 手动替换（更可控，capstone 脚本里就是这么干的）。

- ⚠️ **`predict_fn` 签名约束**：必须接受一个 dict 参数，参数名要和 `data.inputs` 的 key 对应。
  → 解决：capstone 里用 `lambda row: agent(row["question"])`，因为 eval_data 的 inputs 是 `{"row": {"question": ...}}`。

- ⚠️ **`judge_model` 必须显式传**：内置 scorer（如 `Correctness`）默认用 `gpt-4.1-mini`，国内 API 不支持。
  → 解决：`Correctness(model=f"openai:/{os.getenv('DEEPSEEK_MODEL')}")`，让 judge 也走 DeepSeek。

- ⚠️ **LLM judge 兼容性**：用 `make_judge` 时 instructions 必须包含 `{{ inputs }}` / `{{ outputs }}` 等变量占位符。
  → 解决：参考 Phase 7 笔记里 make_judge 的 instructions 模板。

- ⚠️ **gate 数据太少只有 21 条**：分类器对 out_of_scope 容易误判。
  → 解决：生产前收集 100+ 真实 query 重新训练；可以加上"关键词规则"做兜底。

- ⚠️ **PII 必须在 trace 边界脱敏**：等 trace 记完再清洗已经泄漏了。
  → 解决：在 `predict_fn` 入口处用正则把手机号/身份证号替换掉，再让 trace 记录。

- ⚠️ **OpenAI 兼容接口的 base_url**：不同服务商路径不同（如 `https://api.deepseek.com/v1`）。
  → 解决：参考 `env_bootstrap.py` 的写法，写到环境变量里集中管理。

---

## 五、小结：3-5 个 take-aways

1. **生产级 GenAI = ML + LLM 组合拳**：sklearn gate 做粗筛 + LangChain RAG 做主力，既省钱又快，还能审计。
2. **Prompt Registry 是"prompt 的 git"**：prompt 不写死在代码里，迭代不部署、改版本不改代码。
3. **SpanType 是 trace 的语义标签**：标 RETRIEVER / AGENT / PIPELINE 让 trace 树一眼能看懂结构。
4. **LoggedModel 把所有 trace 串成一个版本**：用 `set_active_model` 把同一代 pipeline 的所有 trace 归到 `support-pilot-v1`，方便对比 v1 vs v2。
5. **GenAI eval 要混合内置 + 自定义 scorer**：内置 `Correctness` 打语义，自定义 `has_citation` 打业务规则，组合起来才全面。

---

## 六、下一步改进方向（练习思路）

1. **gate 数据增强**：现在只有 21 条训练数据，收集 100+ 真实用户 query 重新训练 IntentGate。
2. **KB 检索升级**：从关键词检索换成向量检索（FAISS / Chroma / pgvector），retriever 性能提升明显。
3. **Prompt A/B 对比**：注册 v3 prompt（更详细的指令），用 `mlflow.validate_evaluation_results` 对比 v2 vs v3，选 winner。
4. **加 trace 采样 + PII 脱敏**：参考 `scripts/09_deployment/09a_sampling_redaction.py`，生产环境必备。
5. **改成 ResponsesAgent + `mlflow.pyfunc.log_model`**：把整套 pipeline 包成一个 pyfunc 模型上线（参考 `scripts/08_agents/08c_responses_agent.py`）。
6. **用 `mlflow.genai.optimize_prompts` 自动优化 prompt**：参考 `scripts/08_agents/08b_prompt_optimize.py`，让 MLflow 自动找最佳 prompt。
7. **加 LLM judge 评估 gate 决策**：自定义 scorer 检查"out_of_scope 是否被正确拒绝"（capstone 里 `was_rejected_for_oos` 已经写了示例）。

---

✓ capstone 完成