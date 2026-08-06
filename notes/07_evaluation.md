# 阶段 7 学习笔记：GenAI 评估与自定义 Scorer

> 对应脚本：`scripts/07_evaluation/07a_basic_evaluate.py`、`07b_custom_scorer.py`、`07c_prompt_comparison.py`
> 需要 API Key：是

## 🎯 这篇笔记做什么

写完一个 LLM 应用后，怎么知道它"答得好不好"？传统 ML 有 accuracy、RMSE 这些现成指标，但 LLM 的回答是自然语言——既要看"对不对"，也要看"语气是否友好"、"有没有胡编"。这一阶段就教你用 **MLflow GenAI 评估**给 LLM 应用"打分"。

类比：想象你在教一个实习生回答客户问题。
- **内置 Scorer** 就像公司统一出的"评分卡"（正确性、安全性、切题度），谁都能用。
- **@scorer 自定义** 就像你给这位实习生写的"特殊规矩"（必须加引用、不能超过 100 字）。
- **make_judge** 就像你雇了个资深主管当裁判，用自然语言写评分标准（"语气要友好专业"），让 LLM 来评判 LLM。
- **跨版本对比** 就像你改了实习生的培训手册，跑同一份题库，对比"旧版"和"新版"哪个更好。

**跑完的产出**：MLflow UI 里能看到两次评估运行的对比，每个指标都有数字、每个问题都有逐行打分，决策时一目了然。

### 你会学到什么

- 用 `mlflow.genai.evaluate()` 跑一次完整评估
- 配置内置 Scorer（Correctness / Safety / RelevanceToQuery）并解决 judge 模型问题
- 写 `@scorer` 业务规则型评分器
- 用 `make_judge()` 写 LLM-as-judge 主观评分器
- 跨 prompt 版本做 A/B 评估并根据结果做切流决策

### 对应脚本清单

| 脚本 | 一句话作用 | 是否必跑 | 前置 |
|------|-----------|---------|------|
| `07a_basic_evaluate.py` | 用内置 Scorer 评估 LLM 应用 | ✓ 必跑 | Phase 5-6 |
| `07b_custom_scorer.py` | 组合内置 + @scorer + make_judge 评估 | 推荐 | 跑过 07a |
| `07c_prompt_comparison.py` | production vs staging prompt A/B 评估 + 决策建议 | 推荐 | 跑过 07a，注册过 prompt |

### 前置知识

- 已完成 Phase 5-6，会用 `mlflow.openai.autolog()` 和 Prompt Registry
- 已配置 `OPENAI_API_KEY`、`OPENAI_API_BASE`、`DEEPSEEK_MODEL` 环境变量
- 已 `pip install mlflow openai pandas`
- 本地有 `mlflow.db`（Phase 5-6 已创建）

### 跑完必看（UI）

1. 启动 UI：`mlflow ui --port 5000`
2. 选 experiment `07_evaluate`（或 `07_custom_scorer`、`07_prompt_ab`）
3. 点开 Run `production` 或 `staging` 或最新一次评估
4. 看：
   - **Metrics 标签**：`correctness/mean`、`safety/mean`、`relevance_to_query/mean`、`has_citation/mean` 等聚合分数
   - **Artifacts**：评估明细 JSON（每行打分）
   - **Traces 标签**：每次 predict_fn 调用的完整 trace
   - 跨版本对比：勾选两个 Run → 点 Compare → 看每个指标的差异

---

## 一、核心概念：用人话讲清楚

### 1. `mlflow.genai.evaluate()` 是什么？

它是 MLflow 给 GenAI 应用准备的"考试系统"。你给它三样东西：
- **题库**（`data`：DataFrame，每行有 `inputs` 和可选的 `expectations`）
- **考生**（`predict_fn`：你的 LLM 应用，接收题库的一行，返回答案）
- **评分标准**（`scorers`：一组评分器，对答案打分）

它会自动跑题库、用 `predict_fn` 调 LLM 拿答案、再用每个 scorer 打分，最后产出聚合指标和逐行结果。

### 2. 内置 Scorer vs 自定义 Scorer vs LLM-as-judge

| 类型 | 适合场景 | 例子 |
|------|---------|------|
| **内置 Scorer** | 通用质量检查（正确性、安全性、切题） | `Correctness()`、`Safety()`、`RelevanceToQuery()` |
| **@scorer** | 硬性业务规则（必须有引用、不能超字数、关键词计数） | `has_citation`、`is_concise` |
| **make_judge** | 主观/复杂判断（语气风格、合理性） | `brand_tone`（评估语气） |

### 3. judge_model 是什么？为什么必须显式传？

内置 Scorer 和 `make_judge` 本质都是让 **另一个 LLM** 来评分——这个"裁判 LLM"就叫 **judge model**。

MLflow 默认的 judge model 是 `gpt-4.1-mini`（OpenAI 直连）。如果你的 `OPENAI_API_BASE` 指向的是国内代理（DeepSeek、月之暗面等），**它们根本不认识 `gpt-4.1-mini`**，会报 404 或模型不存在错误。

**解决办法**：显式传 `model="openai:/你的模型名"`，让 MLflow 通过你的 `base_url` 调 judge 模型。

```python
# ⚠️ 关键：内置 scorer 默认用 gpt-4.1-mini 当 judge
# 国内服务商不支持，必须显式传 model 参数
# URI 格式必须是 <provider>:/<model-name>
judge_model = f"openai:/{os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-flash')}"

scorers = [
    Correctness(model=judge_model),
    Safety(model=judge_model),
    RelevanceToQuery(model=judge_model),
]
```

### 4. `expected_response` vs `expected_facts`（不能同时给！）

`Correctness` 这个 scorer 想知道"正确答案长什么样"，有两种方式告诉它：

- `expected_response`：完整参考答案字符串（"MLflow 是开源的 ML 生命周期管理平台"）
- `expected_facts`：关键事实列表（`["MLflow 是开源的", "ML 生命周期管理"]`）

**重要**：**两者只能选一个**，不能同时给！如果同时给，MLflow 会触发额外的 judge 调用（用默认的 `gpt-4.1-mini`），国内服务商又会挂掉。

### 5. 跨 prompt 版本对比的思路

把"新旧 prompt"当成两个"考生 A、B"，跑同一份题库，比较谁分高：

```
production (v3) ─┐
                 ├─→ 同一份题库 + 同一组 scorer → 看 metrics 差异
staging (v4)   ─┘
```

决策：分高的胜出 → 把胜出版本用 `set_prompt_alias` 设为新的 production；输的版本保留在 Registry 方便回滚。

---

## 二、代码模式：可复用的模板

### 模板 1：基础评估

```python
import mlflow
import pandas as pd
from mlflow.genai.scorers import Correctness, Safety, RelevanceToQuery

# 1. 准备题库（每行 inputs 是 dict，expectations 是期望答案）
EVAL_DATA = pd.DataFrame([
    {"inputs": {"question": "什么是 MLflow？"},
     "expectations": {"expected_response": "MLflow 是开源的 ML 生命周期管理平台"}},
    {"inputs": {"question": "MLflow Tracking 干什么的？"},
     "expectations": {"expected_response": "记录实验参数、指标、artifact"}},
])

# 2. 定义考生（你的 LLM 应用）
def predict_fn(question: str) -> str:
    import os
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )
    resp = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        messages=[{"role": "user", "content": question}],
        max_tokens=200,
    )
    return resp.choices[0].message.content

# 3. 配置 judge 模型（关键！必须显式传）
judge_model = f"openai:/{os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-flash')}"

# 4. 跑评估
result = mlflow.genai.evaluate(
    data=EVAL_DATA,
    predict_fn=predict_fn,
    scorers=[
        Correctness(model=judge_model),
        Safety(model=judge_model),
        RelevanceToQuery(model=judge_model),
    ],
)

# 5. 看结果
print(result.metrics)        # 聚合指标 dict
print(result.tables.keys())  # ['eval_results'] 等
```

### 模板 2：@scorer 自定义业务规则

```python
from mlflow.genai.scorers import scorer

@scorer(name="has_citation")
def has_citation(outputs: str) -> bool:
    """硬性规则：回答必须包含 [source:xxx] 引用"""
    return "[source:" in (outputs or "")

@scorer(name="response_length_ok")
def response_length_ok(outputs: str) -> bool:
    """回答长度 50-200 字"""
    if not isinstance(outputs, str):
        return False
    return 50 <= len(outputs) <= 200
```

**参数约定**（按需选择，不能随便命名）：
- `inputs`：dict（原始输入）
- `outputs`：任意（predict_fn 返回值）
- `expectations`：dict（数据集期望列）

**返回类型**：`bool` / `float` / `int` / `str`（MLflow 会自动聚合求 mean）

### 模板 3：make_judge LLM-as-judge

```python
# ⚠️ instructions 必须包含至少一个变量：{{ inputs }} / {{ outputs }} / {{ trace }}
brand_tone_judge = mlflow.genai.make_judge(
    name="brand_tone",
    instructions=(
        "评估 {{ outputs }} 的语气：\n"
        "- 友好但不轻浮\n"
        "- 专业但不冷漠\n"
        "- 简洁但不敷衍\n"
        "打分 1-5：1=完全不像品牌，3=一般，5=非常符合"
    ),
    model="openai:/deepseek-v4-flash",  # 同样要显式传 judge model
)
```

**什么时候用 make_judge**：主观判断（语气、合理性、相关性），用自然语言写 rubric 比写代码更直观。

### 模板 4：predict_fn 的签名约束

MLflow 会用第一行数据验证 `predict_fn` 签名——**参数名必须与 inputs 的 key 对应**：

```python
# ✅ 正确：单个参数 "row"，从 row dict 里取值
predict_fn=lambda row: predict_with_prompt_version("production", row["question"])
data = [{"inputs": {"row": {"question": "..."}}}]

# ✅ 也正确：直接命名为 inputs 里的 key
predict_fn=lambda question: answer(question)
data = [{"inputs": {"question": "..."}}]

# ❌ 错误：不接收参数
predict_fn=lambda: "..."
```

### 模板 5：跨版本对比

```python
judge = f"openai:/{os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-flash')}"

# 跑 production
with mlflow.start_run(run_name="production"):
    mlflow.genai.evaluate(
        data=EVAL_DATA,
        predict_fn=lambda row: predict_with_prompt_version("production", row["question"]),
        scorers=[Correctness(model=judge), Safety(model=judge)],
    )

# 跑 staging
with mlflow.start_run(run_name="staging"):
    mlflow.genai.evaluate(
        data=EVAL_DATA,
        predict_fn=lambda row: predict_with_prompt_version("staging", row["question"]),
        scorers=[Correctness(model=judge), Safety(model=judge)],
    )

# 在 UI Compare 两个 Run 看差异
```

---

## 三、实战步骤：按顺序照做

### 步骤 1：跑 07a（基础评估）

```bash
cd <project-root>
conda activate mlflow
python scripts/07_evaluation/07a_basic_evaluate.py
```

跑完看终端输出：会打印所有 scorer 的聚合分数（如 `correctness/mean: 0.800`）。

### 步骤 2：打开 UI 看逐行打分

```bash
mlflow ui --port 5000
```

浏览器打开 `http://localhost:5000`：
1. 选 experiment `07_evaluate`
2. 点开最新的 Run
3. 看 **Metrics** 标签里的聚合分数
4. 看 **Evaluation** 标签（如果有）或 **Artifacts/eval/** 里的明细 JSON——每行能看到：问题、LLM 答案、每个 scorer 的打分 + reasoning

### 步骤 3：跑 07b（自定义 Scorer）

```bash
python scripts/07_evaluation/07b_custom_scorer.py
```

会一次性跑 6 个 scorer（3 个内置 + 3 个自定义）：
- `Correctness()`、`Safety()`（内置）
- `has_citation`、`is_concise`、`mentions_mlflow`（@scorer）
- `brand_tone`（make_judge）

注意：07b 里 `Correctness()` 和 `Safety()` **没有显式传 `model=`**——因为这份脚本不依赖 `expected_response`，这两个 scorer 会跑得比较"宽松"（不调 judge，只做基础检查）。如果要严格对比，必须传 judge_model。

### 步骤 4：跑 07c（跨 prompt 版本对比）

**前置**：确保你在 Phase 6 注册过 `customer-support-qa` 这个 prompt，并且有 `production` 和 `staging` 两个 alias。

```bash
python scripts/07_evaluation/07c_prompt_comparison.py
```

会做：
1. 用 `prompts:/customer-support-qa@production` 跑一遍评估
2. 用 `prompts:/customer-support-qa@staging` 跑一遍评估
3. 终端打印对比表 + 决策建议
4. 在 UI 里勾选 `production` 和 `staging` 两个 Run → 点 Compare 看差异

### 步骤 5：根据对比结果决策

终端输出会告诉你三种情况：
- **staging 比 production 高 10%+** → 建议切到 staging
- **production 明显更好** → 保持 production
- **两者差不多** → 继续 A/B 测试更多数据

切流时用：
```python
mlflow.genai.set_prompt_alias(
    "customer-support-qa",
    alias="production",
    version=<staging 的版本号>,
)
```

---

## 四、避坑清单

### 坑 1：内置 Scorer 默认用 `gpt-4.1-mini`（国内服务商不支持）

**症状**：跑 `Correctness()` 时报 `Model gpt-4.1-mini not found` 或 `404`。

**解决**：显式传 `model=f"openai:/{你的模型名}"`，让 MLflow 通过你的 `base_url` 调 judge 模型。

```python
# ✅ 正确
Correctness(model=f"openai:/{os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-flash')}")

# ❌ 错误（用默认值，国内服务商挂）
Correctness()
```

### 坑 2：`expected_response` 和 `expected_facts` 不能同时给

**症状**：同时给两个字段时，会触发额外的 judge 调用（用默认 `gpt-4.1-mini`），然后报模型不存在。

**解决**：二选一。如果你的参考答案是一个完整字符串，用 `expected_response`；如果只有几个关键事实，用 `expected_facts`。

```python
# ✅ 二选一
{"expectations": {"expected_response": "..."}}
{"expectations": {"expected_facts": ["事实1", "事实2"]}}

# ❌ 两个都给
{"expectations": {"expected_response": "...", "expected_facts": [...]}}
```

### 坑 3：`model=` 参数不能传 None 或空字符串

**症状**：传 `model=None` 会报错或悄悄退回到默认值。

**解决**：用 `os.getenv(..., 'deepseek-v4-flash')` 确保有非空默认值。

### 坑 4：`predict_fn` 签名不对

**症状**：跑评估时第一行就报错 `predict_fn() takes 0 positional arguments`。

**解决**：`predict_fn` 必须接收至少一个参数，参数名要与 `inputs` 里的 key 对应。

```python
# ✅ 正确
data = [{"inputs": {"question": "..."}}]
predict_fn = lambda question: answer(question)

# ✅ 也正确（用 "row" 包一层）
data = [{"inputs": {"row": {"question": "..."}}}]
predict_fn = lambda row: answer(row["question"])
```

### 坑 5：`make_judge` 的 instructions 忘了写变量

**症状**：`make_judge` 报 `instructions must contain at least one variable`。

**解决**：在 instructions 里至少包含 `{{ inputs }}`、`{{ outputs }}`、`{{ trace }}` 其中的一个。

```python
# ✅ 正确
instructions="评估 {{ outputs }} 的语气是否友好专业"

# ❌ 错误（没变量）
instructions="评估回答的语气是否友好专业"
```

### 坑 6：`@scorer` 函数参数名写错

**症状**：MLflow 找不到输入数据。

**解决**：参数名必须是 `inputs`、`outputs`、`expectations` 这三个（不能拼错、不能改名）。

```python
# ✅ 正确
@scorer
def my_scorer(outputs: str) -> bool:
    return len(outputs) > 0

# ❌ 错误
@scorer
def my_scorer(output: str) -> bool:  # "output" 不是约定名
    return len(output) > 0
```

### 坑 7：跨版本对比的题库要固定

**症状**：换题库对比没有意义（A 答对难的题、B 答对简单的题，无法判断哪个 prompt 更好）。

**解决**：跨版本对比必须用**同一份题库**、**同一组 scorer**——只换 `predict_fn` 里的 prompt 版本。

---

## 五、小结：3-5 个 take-aways

1. **`mlflow.genai.evaluate()` 是 GenAI 的"考试系统"**——给它题库（`data`）、考生（`predict_fn`）、评分标准（`scorers`），它自动跑完打分。返回 `result.metrics`（聚合分数）和 `result.tables`（逐行结果）。

2. **必须显式传 `model=` 给内置 Scorer 和 make_judge**——MLflow 默认用 `gpt-4.1-mini`，国内服务商不支持。统一用 `f"openai:/{os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-flash')}"` 当 judge_model。

3. **`expected_response` 和 `expected_facts` 不能同时给**——同时给会触发额外 judge 调用（用默认模型），然后挂掉。二选一。

4. **三种 Scorer 各有分工**：内置 Scorer 做通用质量检查、@scorer 写硬性业务规则、make_judge 处理主观判断（语气风格）。实际项目里三者组合用最灵活。

5. **跨 prompt 版本对比 = 同题库 + 不同 predict_fn**——在 UI 里勾选两个 Run → Compare 看指标差异。分高的胜出，用 `set_prompt_alias` 切到 production，输的版本保留方便回滚。

---

## 附录：完整内置 Scorer 清单

```python
from mlflow.genai.scorers import (
    # 通用质量
    Correctness,         # 对比 expectations（expected_response 或 expected_facts）
    Equivalence,         # 对比两个 outputs 是否等价
    Safety,              # 检测不安全内容
    Fluency,             # 语言流畅度
    # 检索增强（RAG）
    RetrievalGroundedness,   # 回答是否基于检索内容
    RetrievalRelevance,      # 检索内容是否相关
    RetrievalSufficiency,    # 检索内容是否足够
    # 对话
    ConversationalGuidelines,    # 对话是否符合规范
    ConversationalSafety,        # 对话安全性
    ConversationCompleteness,    # 对话完整性
    UserFrustration,             # 用户挫败感检测
    # 工具调用
    ToolCallCorrectness,     # 工具调用是否正确
    ToolCallEfficiency,      # 工具调用效率
    # 其他
    Guidelines,          # 自定义规范（自然语言写）
    PIIDetection,         # 检测 PII（个人敏感信息）
    RegexMatch,           # 正则匹配
    RelevanceToQuery,     # 回答是否切题
    ResponseLength,       # 回答长度
)
```

完整工作流回顾：

```
1. mlflow.genai.evaluate() 评估 baseline
2. 改 prompt / 改模型 / 改 predict_fn
3. 再 evaluate 一遍
4. UI Compare 两个 Run
5. winner → set_prompt_alias 切生产
6. 失败的 → 保留旧版本以便回滚
```