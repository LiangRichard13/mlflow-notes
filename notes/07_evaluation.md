# 阶段 7 学习笔记：GenAI 评估与自定义 Scorer

> 对应脚本：`07_evaluation/07a_basic_evaluate.py`、`07b_custom_scorer.py`、`07c_prompt_comparison.py`

## 一、`mlflow.genai.evaluate()` 基础

```python
result = mlflow.genai.evaluate(
    data=df,                   # pandas DataFrame
    predict_fn=my_predict,     # 接受 row dict，返回字符串
    scorers=[Correctness(), Safety(), my_scorer],
)
```

**返回**：
- `result.metrics`：聚合指标（dict）
- `result.tables`：逐行打分（DataFrame）

## 二、数据格式（关键！）

每行必须有 `inputs` 列（dict），可选 `expectations` 列：

```python
# ✅ 正确（只用 expected_response）
{"inputs": {"q": "..."}, "expectations": {"expected_response": "..."}}

# ✅ 正确（只用 expected_facts）
{"inputs": {"q": "..."}, "expectations": {"expected_facts": ["事实1", "事实2"]}}

# ⚠️ 同时给两个会触发额外的 judge 调用（用默认模型，国内服务商不支持）
{"inputs": {...}, "expectations": {"expected_response": "...", "expected_facts": [...]}}
```

## 三、predict_fn 的签名约束

```python
# MLflow 会用第一行数据验证 predict_fn 签名
# predict_fn 参数名必须与 inputs 的 key 对应

# ✅ 正确：单个参数 "row"
predict_fn=lambda row: f"Answer to {row['q']}"
data = [{"inputs": {"row": {"q": "..."}}}]

# ❌ 错误：predict_fn 不接收参数
predict_fn=lambda: "..."
```

## 四、内置 Scorers

```python
from mlflow.genai.scorers import Correctness, Safety, RelevanceToQuery

# ⚠️ 关键：内置 scorer 默认用 gpt-4.1-mini 当 judge
# 国内服务商不支持，必须显式传 model 参数
# URI 格式必须是 <provider>:/<model-name>
judge_model = f"openai:/{os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-flash')}"

scorers = [
    Correctness(model=judge_model),
    Safety(model=judge_model),
    RelevanceToQuery(model=judge_model),
    Guidelines(model=judge_model, guidelines="回答必须专业"),
]
```

**完整内置 scorer 列表**（mlflow.genai.scorers）：
- 通用：`Correctness`, `Equivalence`, `Safety`, `Fluency`
- 检索增强：`RetrievalGroundedness`, `RetrievalRelevance`, `RetrievalSufficiency`
- 对话：`ConversationalGuidelines`, `ConversationalSafety`, `ConversationCompleteness`, `UserFrustration`
- 工具：`ToolCallCorrectness`, `ToolCallEfficiency`
- 其他：`Guidelines`, `PIIDetection`, `RegexMatch`, `RelevanceToQuery`, `ResponseLength`

## 五、自定义 Python Scorer（@scorer）

```python
from mlflow.genai.scorers import scorer

@scorer(name="has_citation")
def has_citation(outputs: str) -> bool:
    return "[source:" in (outputs or "")

@scorer(name="response_quality")
def response_quality(inputs: dict, outputs: str) -> float:
    # 参数名必须从 inputs/outputs/expectations 中选
    if not outputs:
        return 0.0
    return min(len(outputs) / 100, 1.0)
```

参数约定（按需选择）：
- `inputs`：dict（原始输入）
- `outputs`：任意（predict_fn 返回值）
- `expectations`：dict（数据集期望）

返回类型：bool / float / int / str

## 六、LLM-as-Judge Scorer（make_judge）

```python
# ⚠️ instructions 必须包含至少一个变量：{{ inputs }} / {{ outputs }} / {{ trace }}
tone_judge = mlflow.genai.make_judge(
    name="brand_tone",
    instructions=(
        "评估 {{ outputs }} 的语气：\n"
        "- 友好但不轻浮\n"
        "- 专业但不冷漠\n"
        "打分 1-5"
    ),
    model="openai:/deepseek-v4-flash",
)
```

## 七、跨版本对比（Prompt A/B）

```python
# 用 production alias 跑一遍
with mlflow.start_run(run_name="production"):
    result_a = mlflow.genai.evaluate(data=df, predict_fn=pred_prod, scorers=[...])

# 用 staging alias 跑一遍
with mlflow.start_run(run_name="staging"):
    result_b = mlflow.genai.evaluate(data=df, predict_fn=pred_stg, scorers=[...])

# 在 UI Compare 两个 Run，看差异
```

## 八、prompt 渲染（chat 格式 + 变量）

```python
prompt_obj = mlflow.genai.load_prompt("prompts:/name@alias")
print(prompt_obj.variables)        # {'question', 'company'}

# 文本 prompt
if prompt_obj.is_text_prompt:
    text = prompt_obj.format(question="...", company="...")
# chat prompt (list[dict])
else:
    messages = prompt_obj.format(question="...", company="...")
```

## 九、完整工作流

```
1. mlflow.genai.evaluate() 评估 baseline
2. 改 prompt / 改模型 / 改 predict_fn
3. 再 evaluate 一遍
4. UI Compare 两个 Run
5. winner → set_prompt_alias 切生产
6. 失败的 → 保留旧版本以便回滚
```