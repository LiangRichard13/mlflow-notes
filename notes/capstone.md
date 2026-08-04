# 毕业项目 SupportPilot 学习笔记

> 对应脚本：`capstone/capstone_support_pilot.py`

## 一、项目目标

**SupportPilot**：一个生产级 GenAI 客服 Copilot，组合：
- **传统 ML 安全网**：sklearn 意图分类器（gate）拦截 out-of-scope 问题
- **GenAI 主力**：LangChain RAG Agent（基于 Prompt Registry 的 prompt + KB 检索 + LLM）

## 二、架构

```
用户问题
   │
   ▼
┌─────────────────┐
│ sklearn gate     │   ← TF-IDF + 逻辑回归，TF-IDF Pipeline 自动 log
│ IntentGate      │     Model Registry 别名 @champion
└────────┬────────┘
         │ intent ∈ {in_scope, out_of_scope}
         │
   ┌─────┴─────┐
   │           │
   ▼           ▼
reject     ┌─────────────────┐
"无法回答" │ LangChain Agent  │  ← Prompt Registry @production
           │ + KB 检索       │     autolog + trace
           │ + DeepSeek      │
           └────────┬────────┘
                    │ 带 [source:xxx] 引用
                    ▼
                  答案
```

## 三、涉及的全部 MLflow 3 能力

| 能力 | 用在哪里 |
|------|---------|
| `mlflow.sklearn.autolog` | IntentGate 训练 |
| `mlflow.sklearn.log_model` | IntentGate 模型带签名注册 |
| Model Registry + Aliases | `IntentGate@champion` |
| Prompt Registry | `support-pilot-prompt@production`（v16+）|
| `mlflow.langchain` autolog | LangChain chain 调用追踪 |
| `mlflow.openai.autolog` | DeepSeek 调用追踪 |
| `@mlflow.trace(span_type=...)` | 自定义 Span（RETRIEVER/AGENT/PIPELINE）|
| `set_active_model` | 关联 trace 到 LoggedModel `support-pilot-v1` |
| `mlflow.genai.evaluate` | 系统化评估 |
| `Correctness` (内置 judge) | LLM-as-judge 评估 |
| `@scorer` 自定义 | `has_citation` 检查引用 |
| Trace 嵌套 | pipeline → agent → search_kb → LLM 四层 Span |

## 四、运行后看到的结果

```
聚合指标:
  has_citation/mean: 0.750   # 75% 回答带 [source:xxx]
  correctness/mean: 0.500   # 50% 答对 baseline
```

8 个测试用例：
- 5 个 in_scope（业务问题）→ 通过 gate，被 agent 回答
- 3 个 out_of_scope（无关问题）→ 训练数据太少，被误判为 in_scope（需更多数据）

## 五、UI 查看清单

```bash
conda activate mlflow
mlflow ui --port 5000
```

打开 `http://localhost:5000`：
1. experiment `capstone_support_pilot` → `intent-gate` Run：autolog sklearn 详情
2. `capstone-demo` Run：8 次端到端 trace 树（pipeline → agent → search_kb → LLM）
3. 左侧 `Logged Models` → `support-pilot-v1`
4. 左侧 `Prompts` → `support-pilot-prompt` v1...v16
5. 左侧 `Models` → `IntentGate` → `champion` 别名

## 六、改进方向（下一步练习）

1. **gate 数据太少**（只有 21 条）：收集 100+ 真实 query 重新训练
2. **KB 检索太简单**：换成向量检索（FAISS / Chroma）
3. **Prompt v1 vs v2 评估对比**：用 `mlflow.validate_evaluation_results` 选赢家
4. **加 trace 采样 + PII 脱敏**：参考 `09_deployment/09a_sampling_redaction.py`
5. **改成 ResponsesAgent + mlflow.pyfunc.log_model**：参考 `08_agents/08c_responses_agent.py`
6. **用 `mlflow.genai.optimize_prompts` 自动优化 prompt**：参考 `08_agents/08b_prompt_optimize.py`

## 七、核心教训

1. **prompt 渲染的坑**：`mlflow.genai.PromptVersion.format(**kwargs)` 对缺失变量会报错。
   - 必须在 `format()` 之前用 `prompt_obj.variables` 集合补齐缺失变量
   - 或者直接用 `{{ var }}` 手动替换（更可控）
2. **predict_fn 签名约束**：必须接受一个 dict 参数，参数名要与 data.inputs 的 key 对应
3. **judge_model 必须显式传**：内置 scorer 默认 `gpt-4.1-mini`，国内服务商不支持
4. **LLM judge 兼容性**：用 `make_judge` 时 instructions 必须包含 `{{ inputs }}` / `{{ outputs }}` 等变量
5. **PII 必须在 trace 边界脱敏**：不能等 trace 记完再清洗（已经泄漏了）