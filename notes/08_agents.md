# 阶段 8 学习笔记：版本追踪、提示词优化与 ResponsesAgent

> 对应脚本：`scripts/08_agents/08a_active_model.py`、`08b_prompt_optimize.py`、`08c_responses_agent.py`、`simple_qa_agent.py`

## 🎯 这篇笔记做什么

你已经把 LLM 应用跑起来了，现在要把它推向生产。生产环境有三个最棘手的问题：第一，你怎么知道线上跑的是哪个版本的代码？第二，谁负责把"蹩脚"的 prompt 改得更好——你自己手动改，还是让程序自动迭代？第三，你的 LLM 应用怎么变成一个标准的、能被 `mlflow models serve` 直接拉起来的服务？

这一阶段解决的就是这三个问题。我们会学到 MLflow 3 的三件新武器：**LoggedModel**（独立的"模型版本"实体，替代了 Run 的附庸地位）、**optimize_prompts**（让 reflection 模型自动改写你的 prompt 并打分）、**ResponsesAgent**（一个跟 OpenAI Responses API 兼容的 Agent 基类，可以打包成模型服务）。

**类比**：想象你开了家奶茶店。LoggedModel 就像"配方卡片"——"v1 经典版"、"v2 加椰果版"——每改一次配方就登记一张卡片，店里哪天卖出去的奶茶都能溯源到用了哪张卡片。optimize_prompts 就像请了个"试喝员 + 配方师"组合，每天试喝新品、给打分、自动帮你改进配方。ResponsesAgent 就像"标准操作手册"——按手册做的奶茶放哪家分店都是同一个味道。

**产出物**：跑完三个脚本后，你在 MLflow UI 里能看到 `agent-v1` 和 `agent-v2` 两个独立的 LoggedModel、能看到 `optimize-demo` prompt 的多版本演进历史、能用 `mlflow.pyfunc.load_model` 加载一个 ResponsesAgent 并推理。

### 你会学到什么

- **能用 `set_active_model` 把同一份代码的不同版本登记为不同的 LoggedModel**，让 trace 自动归类到对应版本
- **能用 `mlflow.genai.optimize_prompts` 让 reflection 模型自动迭代改进 prompt**，每版自动注册到 Prompt Registry
- **能用 `ResponsesAgent` 基类把自定义 LLM 应用打包成 MLflow 模型**，并兼容 OpenAI Responses API
- **能用 Models-from-code 方式（`set_model()` + 文件路径字符串）打包复杂 Agent**，避免 pickle 序列化失败
- **能搜索和对比多个 LoggedModel**（A/B 测试、生产回滚、版本溯源）

### 对应脚本清单

| 脚本 | 一句话作用 | 是否必跑 | 前置 |
|------|-----------|---------|------|
| `08a_active_model.py` | 用 `set_active_model` 把两个 Agent 版本关联到不同的 LoggedModel | ✓ 必跑 | 阶段 4（tracing） |
| `08b_prompt_optimize.py` | 注册初始 prompt 并尝试用 MetaPromptOptimizer 自动改写 | 推荐 | 阶段 6（Prompt Registry） |
| `08c_responses_agent.py` | 用 Models-from-code 方式打包 ResponsesAgent 到 Registry | 推荐 | 必跑 08a |
| `simple_qa_agent.py` | `SimpleQAAgent` 的定义（被 08c import，不是独立脚本） | — | — |

### 前置知识

- **已完成阶段 4（tracing）**：知道 `@mlflow.trace`、`mlflow.openai.autolog()` 是什么
- **已完成阶段 6（Prompts）**：知道 `register_prompt` 和 `prompts:/name/version` URI
- **API key**：需要 OpenAI 兼容服务的 API key（脚本读 `OPENAI_API_KEY`、`OPENAI_API_BASE`、`DEEPSEEK_MODEL` 三个环境变量，默认是 DeepSeek）
- **假设你懂**：Python 类继承、dict/对象互转、context manager（`with`）
- **不假设你懂**：MLflow 3 的 LoggedModel 概念、prompt 优化算法细节、OpenAI Responses API 协议

### 跑完必看（UI）

1. 启动 UI：`mlflow ui --port 5000`
2. 左侧菜单选 **Logged Models**：
   - 看到 `agent-v1` 和 `agent-v2` 两个实体
   - 点开任意一个，看 **Traces** 标签：所有用此版本的 trace 都归在这里
   - 顶部 **Compare** 按钮：勾选两个 LoggedModel，对比 trace 数量、延迟
3. 左侧菜单选 **Prompts → optimize-demo**：
   - 看 v1 → v2 → v3… 的版本演进
   - 每版带 commit message 和 template 全文
4. 左侧菜单选 **Experiments → 08_responses_agent**：
   - 点开 Run `agent-packaging`
   - **Artifacts** 标签：能看到 `MLmodel` 文件、`requirements.txt`、agent 的 Python 源码

---

## 一、核心概念：用人话讲清楚

### 1. LoggedModel —— MLflow 3 的"一等公民"

在 MLflow 2 里，模型只是 Run 的一个 artifact（产出物），附庸在某个 Run 上。这意味着如果你想"看某个模型的所有历史 trace"，会很别扭——得先找 Run，再找 Model artifact，再找关联的 trace。

MLflow 3 把模型拎出来变成**独立实体**。LoggedModel 有自己的 `model_id`（`m-xxx`）、自己的 `aliases`（`@champion`、`@challenger`）、自己的 trace 列表。它可以跨 Run、跨实验存在。

**类比**：MLflow 2 像图书馆里"每本书只能在某个书架上"；MLflow 3 像"每本书有自己的 ISBN，可以放到任何书架、被任何人引用"。

**关键 API**：

```python
# 登记/激活一个 LoggedModel（同一 name 多次调用会自动复用）
mlflow.set_active_model(name="agent-v1")

# 之后所有 trace 自动关联到 agent-v1
agent_v1("问题")

# 搜索所有 LoggedModel
mlflow.search_logged_models(experiment_ids=[exp_id])
```

### 2. set_active_model —— 隐式的"当前模型指针"

它做的事情很简单：在当前 run 上下文里设一个"当前 LoggedModel"的指针。之后这个 run 里发生的所有 trace 自动打上"我属于这个 LoggedModel"的标签。

**什么时候用**：每个 git commit → 一个 LoggedModel；A/B 测试时把流量分到不同 LoggedModel。

⚠️ **必须在 `@mlflow.trace` 装饰的函数里调用，或在 trace 上下文里调用**。否则指针设了但 trace 没认领。

### 3. MetaPromptOptimizer / GepaPromptOptimizer —— 自动改 prompt

你给一个初始 prompt、一批带正确答案的训练数据、一个评分函数。optimizer 会：
1. 让 LLM 用当前 prompt 跑数据 → 得到每个 case 的输出和分数
2. 把"分数低 + 输出"的样本丢给 reflection 模型 → 让它分析"prompt 哪里不好"
3. 根据分析改写 prompt → 注册新版本
4. 重复 2-3 直到分数不再涨

**类比**：像一个家教老师——它给学生（LLM）做卷子（train_data），看错题，让出题人（reflection 模型）改卷子（prompt），再让学生重做，直到分数提不动。

**两个 optimizer 对比**：

| Optimizer | 依赖 | 速度 | 智能程度 |
|-----------|------|------|---------|
| `MetaPromptOptimizer` | 内置（不需要额外包） | 快 | 中 |
| `GepaPromptOptimizer` | 需 `pip install gepa` | 慢 | 高 |

⚠️ **国内服务商兼容性**：GEPA 在 DeepSeek 上偶尔报 reflection 调用错误；MetaPrompt 更稳定。如果 optimize 失败，看避坑清单。

### 4. ResponsesAgent —— 兼容 OpenAI Responses API 的 Agent 基类

OpenAI 在 2025 年推出了新的 Responses API（替代 Chat Completions）。MLflow 3 的 `ResponsesAgent` 就是"你的自定义 Agent"和"标准 OpenAI Responses 格式"之间的翻译层：

- 你继承 `ResponsesAgent` 实现 `predict()` 方法
- MLflow 自动把你的请求/响应翻译成 OpenAI Responses 格式
- 别人用 OpenAI SDK 调你的服务时，完全感知不到差别

**什么时候用**：你想让你的 LLM 应用暴露成一个"标准服务"，让前端/其他服务用 OpenAI 协议直接调用，而不用关心你内部怎么实现的。

### 5. Models-from-code —— 不 pickle，改用源码

MLflow 3 之前，`mlflow.pyfunc.log_model(python_model=my_model)` 会尝试用 `pickle` 序列化你的对象。但很多对象（比如 OpenAI 客户端、网络连接、lambda）pickle 不了或加载回来会失效。

MLflow 3 的解决方案：**直接把类定义所在的 .py 文件路径传过去**，MLflow 加载时 import 这个文件，找里面的类。简单粗暴但有效。

**类比**：以前是"把家具拆了打包快递"（容易坏），现在是"把整个房间拍照给你照着装修"（更可靠）。

关键就是**文件末尾必须调用 `set_model(YourClass())`**——告诉 MLflow "这个文件里哪个对象是模型"。

---

## 二、代码模式：可复用的模板

### 模式 1：每个代码版本一个 LoggedModel

```python
# 什么时候用：每次 git commit 后跑实验，自动归属到对应版本
import mlflow

with mlflow.start_run(run_name="agent-v2-batch"):
    mlflow.set_active_model(name="agent-v2")   # 后续 trace 自动归属
    for q in test_set:
        my_agent(q)   # 这个函数被 @mlflow.trace 装饰
```

### 模式 2：自动 prompt 优化

```python
# 什么时候用：你有一批带正确答案的样本，想自动改进 prompt
import mlflow
from mlflow.genai.optimize import MetaPromptOptimizer
from mlflow.genai.scorers import Correctness
import pandas as pd

train_data = pd.DataFrame([
    {"inputs": {"question": "..."}, "expectations": {"expected_response": "..."}},
])

result = mlflow.genai.optimize_prompts(
    predict_fn=lambda q: my_llm(q),                   # 你的预测函数
    train_data=train_data,                             # 训练数据
    prompt_uris=[f"prompts:/optimize-demo/1"],         # 起点 prompt
    optimizer=MetaPromptOptimizer(reflection_model="openai:/xxx"),
    scorers=[Correctness(model="openai:/xxx")],        # 打分函数
)
```

### 模式 3：自定义 ResponsesAgent

```python
# 什么时候用：想把自定义 LLM 应用打包成 MLflow 模型并兼容 OpenAI Responses API
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from mlflow.entities.span import SpanType
import mlflow

class MyAgent(ResponsesAgent):
    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        # 把 request.input 统一转成 [{role, content}] 列表
        messages = [{"role": m.role, "content": m.content} for m in request.input]
        resp = openai_client.chat.completions.create(model="...", messages=messages)
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=resp.choices[0].message.content)],
            custom_outputs=None,
        )
```

### 模式 4：Models-from-code 打包（必须配 set_model）

```python
# my_agent.py（独立文件）
from mlflow.models import set_model

class MyAgent(ResponsesAgent):
    def predict(self, request):
        ...

# 文件末尾：告诉 MLflow 哪个对象是模型
set_model(MyAgent())

# log 时：
mlflow.pyfunc.log_model(
    python_model="path/to/my_agent.py",   # 字符串路径，不是类实例！
    name="my-agent",
    pip_requirements=["openai", "mlflow>=3.0"],
)
```

### 模式 5：加载并推理（注意 dict 不是对象）

```python
# 什么时候用：把 log 好的模型从 Registry 加载回来用
loaded = mlflow.pyfunc.load_model("models:/m-<model_id>")

# ⚠️ PyFuncModel.predict() 接收 dict-like，不是 ResponsesAgentRequest 对象
api_request = {
    "input": [{"role": "user", "content": "..."}],
    "temperature": 0.3,
}
result = loaded.predict(api_request)
```

---

## 三、实战步骤：按顺序照做

### 第 1 步：确认环境

```bash
conda activate mlflow
# 确认环境变量已设置
echo $OPENAI_API_KEY
echo $OPENAI_API_BASE
echo $DEEPSEEK_MODEL
```

### 第 2 步：跑版本追踪（08a）

```bash
cd <project-root>
python scripts/08_agents/08a_active_model.py
```

输出会显示：
- v1 三个问题 + 答案（每个一次 LLM 调用）
- v2 三个问题 + 答案（每个两次 LLM 调用：初答 + 反思）
- 跨两个 LoggedModel 的对比列表

### 第 3 步：跑 prompt 优化（08b）

```bash
python scripts/08_agents/08b_prompt_optimize.py
```

可能会两种结果：
- **成功**：看到 v1 → v2 自动改写，template 文本被改进了
- **失败**：看到 `⚠️ 优化过程失败` 的提示——这是预期内的，国内服务商常见。脚本仍然会把 v1 注册到 Registry，只是不会自动产生 v2。没关系，看下面的 08c 也能继续。

### 第 4 步：跑 ResponsesAgent 打包（08c）

```bash
python scripts/08_agents/08c_responses_agent.py
```

输出会显示：
- 直接调用 `SimpleQAAgent.predict()` 的结果
- 模型被 log 到 Registry 的 URI（`models:/m-xxx`）
- 加载回来再推理的结果

### 第 5 步：开 UI 检查

```bash
# 另开终端
mlflow ui --port 5000
```

浏览器开 `http://localhost:5000`，按上面"跑完必看"部分的路径看 Logged Models 和 Prompts。

### 第 6 步（选跑）：把模型 serve 起来

```bash
# 等 08c 跑完，会拿到 model_uri，serve 它
mlflow models serve -m models:/m-<你的model_id> -p 5001

# 另开终端，用 OpenAI 协议调用
curl http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "你好"}]}'
```

---

## 四、避坑清单

- ⚠️ **`Failed to serialize Python model`** → 改用 Models-from-code：把 agent 类放到独立 .py 文件，文件末尾调用 `set_model(YourClass())`，`log_model` 时传文件路径字符串而不是类实例。
- ⚠️ **`predict()` 返回空 text** → 检查 `request.input` 里 `msg.content` 是字符串还是 list。ResponsesAgent 的 Message 类型允许 content 是字符串或 list[ContentPart]，统一处理（见 `simple_qa_agent.py` 第 41-46 行的写法）。
- ⚠️ **PyFuncModel schema 校验失败** → `loaded.predict()` 传 dict 而不是 `ResponsesAgentRequest` 对象。PyFuncModel 的 schema 校验不认识自定义类。
- ⚠️ **优化器在 DeepSeek 上 reflection 失败** → 用 `MetaPromptOptimizer`（更稳定）；或干脆手写循环：改 prompt → `register_prompt`（v2）→ `mlflow.genai.evaluate()` → 对比 score。
- ⚠️ **`set_active_model` 没生效** → 必须在 `@mlflow.trace` 装饰的函数里调用，或在 trace 上下文里。设了指针但没 trace 跑到，归属就是空的。
- ⚠️ **08b 跑失败但脚本没崩** → 正常现象。脚本用 `try/except` 包住了 optimize 调用，失败时打印提示但不中断。v1 仍然被注册，可以手动改进 prompt 然后 `register_prompt` 升 v2。

---

## 五、小结：3-5 个 take-aways

- **LoggedModel 是 MLflow 3 的核心升级**：模型不再是 Run 的附庸，而是独立的"版本实体"，能跨 Run/实验搜索、能注册别名、能直接挂载所有相关 trace。
- **`set_active_model` 是"无侵入"的版本标注**：一行代码就能让后续所有 trace 自动归属到指定 LoggedModel，不用手动给每个 trace 打标签。
- **`optimize_prompts` 是"懒人的福音"但要选对 optimizer**：MetaPrompt 稳定够用、GEPA 强大但依赖多且国内服务商兼容性差——生产环境优先 MetaPrompt + 手写评估循环兜底。
- **ResponsesAgent + Models-from-code 是 LLM 应用上生产的标配**：前者解决"协议兼容"，后者解决"复杂对象打包"。两者配合让你的 Agent 既能被 OpenAI SDK 调用、又能避开 pickle 的坑。
- **每次部署前先在 UI 里确认 LoggedModel 状态**：看 trace 数量、看延迟分布、看别名是否设对——这三件事做完才能安心上线。