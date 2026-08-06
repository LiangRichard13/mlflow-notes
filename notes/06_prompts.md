# 阶段 6 学习笔记：Prompt Registry 与框架 Flavors

> 对应脚本：`scripts/06_prompts/06a_register_prompt.py`、`06b_alias_lifecycle.py`、`06c_langchain.py`

## 🎯 这篇笔记做什么

前 5 个阶段我们都在跟踪「模型」和「数据」，但 LLM 应用最该被管起来的其实是**提示词（Prompt）**——一行提示词的改动就可能让生产环境崩溃。MLflow 的 **Prompt Registry** 就是专门管提示词的 Git：每次改提示词都登记一个新版本（v1、v2、v3...），你可以给不同版本贴 `production`、`staging` 别名，**应用永远只通过别名加载**，要切版本、回滚都是一行代码的事。

类比：写代码会用 Git + main/dev 分支管理；写提示词就该用 Prompt Registry + `@production`/`@staging` 别名。Prompt Registry 就是「提示词界的 GitHub」。

产出物：跑完三个脚本后，你会在 MLflow 里拥有：
- 一个叫 `customer-support-qa` 的提示词，3 个版本（v1/v2/v3），分别带不同模板和 tags
- `production` 别名指向 v2、`staging` 别名指向 v3
- 一个真实 LLM 推理的 Trace，证明 `@production` 真的能用
- 一段 LangChain 代码，每次调用都自动留 trace

### 你会学到什么

- 能用 `mlflow.genai.register_prompt()` 注册带 Jinja2 的版本化提示词
- 能用 alias（production/staging）管理提示词生命周期并做零停机切换
- 能用 `PromptModelConfig` 把模型参数和提示词绑一起
- 能用 `mlflow.langchain.autolog()` 一行开启 LangChain 全链路追踪
- 能把 MLflow Registry 里的 prompt 喂给 LangChain 的 `ChatPromptTemplate`

### 对应脚本清单

| 脚本 | 一句话作用 | 是否必跑 | 前置 |
|------|-----------|---------|------|
| `06a_register_prompt.py` | 注册 v1/v2/v3 三个版本的 `customer-support-qa` 提示词 | ✓ 必跑 | 无 |
| `06b_alias_lifecycle.py` | 设 production/staging alias + 真实推理 | ✓ 必跑 | 跑过 06a |
| `06c_langchain.py` | LangChain autolog + 从 Registry 加载 prompt | 推荐 | 跑过 06a、06b |

### 前置知识

- 已完成 Phase 1–5（理解 Run、Experiment、Artifact）
- 已安装：`mlflow`、`openai`、`langchain-openai`、`langchain-core`、`pydantic`
- **需要 API key**（`OPENAI_API_KEY` / `OPENAI_API_BASE` / `DEEPSEEK_MODEL`，写在 `.env` 里，参考 Phase 4 的 `env_bootstrap.py`）
- 假设你懂：Jinja2 基本语法（`{{ var }}`、`{% if %}`），没用过也不影响跑通

### 跑完必看（UI）

1. 启动 UI：`mlflow ui --port 5000`
2. **左侧栏 → Prompts**：
   - 点开 `customer-support-qa`，看 Versions 标签下的 v1 / v2 / v3
   - 看每个版本的 commit_message 和 tags（style、author、format）
   - 看 Aliases：`production` → v2，`staging` → v3
3. **左侧栏 → Experiments → `06_prompt_registry`**：看三个 register 操作的 Run 记录
4. **左侧栏 → Experiments → `06_prompt_alias` → Run `prod-prompt-inference`**：
   - Traces 标签：能看到真实 LLM 调用，prompt 内容、模型参数、回复都在里面
5. **左侧栏 → Experiments → `06_langchain` → Run `langchain-rag-demo`**：
   - Traces 标签：展开看 span 树，能看到 chain → prompt 渲染 → LLM 的完整链路
6. **左侧栏 → Experiments → `06_langchain` → Run `kb-rag`**：
   - 看 `kb_rag_chain` 的 span 树：`kb_search (RETRIEVER)` → `ChatOpenAI (LLM)`

---

## 一、核心概念：用人话讲清楚

### 1. Prompt Version（提示词版本）

每调用一次 `mlflow.genai.register_prompt(name=...)`，如果 `name` 已经存在，就自动创建一个新版本（v1 → v2 → v3 ...）。**模板一旦注册就不可改**，要改就注册新版。类比：你提交代码后想改，得开新 commit，不能改历史 commit。

### 2. Alias（别名）= 指针

别名不是版本，是**指向某个版本的指针**。你可以随时把 `@production` 从 v2 改指 v3，**所有通过 `prompts:/name@production` 加载提示词的应用，下次启动就自动用新版**。这是零停机发布和一秒回滚的关键。常见别名约定：
- `@production`：线上跑的稳定版
- `@staging`：测试版，准备晋升 production
- `@champion` / `@challenger`：A/B 测试用

### 3. URI 语法 `prompts:/name@alias`

`mlflow.genai.load_prompt()` 支持两种写法：
- `prompts:/customer-support-qa/2` → 加载指定版本（写死，不灵活）
- `prompts:/customer-support-qa@production` → 加载 alias 指向的版本（推荐，永远跟最新发布走）

### 4. Jinja2 模板

`register_prompt` 的 `template` 参数支持 Jinja2：
- 变量：`{{ question }}`
- 条件：`{% if tier == 'premium' %}...{% endif %}`
- 循环：`{% for item in items %}...{% endfor %}`

调用 `prompt.format(var1=..., var2=...)` 就能把变量填进去。`prompt.variables` 会自动列出模板里识别到的所有变量名（注意：if/for 里的变量不会出现在 `variables` 里）。

### 5. PromptModelConfig（模型参数打包）

把模型名、温度、max_tokens 这些参数和某个 prompt 版本绑一起，存到 Registry 里。部署时通过 `prompt.model_config` 一次性拿到，避免「prompt 改了但 temperature 没改」的坑。

### 6. LangChain autolog

`mlflow.langchain.autolog()` 一行开启后，所有 LangChain 的 chain、LLM 调用都会**自动**写 trace 到当前 Run。不用手写 `@mlflow.trace`，链路上每一步（retriever、prompt template、LLM）都会被记下来。

---

## 二、代码模式：可复用的模板

### 模式 1：注册文本 prompt

```python
v1 = mlflow.genai.register_prompt(
    name="customer-support-qa",
    template="你是 {{ company }} 的客服助手。问题：{{ question }}",
    commit_message="v1: 简洁版",
    tags={"style": "concise", "author": "alice"},
)
```
**什么时候用**：单一字符串模板，最简单的 LLM 场景。

### 模式 2：注册 chat 格式 prompt

```python
v3 = mlflow.genai.register_prompt(
    name="customer-support-qa",
    template=[
        {"role": "system", "content": "你是 {{ company }} 的客服。"},
        {"role": "user", "content": "{{ question }}"},
    ],
    commit_message="v3: 改用 chat 格式",
)
```
**什么时候用**：需要 system message、few-shot 示例、多轮对话结构。

### 模式 3：通过 alias 加载并渲染

```python
prompt = mlflow.genai.load_prompt("prompts:/customer-support-qa@production")
messages = prompt.format(company="DeepSeek", question="什么是 V4？")
# messages 现在可以直接喂给 OpenAI/Anthropic SDK 的 chat.completions
```
**什么时候用**：应用启动时加载生产 prompt，然后渲染变量。这是**最常用**的模式。

### 模式 4：设 alias / 删 alias

```python
mlflow.genai.set_prompt_alias("customer-support-qa", "production", version=3)
mlflow.genai.delete_prompt_alias("customer-support-qa", "staging")
```
**什么时候用**：发布新版（指向新版本）、下线别名。

### 模式 5：绑定 PromptModelConfig

```python
from mlflow.entities.model_registry.prompt_version import PromptModelConfig

config = PromptModelConfig(model_name="deepseek-v4-flash", temperature=0.3, max_tokens=300)
mlflow.genai.set_prompt_model_config("customer-support-qa", version=2, model_config=config)
```
**什么时候用**：想确保「prompt 改了什么模型、什么参数」跟着一起走。

### 模式 6：LangChain autolog + 从 Registry 加载

```python
mlflow.langchain.autolog()

mlflow_prompt = mlflow.genai.load_prompt("prompts:/customer-support-qa@staging")
lc_messages = [(m["role"], m["content"]) for m in mlflow_prompt.template]
prompt = ChatPromptTemplate.from_messages(lc_messages)

chain = prompt | ChatOpenAI(model="deepseek-v4-flash", ...)
resp = chain.invoke({"company": "X", "question": "Y"})  # 自动 trace
```
**什么时候用**：已有 LangChain 项目想接入 MLflow Registry 做提示词管理。

---

## 三、实战步骤：按顺序照做

### Step 0：环境准备

```bash
conda activate mlflow
# 确保 .env 里有 OPENAI_API_KEY / OPENAI_API_BASE / DEEPSEEK_MODEL
# 启动 MLflow（用 sqlite 后端）
mlflow ui --port 5000   # 另开终端
```

### Step 1：注册三个版本

```bash
cd <project-root>
python scripts/06_prompts/06a_register_prompt.py
```

预期输出：`✓ customer-support-qa v1 / v2 / v3`，并列出全部版本。

### Step 2：设 alias + 真实推理

```bash
python scripts/06_prompts/06b_alias_lifecycle.py
```

预期输出：
- `production → v2`，`staging → v3`
- 加载 production prompt 渲染效果
- 真实 LLM 推理返回（A: ... tokens: ...）

### Step 3：LangChain autolog

```bash
python scripts/06_prompts/06c_langchain.py
```

预期输出：
- 3 次 chain 调用，每次都被 autolog 记录
- 列出 trace 数量和耗时
- `kb-rag` Run 跑完，可去 UI 看 span 树

### Step 4：UI 验证（参考「跑完必看」清单）

按上面 6 个步骤检查 Prompts 和 Experiments。

---

## 四、避坑清单

- ⚠️ **`mlflow.genai.search_prompts()` 不返回版本号** → 它返回 `Prompt` 对象（只有 name）。要查具体版本用 `client.search_prompt_versions(name=...)`，里面有 `version`、`commit_message`、`tags`。

- ⚠️ **`register_prompt` 的 `name` 已存在不会覆盖**，而是创建新版本 → 这是设计如此（要的就是版本化），别以为是 bug。要「重置」得手动删旧版本或在脚本里加判断。

- ⚠️ **`prompt.variables` 不包含 `{% if %}` / `{% for %}` 里的变量** → 它只识别 `{{ var }}` 形式的变量。如果你的模板里有条件分支，分支里的变量调用 `format()` 时也得传进去。

- ⚠️ **跑 `06b` 前必须先跑 `06a`** → 否则找不到 `customer-support-qa` 这个 prompt，脚本会 `RuntimeError`。

- ⚠️ **LangChain autolog 必须在导入 LangChain 对象前调用** → 虽然大多数情况后调也能用，但稳妥起见在 `import langchain` 之后第一时间 `mlflow.langchain.autolog()`。

- ⚠️ **chat 格式 prompt 转 LangChain 时是 `[(role, content)]` 不是 `[{"role":..., "content":...}]`** → 看模式 6 的 `lc_messages = [(m["role"], m["content"]) for m in mlflow_prompt.template]`，别忘了转换。

---

## 五、小结：5 个 take-aways

- **Prompt Registry = 提示词界的 Git**，每次改动都是新版本，不可改历史。
- **永远通过 `prompts:/name@alias` 加载 prompt**，别写死版本号，发布/回滚就是一行 `set_prompt_alias`。
- **`@production` / `@staging` 是约定俗成的别名**，符合团队直觉，A/B 测试用 `@champion` / `@challenger`。
- **`PromptModelConfig` 解决「prompt 改了但参数忘改」的痛点**，把模型参数也存进 Registry。
- **`mlflow.langchain.autolog()` 一行接入 LangChain 追踪**，再配合 `prompts:/name@alias` 加载，就是生产级 LLM 应用的标配组合。
