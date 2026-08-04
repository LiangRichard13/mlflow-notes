# 阶段 6 学习笔记：Prompt Registry 与框架 Flavors

> 对应脚本：`06_prompts/06a_register_prompt.py`、`06b_alias_lifecycle.py`、`06c_langchain.py`

## 一、注册提示词

```python
import mlflow

# 文本模板（带 Jinja2）
v1 = mlflow.genai.register_prompt(
    name="customer-support-qa",
    template="你是 {{ company }} 的客服助手。回答：{{ question }}",
    commit_message="v1: 简洁版",
    tags={"style": "concise", "author": "alice"},
)

# 消息列表（OpenAI Chat 格式）
v2 = mlflow.genai.register_prompt(
    name="customer-support-qa",
    template=[
        {"role": "system", "content": "你是 {{ company }} 的客服。"},
        {"role": "user", "content": "{{ question }}"},
    ],
    commit_message="v2: 改用 chat 格式",
)
```

Jinja2 模板支持：变量、条件、循环、过滤器。

## 二、查所有版本

⚠️ **关键陷阱**：`mlflow.genai.search_prompts()` 返回 `Prompt` 对象（没 version 字段）。
要查具体版本用 `client.search_prompt_versions(name=...)`。

```python
from mlflow import MlflowClient
client = MlflowClient()

for ver in client.search_prompt_versions(name="customer-support-qa"):
    print(f"v{ver.version}: {ver.commit_message}")
```

## 三、别名管理（生命周期）

```python
# 把 v2 设 production，v3 设 staging
mlflow.genai.set_prompt_alias("customer-support-qa", "production", version=2)
mlflow.genai.set_prompt_alias("customer-support-qa", "staging", version=3)

# 通过别名加载（最常用！）
prompt = mlflow.genai.load_prompt("prompts:/customer-support-qa@production")
print(prompt.template)   # 原始模板
formatted = prompt.format(company="DeepSeek", question="什么是 V4？")  # 渲染

# 删除别名
mlflow.genai.delete_prompt_alias("customer-support-qa", "staging")
```

**别名模型**（替代已废弃的 stage）：
- `@production`：线上用
- `@staging`：灰度测试
- `@champion`：当前冠军（对比新版本）
- `@challenger`：新版本候选

切换 production 一行代码，所有引用 `prompts:/name@production` 的应用下次加载都自动用新版——**零停机回滚**。

## 四、PromptModelConfig（绑模型参数）

```python
from mlflow.entities.model_registry.prompt_version import PromptModelConfig

config = PromptModelConfig(
    model_name="deepseek-v4-flash",
    temperature=0.3,
    max_tokens=300,
)
mlflow.genai.set_prompt_model_config("customer-support-qa", version=2, model_config=config)
```

部署时模型参数跟着 prompt 一起加载，避免"prompt 改了但模型参数没改"。

## 五、ResponseFormat（结构化输出）

```python
from pydantic import BaseModel

class Answer(BaseModel):
    reply: str
    confidence: float

mlflow.genai.register_prompt(
    name="structured-qa",
    template="...",
    response_format=Answer,   # 强约束 LLM 返回 JSON
)
```

## 六、LangChain autolog

```python
mlflow.langchain.autolog()    # 一行开启所有 LangChain 调用追踪

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([...])
llm = ChatOpenAI(model="...", openai_api_base="...")  # DeepSeek 兼容 OpenAI

chain = prompt | llm
resp = chain.invoke({"q": "..."})   # 自动产生 trace
```

把 MLflow Registry 的 prompt 喂给 LangChain：

```python
mlflow_prompt = mlflow.genai.load_prompt("prompts:/my-prompt@production")
lc_messages = [(m["role"], m["content"]) for m in mlflow_prompt.template]
lc_prompt = ChatPromptTemplate.from_messages(lc_messages)
```

## 七、完整流程：prompt 版本化 → 服务

```
1. 在 .env 写 DEEPSEEK_MODEL 等参数
2. mlflow.genai.register_prompt(...) 注册（v1, v2, v3...）
3. mlflow.genai.set_prompt_alias(name, "production", version=N)
4. 应用启动时：mlflow.genai.load_prompt("prompts:/name@production")
5. 测试新版本：set_prompt_alias(..., version=M) 切换
6. 出问题：set_prompt_alias(..., version=N) 一行回滚
```