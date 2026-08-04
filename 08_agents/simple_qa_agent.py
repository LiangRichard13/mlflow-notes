"""
SimpleQAAgent - 用于 mlflow.pyfunc.log_model 的"代码形式"打包

⚠️ MLflow 3 要求 Python 模型必须从独立 .py 文件加载
   （序列化形式不支持复杂依赖如 OpenAI 客户端）

把 agent 类放到独立文件，log_model 时通过 code_paths 引入。
"""

import os
from openai import OpenAI
import mlflow
from mlflow.entities.span import SpanType
from mlflow.models import set_model
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
)


class SimpleQAAgent(ResponsesAgent):
    """一个简单的 Q&A Agent，包装 DeepSeek 调用"""

    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
        )
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    @mlflow.trace(span_type=SpanType.AGENT, name="qa_agent.predict")
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        # request.input 可能是 list[Message] 或 list[dict]，统一处理
        messages = []
        for msg in request.input:
            if hasattr(msg, "role") and hasattr(msg, "content"):
                role = msg.role
                content = msg.content
                # content 可能是字符串或 list
                if isinstance(content, list):
                    # 提取所有 text 字段
                    content = " ".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    )
                messages.append({"role": role, "content": content})
            elif isinstance(msg, dict):
                messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=200,
            temperature=0.3,
        )
        text = resp.choices[0].message.content
        output_items = [
            self.create_text_output_item(
                text=text,
                id=resp.id,
            )
        ]
        return ResponsesAgentResponse(
            output=output_items,
            custom_outputs=None,
        )


# ⚠️ Models-from-code 必须调用 set_model() 告诉 MLflow 哪个是模型类
set_model(SimpleQAAgent())