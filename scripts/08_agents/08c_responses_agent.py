"""
阶段 8 示例 3: 自定义 ResponsesAgent + mlflow.pyfunc.log_model
===========================================================

目标：
  - 用"代码形式"打包 ResponsesAgent 到 MLflow（MLflow 3 推荐方式）
  - 用 mlflow.pyfunc.log_model 记录到 Registry
  - 加载模型验证推理

⚠️ ResponsesAgent 是 MLflow 3 的新基类，兼容 OpenAI Responses API。

运行：
  conda activate mlflow
  python scripts/08_agents/08c_responses_agent.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "05_tracing"))
import env_bootstrap
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _paths


import mlflow
from mlflow.types.responses import ResponsesAgentRequest
from simple_qa_agent import SimpleQAAgent


def main():
    mlflow.set_experiment("08_responses_agent")
    mlflow.openai.autolog()

    # ============ 1. 直接用 agent 跑一次 ============
    print("=" * 60)
    print("🤖 跑 SimpleQAAgent.predict()：")
    print("=" * 60)

    agent = SimpleQAAgent()
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "用一句话解释什么是 MLflow 3 的 LoggedModel。"}],
    )

    with mlflow.start_run(run_name="agent-direct-test") as run:
        response = agent.predict(request)
        first = response.output[0]
        txt = first.get("text", "") if isinstance(first, dict) else str(first)
        print(f"  Q: 用一句话解释什么是 MLflow 3 的 LoggedModel。")
        print(f"  A: {txt[:200]}")

    # ============ 2. 用"Models-from-code"方式打包（MLflow 3 推荐）============
    # ⚠️ 关键：python_model 传文件路径字符串，不是类实例！
    #    MLflow 加载时会 import 这个文件，找到 ResponsesAgent 子类
    print("\n" + "=" * 60)
    print("📦 用 Models-from-code 方式打包 Agent：")
    print("=" * 60)

    agent_path = str(Path(__file__).parent / "simple_qa_agent.py")
    print(f"  agent code: {agent_path}")

    with mlflow.start_run(run_name="agent-packaging") as run:
        model_info = mlflow.pyfunc.log_model(
            python_model=agent_path,   # ← 文件路径！
            name="qa-agent",
            input_example=request,
            pip_requirements=["openai", "mlflow>=3.0"],
        )
        print(f"  ✓ 模型 URI: {model_info.model_uri}")
        print(f"  ✓ model_id: {model_info.model_id}")

    # ============ 3. 加载并验证 ============
    # ⚠️ PyFuncModel.predict() 接收 dict-like 输入（不是 ResponsesAgentRequest 对象）
    print("\n" + "=" * 60)
    print("🔄 从 URI 加载并推理：")
    print("=" * 60)

    loaded_agent = mlflow.pyfunc.load_model(model_info.model_uri)
    print(f"  加载类型: {type(loaded_agent).__name__}")

    # 把 ResponsesAgentRequest 转为 Responses API 风格 dict
    api_request = {
        "input": [{"role": "user", "content": "用一句话说 MLflow 3 LoggedModel 是什么。"}],
        "temperature": 0.3,
    }
    try:
        response2 = loaded_agent.predict(api_request)
        # 提取文本
        if isinstance(response2, dict) and "output" in response2:
            first = response2["output"][0]
            if isinstance(first, dict):
                txt = first.get("text", "")
                # text 可能在 content 列表里
                if not txt and "content" in first:
                    content = first["content"]
                    if isinstance(content, list):
                        txt = content[0].get("text", "") if content else ""
        else:
            txt = str(response2)[:200]
        print(f"  A: {txt[:200]}")
    except Exception as e:
        print(f"  ⚠️ 加载模型推理失败（pyfunc 签名校验）: {type(e).__name__}")
        print(f"     {str(e)[:150]}")
        print(f"  💡 这个错误说明 PyFuncModel.predict 对 ResponsesAgent 包装还在演进")
        print(f"     实际部署请用 mlflow models serve + OpenAI 兼容 SDK 调用")

    print("\n" + "=" * 60)
    print("💡 关键点：")
    print("=" * 60)
    print("""
    1. MLflow 3 要求 python_model 必须是可 import 的类
       - 把 SimpleQAAgent 放在 simple_qa_agent.py
       - log_model 时 code_paths=[dirname] 让加载时能找到

    2. ResponsesAgent 自动兼容 OpenAI Responses API：
       - 调用方可以发 OpenAI 格式请求
       - 你的 predict() 返回 ResponsesAgentResponse
       - 自动序列化为 OpenAI Responses 格式

    3. 部署：
       mlflow models serve -m <model_uri> -p 5001
       或 pyfunc.load_model 后用 Python 直接调用

    4. 更高级：Agent Server（>=3.6.0）
       from mlflow.genai.agent_server import invoke, stream
    """)


if __name__ == "__main__":
    main()