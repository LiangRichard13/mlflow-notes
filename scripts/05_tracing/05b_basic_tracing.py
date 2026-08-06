"""
阶段 5 示例 1: mlflow.openai.autolog - 一行开启追踪
=================================================

目标：
  - 一行代码 mlflow.openai.autolog() 开启所有 OpenAI 调用的追踪
  - 在 UI 看每次 LLM 调用的输入/输出/token/延迟
  - 理解 Trace = 一棵 Span 树

运行：
  conda activate mlflow
  python scripts/05_tracing/05b_basic_tracing.py
  # 然后另开 mlflow ui 看 http://localhost:5000
"""

import env_bootstrap
import mlflow
import os
from openai import OpenAI


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("05_basic_tracing")

    # ============ 关键一行：开启 autolog ============
    # 之后所有 OpenAI client.chat.completions.create() 调用都会被自动追踪
    mlflow.openai.autolog()

    # 拿当前模型（从 .env 读）
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )

    # ============ 单次调用追踪 ============
    print("单次调用追踪：")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是 DeepSeek，简洁回答。"},
            {"role": "user", "content": "用一句话解释 MLflow 是干什么的？"},
        ],
        max_tokens=100,
        temperature=0.3,
    )
    print(f"  Q: MLflow 是干什么的？")
    print(f"  A: {resp.choices[0].message.content}")
    print(f"  tokens: {resp.usage.total_tokens}")
    print()

    # ============ 多轮对话追踪（同一 trace 包含多次 LLM 调用）============
    print("多轮对话追踪（3 次 LLM 调用构成一棵 Span 树）：")

    # 显式 start_run 把多次调用归到同一个 Run 里
    with mlflow.start_run(run_name="multi-turn-chat") as run:
        # 手动设个 trace tag（autolog 默认会按 run 归类）
        mlflow.set_tag("chat_topic", "MLflow 入门")

        messages = [
            {"role": "system", "content": "你是 DeepSeek，回答简短。"},
            {"role": "user", "content": "MLflow 的四大核心组件是什么？"},
        ]

        # 第一轮
        resp1 = client.chat.completions.create(model=model, messages=messages, max_tokens=150)
        answer1 = resp1.choices[0].message.content
        messages.append({"role": "assistant", "content": answer1})
        messages.append({"role": "user", "content": "第一个能再详细说一下吗？"})
        print(f"  Q1: MLflow 的四大核心组件？")
        print(f"  A1: {answer1[:80]}...")

        # 第二轮
        resp2 = client.chat.completions.create(model=model, messages=messages, max_tokens=200)
        answer2 = resp2.choices[0].message.content
        messages.append({"role": "assistant", "content": answer2})
        messages.append({"role": "user", "content": "能给个 mlflow.sklearn.autolog() 的例子吗？"})
        print(f"  Q2: 第一个详细说说")
        print(f"  A2: {answer2[:80]}...")

        # 第三轮
        resp3 = client.chat.completions.create(model=model, messages=messages, max_tokens=300)
        answer3 = resp3.choices[0].message.content
        print(f"  Q3: autolog 例子")
        print(f"  A3: {answer3[:80]}...")

        # 记录 Run 级指标
        total_tokens = resp1.usage.total_tokens + resp2.usage.total_tokens + resp3.usage.total_tokens
        mlflow.log_metric("total_tokens", total_tokens)
        mlflow.log_metric("num_turns", 3)

    print(f"\n✓ Run 已记录: {run.info.run_id[:8]}")
    print(f"  共 {total_tokens} tokens，3 次 LLM 调用")

    print("\n" + "=" * 60)
    print("📊 在 UI 看：")
    print("=" * 60)
    print("""
    1. mlflow ui --port 5000
    2. 选 experiment "05_basic_tracing"
    3. 点开 multi-turn-chat 这个 Run
    4. 看 "Traces" 标签：能看到 3 个 Trace（每次 chat.completions 一个）
    5. 点开任一 Trace，看 Span 树：
       - root span: chat completions 调用
         - attributes: model, temperature, max_tokens
         - 输入 messages（user/system/assistant）
         - 输出 content
         - usage: prompt_tokens, completion_tokens, total_tokens
         - latency: 实际耗时
    """)


if __name__ == "__main__":
    main()