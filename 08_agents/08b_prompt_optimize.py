"""
阶段 8 示例 2: mlflow.genai.optimize_prompts - 自动优化提示词
============================================================

目标：
  - 演示 mlflow.genai.optimize_prompts() API 的用法
  - 理解 optimizer（GEPA / MetaPrompt）如何自动改进 prompt
  - ⚠️ 实际跑优化可能因服务商兼容性而失败（见下方说明）

运行：
  conda activate mlflow
  python 07_agents/08b_prompt_optimize.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "04_tracing"))
import env_bootstrap

import mlflow
import os
import pandas as pd
from openai import OpenAI


# 初始 prompt（故意简单，演示 optimizer 如何改进）
INITIAL_PROMPT = (
    "你是助手。回答：{{ question }}"
)


def predict_with_prompt(prompt_text: str, question: str) -> str:
    """用 prompt + question 调 LLM"""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )
    full_prompt = prompt_text.replace("{{ question }}", question)
    resp = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        messages=[{"role": "user", "content": full_prompt}],
        max_tokens=150,
        temperature=0.3,
    )
    return resp.choices[0].message.content


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("08_prompt_optimize")
    mlflow.openai.autolog()

    # ============ 1. 注册初始 prompt ============
    print("注册初始 prompt...")
    v1 = mlflow.genai.register_prompt(
        name="optimize-demo",
        template=INITIAL_PROMPT,
        commit_message="v1: 朴素初始版",
    )
    print(f"  ✓ {v1.name} v{v1.version}")
    print(f"    template: {v1.template}")

    # ============ 2. 演示 MetaPromptOptimizer 的 API ============
    # MetaPromptOptimizer 更轻量，只用 reflection 模型改写 prompt
    # 比 GEPA 简单，但不需要 GEPA 库
    from mlflow.genai.optimize import MetaPromptOptimizer
    from mlflow.genai.scorers import Correctness

    judge_model = f"openai:/{os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-flash')}"

    train_data = pd.DataFrame([
        {"inputs": {"question": "什么是 MLflow？"},
         "expectations": {"expected_response": "MLflow 是开源 ML 生命周期管理平台"}},
        {"inputs": {"question": "MLflow 3 的 LoggedModel？"},
         "expectations": {"expected_response": "LoggedModel 是 MLflow 3 的一等公民"}},
    ])

    def predict_fn(question: str) -> str:
        from mlflow import MlflowClient
        client = MlflowClient()
        versions = list(client.search_prompt_versions(name="optimize-demo"))
        latest = sorted(versions, key=lambda v: int(v.version))[-1]
        return predict_with_prompt(latest.template, question)

    print("\n" + "=" * 60)
    print("🔄 跑 MetaPromptOptimizer（reflection + 改写）：")
    print("=" * 60)

    try:
        result = mlflow.genai.optimize_prompts(
            predict_fn=predict_fn,
            train_data=train_data,
            prompt_uris=[f"prompts:/optimize-demo/{v1.version}"],
            optimizer=MetaPromptOptimizer(
                reflection_model=judge_model,
            ),
            scorers=[Correctness(model=judge_model)],
        )
        print(f"\n✓ 优化完成")
        if hasattr(result, "optimized_prompts"):
            print(f"  优化后 prompt: {result.optimized_prompts[0].template[:200]}")
    except Exception as e:
        print(f"\n⚠️ 优化过程失败（这在非 OpenAI 服务商上常见）:")
        print(f"   {type(e).__name__}: {str(e)[:200]}")
        print(f"\n💡 但版本已经被注册到 Registry，可以手动改进 prompt 然后注册新版本")

    # ============ 3. 看版本演进 ============
    from mlflow import MlflowClient
    client = MlflowClient()
    versions = sorted(client.search_prompt_versions(name="optimize-demo"), key=lambda v: int(v.version))
    print(f"\n📋 Prompt 版本历史：")
    for v in versions:
        print(f"  v{v.version}: {v.commit_message}")
        print(f"    {v.template[:100]}")

    print("\n" + "=" * 60)
    print("💡 提示词优化的现实方案：")
    print("=" * 60)
    print("""
    1. 调 mlflow.genai.optimize_prompts() 让 optimizer 自动迭代
       - 优点：完全自动，可追溯
       - 缺点：依赖 optimizer 库与服务商兼容性

    2. 手写循环 + 评估：
       - 改 prompt → register_prompt（v2）
       - mlflow.genai.evaluate() 评估
       - 对比 v1 和 v2 的 score
       - 选赢家 set_prompt_alias("production", version=2)

    3. 集成到 CI/CD：
       - 每次 PR 自动跑 eval
       - 如果 score 提升就允许合并

    推荐先用方案 2（更可控），熟悉后再用方案 1（更省事）。
    """)


if __name__ == "__main__":
    main()