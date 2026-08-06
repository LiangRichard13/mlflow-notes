"""
阶段 6 示例 2: 别名管理 + PromptModelConfig
=========================================

目标：
  - 用 alias（production/staging/latest）管理提示词生命周期
  - 用 PromptModelConfig 绑定模型参数（model + temperature）
  - 在 app 里通过 prompts:/name@alias 加载提示词

运行：
  conda activate mlflow
  python scripts/06_prompts/06b_alias_lifecycle.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "05_tracing"))
import env_bootstrap
import mlflow
import os
from mlflow.entities.model_registry.prompt_version import PromptModelConfig


PROMPT_NAME = "customer-support-qa"


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("06_prompt_alias")

    # ============ 1. 找最新注册的提示词版本 ============
    # ⚠️ mlflow.genai.search_prompts 返回 Prompt 对象（无 version 字段）
    #    要查具体版本用 client.search_prompt_versions(name=...)
    from mlflow import MlflowClient
    client = MlflowClient()

    versions = list(client.search_prompt_versions(name=PROMPT_NAME))
    if not versions:
        raise RuntimeError("请先跑 06a_register_prompt.py")

    # 按 version 排序
    versions.sort(key=lambda v: int(v.version))
    latest_ver = int(versions[-1].version)
    print(f"找到 {len(versions)} 个版本，最新 v{latest_ver}")

    # ============ 2. 设置别名：production / staging / champion ============
    # 演示：把 v2 设 production（详细版给生产用），v3 设 staging（多消息版本待测）
    mlflow.genai.set_prompt_alias(PROMPT_NAME, alias="production", version=2)
    mlflow.genai.set_prompt_alias(PROMPT_NAME, alias="staging", version=latest_ver)
    print(f"  ✓ production → v2")
    print(f"  ✓ staging → v{latest_ver}")

    # ============ 3. 绑定 PromptModelConfig ============
    # 模型参数和提示词绑定，部署时一起加载
    config = PromptModelConfig(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        temperature=0.3,
        max_tokens=300,
    )
    mlflow.genai.set_prompt_model_config(PROMPT_NAME, version=2, model_config=config)
    print(f"  ✓ v2 model_config: {config}")

    # ============ 4. 通过别名加载提示词 ============
    print("\n" + "=" * 60)
    print("📦 通过 prompts:/name@alias 加载：")
    print("=" * 60)

    prod_prompt = mlflow.genai.load_prompt(f"prompts:/customer-support-qa@production")
    print(f"\n  production prompt:")
    print(f"    name: {prod_prompt.name}")
    print(f"    version: {prod_prompt.version}")
    print(f"    model_config: {prod_prompt.model_config}")
    print(f"    template (前 200 字符):\n    {prod_prompt.template[:200]}")

    # ============ 5. 用 Jinja2 format() 渲染 ============
    print("\n  渲染效果：")
    rendered = prod_prompt.format(
        company="Anthropic",
        agent_name="Claude",
        tier="premium",
        max_words=100,
        question="你们的 MLflow 集成收费吗？",
    )
    print(rendered[:400])

    # ============ 6. 用真实 LLM 推理（验证 production 真的可用）============
    print("\n" + "=" * 60)
    print("🤖 用 production 提示词真实推理：")
    print("=" * 60)

    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )

    # 用 v3（chat 格式）
    staging_prompt = mlflow.genai.load_prompt(f"prompts:/customer-support-qa@staging")
    messages = staging_prompt.format(
        company="DeepSeek",
        question="你们的 V4-Flash 比 V3 强多少？",
    )
    print(f"  渲染的 messages: {messages}")

    # 调用 LLM（autolog 会自动追踪）
    mlflow.openai.autolog()
    with mlflow.start_run(run_name="prod-prompt-inference"):
        resp = client.chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            messages=messages,
            max_tokens=200,
            temperature=0.3,
        )
    print(f"\n  A: {resp.choices[0].message.content[:300]}")
    print(f"  tokens: {resp.usage.total_tokens}")

    # ============ 7. 演示"热切换" ============
    print("\n" + "=" * 60)
    print("💡 演示热切换：")
    print("=" * 60)
    print("""
    场景：v2 表现一般，v3 测试通过，要切到生产

        mlflow.genai.set_prompt_alias("customer-support-qa", "production", version=3)

    一行代码，所有 prompts:/customer-support-qa@production 的应用
    下次加载都用 v3。零停机。
    """)


if __name__ == "__main__":
    main()