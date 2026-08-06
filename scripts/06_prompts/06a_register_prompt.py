"""
阶段 6 示例 1: 注册版本化提示词 + Jinja2 模板
============================================

目标：
  - mlflow.genai.register_prompt() 注册带 Jinja2 模板的提示词
  - 同名注册会创建新版本（v1, v2, v3...）
  - 用 tags / commit_message 标注每版用途

运行：
  conda activate mlflow
  python scripts/06_prompts/06a_register_prompt.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "05_tracing"))
import env_bootstrap
import mlflow


PROMPT_NAME = "customer-support-qa"


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("06_prompt_registry")

    # ============ 1. 注册 v1：简洁版 ============
    print("注册 v1（简洁版）...")
    v1 = mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=(
            "你是 {{ company }} 的客服助手。请简洁回答用户问题。\n\n"
            "用户问题：{{ question }}"
        ),
        commit_message="v1: 简洁版，限定公司角色",
        tags={
            "style": "concise",
            "author": "alice",
        },
    )
    print(f"  ✓ {v1.name} v{v1.version}")
    print(f"    URI: {v1.uri}")
    print(f"    variables: {v1.variables}")   # Jinja2 自动识别的变量

    # ============ 2. 注册 v2：详细版（带上下文、约束）============
    print("\n注册 v2（详细版，带约束）...")
    v2 = mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=(
            "你是 {{ company }} 的客服助手 {{ agent_name }}。\n"
            "{% if tier == 'premium' %}\n"
            "  这是 VIP 客户，请优先处理，提供个性化建议。\n"
            "{% endif %}\n\n"
            "约束：\n"
            "  - 只用事实信息回答，不要编造\n"
            "  - 回答不超过 {{ max_words }} 字\n"
            "  - 末尾加 [source: KB-id]\n\n"
            "用户问题：{{ question }}"
        ),
        commit_message="v2: 加 tier 分支 + 字数约束 + 引用格式",
        tags={
            "style": "detailed",
            "author": "bob",
            "uses_jinja": "true",
        },
    )
    print(f"  ✓ v{v2.version}")
    print(f"    variables: {v2.variables}")   # 注意 Jinja2 的 if 不会出现在变量里

    # ============ 3. 注册 v3：使用 chat 消息格式 ============
    print("\n注册 v3（多消息格式）...")
    v3 = mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=[
            {"role": "system", "content": "你是 {{ company }} 的客服助手。"},
            {"role": "user", "content": "{{ question }}"},
        ],
        commit_message="v3: 改用 OpenAI Chat Completions 消息格式",
        tags={
            "format": "chat",
            "style": "balanced",
        },
    )
    print(f"  ✓ v{v3.version}")
    print(f"    is_text_prompt: {v3.is_text_prompt}")   # False，因为是消息列表

    # ============ 4. 列出所有版本 ============
    # ⚠️ search_prompts 返回 Prompt 对象（不带版本号），要查版本用 client.search_prompt_versions
    from mlflow import MlflowClient
    client = MlflowClient()
    print("\n" + "=" * 60)
    print(f"📋 {PROMPT_NAME} 全部版本：")
    print("=" * 60)
    for ver in client.search_prompt_versions(name=PROMPT_NAME):
        print(f"  v{ver.version}: commit='{ver.commit_message}'")
        print(f"    tags: {ver.tags}")
        print(f"    uri: {ver.uri}")

    print("\n下一步：06b_alias_lifecycle.py")


if __name__ == "__main__":
    main()