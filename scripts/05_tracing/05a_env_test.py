"""
阶段 5 示例 0: 环境连通性测试
==============================

先跑这个，确认 MLflow 能正确调用 DeepSeek。
不涉及追踪，只是验证 key、base、model 都对。

运行：
  conda activate mlflow
  python scripts/05_tracing/05a_env_test.py
"""

import env_bootstrap  # 自动加载 .env + 桥接变量
import os


def main():
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    print(f"  OPENAI_API_KEY:    {'已设置 (' + OPENAI_API_KEY[:4] + '...)' if (OPENAI_API_KEY := os.getenv('OPENAI_API_KEY')) else '✗ 未设置'}")
    print(f"  OPENAI_API_BASE:   {os.getenv('OPENAI_API_BASE', '未设置（将走 OpenAI 官方）')}")
    print(f"  DEEPSEEK_MODEL:    {os.getenv('DEEPSEEK_MODEL', '未设置')}")
    print()

    # 用 OpenAI SDK 发一次请求
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    print(f"调用 {model} ...")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "用一句话自我介绍"}],
        max_tokens=200,
    )
    print("\n模型响应：")
    print(f"  {resp.choices[0].message.content}")
    print(f"\n  tokens: {resp.usage.total_tokens} (prompt={resp.usage.prompt_tokens}, completion={resp.usage.completion_tokens})")
    print(f"  model: {resp.model}")
    print("\n✓ Phase 5 环境就绪！可以开始追踪了。")


if __name__ == "__main__":
    main()