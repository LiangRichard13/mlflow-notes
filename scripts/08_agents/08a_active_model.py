"""
阶段 8 示例 1: set_active_model - 版本化追踪
==========================================

目标：
  - mlflow.set_active_model() 把后续追踪关联到特定 LoggedModel
  - 模拟"每个代码版本一个 LoggedModel"
  - 在 UI 中按 LoggedModel 维度查看 trace

运行：
  conda activate mlflow
  python scripts/08_agents/08a_active_model.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "05_tracing"))
import env_bootstrap

import mlflow
import os
from openai import OpenAI
from mlflow.entities.span import SpanType


# 模拟两个不同版本的 Agent
@mlflow.trace(span_type=SpanType.AGENT, name="qa_agent_v1")
def agent_v1(question: str) -> str:
    """v1: 简单的单步 LLM 调用"""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )
    resp = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        messages=[{"role": "user", "content": question}],
        max_tokens=100,
    )
    return resp.choices[0].message.content


@mlflow.trace(span_type=SpanType.AGENT, name="qa_agent_v2")
def agent_v2(question: str) -> str:
    """v2: 加了反思步骤（先回答，再让 LLM 自我审查）"""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )
    # Step 1: 初答
    first = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        messages=[{"role": "user", "content": question}],
        max_tokens=150,
    ).choices[0].message.content

    # Step 2: 反思改进
    refined = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": first},
            {"role": "user", "content": "上面的回答准确吗？请给出最终改进版。"},
        ],
        max_tokens=150,
    ).choices[0].message.content
    return refined


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("08_version_tracking")
    mlflow.openai.autolog()

    questions = [
        "MLflow 3 最大的变化是什么？",
        "Prompt Registry 怎么用别名？",
        "Trace 和 Span 的关系？",
    ]

    # ============ 1. 跑 v1，关联到 LoggedModel 'agent-v1' ============
    print("=" * 60)
    print("📦 v1 三个问题（关联到 LoggedModel 'agent-v1'）：")
    print("=" * 60)

    with mlflow.start_run(run_name="agent-v1-batch"):
        mlflow.set_active_model(name="agent-v1")   # ← 关键
        for q in questions:
            ans = agent_v1(q)
            print(f"\n  Q: {q}")
            print(f"  A: {ans[:80]}...")

    # ============ 2. 跑 v2，关联到 LoggedModel 'agent-v2' ============
    print("\n" + "=" * 60)
    print("📦 v2 三个问题（关联到 LoggedModel 'agent-v2'）：")
    print("=" * 60)

    with mlflow.start_run(run_name="agent-v2-batch"):
        mlflow.set_active_model(name="agent-v2")   # ← 关键
        for q in questions:
            ans = agent_v2(q)
            print(f"\n  Q: {q}")
            print(f"  A: {ans[:80]}...")

    # ============ 3. 搜索 LoggedModel 对比 ============
    print("\n" + "=" * 60)
    print("🔍 跨两个 LoggedModel 对比：")
    print("=" * 60)

    exp_id = mlflow.get_experiment_by_name("08_version_tracking").experiment_id
    logged_models = mlflow.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[{"field_name": "creation_timestamp", "ascending": False}],
        output_format="list",
    )

    for lm in logged_models:
        # 找属于这个 LoggedModel 的 trace
        traces = mlflow.search_traces(
            experiment_ids=[exp_id],
            filter_string=f"metadata.`mlflow.sourceRun` LIKE '%'",
            max_results=100,
        )
        # 过滤出属于当前 LoggedModel 的 trace
        if lm.name.startswith("agent-"):
            matching = [t for t in traces.iterrows()
                       if lm.model_id in (t[1].get("trace_metadata", {}).get("mlflow.sourceRun", "") or "")]
            print(f"  LoggedModel: {lm.name} (id={lm.model_id[:12]})")

    print("\n" + "=" * 60)
    print("📊 在 UI 中看：")
    print("=" * 60)
    print("""
    1. 左侧菜单选 'Logged Models' → 看到 agent-v1 和 agent-v2
    2. 点任一个 LoggedModel → 看它的 Traces 标签页
    3. 在 'Compare' 选两个 LoggedModel → 看 trace 数量、延迟、token 差异

    💡 实战用法：
       - 每个 git commit 自动关联一个 LoggedModel
       - A/B 测试时按 LoggedModel 切片分析
       - 出问题时能精确定位"哪个版本的哪次调用挂了"
    """)


if __name__ == "__main__":
    main()