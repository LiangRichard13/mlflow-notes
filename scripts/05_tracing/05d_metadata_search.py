"""
阶段 5 示例 3: Trace 元数据 + search_traces
==========================================

目标：
  - 给当前 trace 加 user / session / 业务元数据
  - 用 mlflow.search_traces() 程序化搜索
  - 理解"按 user 看历史会话" / "按 session 聚合" 等常见分析

运行：
  conda activate mlflow
  python scripts/05_tracing/05d_metadata_search.py
"""

import env_bootstrap
import mlflow
import os
import uuid
import random
from openai import OpenAI


# 模拟多用户多会话
USERS = ["alice", "bob", "charlie"]


@mlflow.trace(span_type="CHAT", name="chat_turn")
def chat(user_id: str, session_id: str, question: str) -> str:
    """单次对话：包含 LLM 调用 + 元数据标记"""
    # 关键：装饰器让 chat() 内部有 active trace
    # 然后 update_current_trace 才能把 user/session 关联上去
    mlflow.update_current_trace(
        user=user_id,               # 存为 metadata.mlflow.trace.user
        session_id=session_id,      # 存为 metadata.mlflow.trace.session
        tags={
            "user_segment": "premium" if user_id == "alice" else "free",
        },
    )

    # 真实 LLM 调用（autolog 会自动追踪）
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


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("05_metadata")
    mlflow.openai.autolog()

    # ============ 1. 模拟多用户多会话 ============
    print("模拟 3 个用户 × 每个 2 轮对话 × 5 轮（15 次调用）...")

    with mlflow.start_run(run_name="multi-user-sim") as run:
        for user in USERS:
            session = f"sess-{user}-{uuid.uuid4().hex[:8]}"
            print(f"\n[用户 {user}, session {session[:16]}...]")

            for turn in range(5):
                questions = [
                    "推荐一首适合工作时听的歌",
                    "MLflow 的 Tracing 是什么？",
                    "Python 怎么读 CSV？",
                    "用一句话介绍 LangChain",
                    "写个 SQL 取前 10 行",
                ]
                q = random.choice(questions)
                answer = chat(user, session, q)
                print(f"  Q{turn+1}: {q}")
                print(f"    A: {answer[:60]}...")

    print(f"\n✓ Run: {run.info.run_id[:8]}")

    # ============ 2. search_traces 查询 ============
    print("\n" + "=" * 60)
    print("🔍 用 mlflow.search_traces 查询：")
    print("=" * 60)

    exp_id = mlflow.get_experiment_by_name("05_metadata").experiment_id

    # 查 1: 所有 status=OK 的 trace
    print("\n[1] 所有 OK 的 trace:")
    traces = mlflow.search_traces(
        experiment_ids=[exp_id],
        filter_string="status = 'OK'",
        max_results=20,
    )
    print(f"  数量: {len(traces)}")
    if len(traces) > 0:
        print(f"  列: {list(traces.columns)[:8]}")

    # 查 2: 找特定用户的 trace
    print("\n[2] 用户 alice 的 trace:")
    traces = mlflow.search_traces(
        experiment_ids=[exp_id],
        filter_string="metadata.`mlflow.trace.user` = 'alice'",
        max_results=10,
    )
    print(f"  数量: {len(traces)}")

    # 查 3: 找特定 session 的所有 trace
    print("\n[3] 按 session_id 聚合:")
    # 拿一个 alice 的 session_id 来演示（先查 alice 的 trace，从中拿一个 session_id）
    sample_traces = mlflow.search_traces(
        experiment_ids=[exp_id],
        filter_string="metadata.`mlflow.trace.user` = 'alice'",
        max_results=1,
    )
    if len(sample_traces) > 0:
        alice_session = sample_traces.iloc[0]["trace_metadata"].get("mlflow.trace.session", "")
        print(f"  选 alice session: {alice_session[:24]}...")
        sess_traces = mlflow.search_traces(
            experiment_ids=[exp_id],
            filter_string=f"metadata.`mlflow.trace.session` = '{alice_session}'",
        )
        print(f"  该 session 内 trace 数: {len(sess_traces)}")

    # 查 4: 按 latency 排序
    # ⚠️ search_traces 的 order_by 用字符串 list，不是 list[dict]
    # ⚠️ 字段名是 execution_time_ms，不是 execution_duration
    print("\n[4] 按延迟倒序（最慢的 5 次 LLM 调用）:")
    traces = mlflow.search_traces(
        experiment_ids=[exp_id],
        order_by=["execution_time_ms DESC"],
        max_results=5,
    )
    print(f"  前 5 个最慢的:")
    if len(traces) > 0:
        for _, t in traces.iterrows():
            user = t["trace_metadata"].get("mlflow.trace.user", "-")
            print(f"    - {t['execution_duration']:.0f}ms | user={user}")

    # ============ 3. 给 trace 打 feedback tag ============
    print("\n" + "=" * 60)
    print("💬 给 trace 加 feedback 标签（用于评估闭环）：")
    print("=" * 60)

    # 模拟：用户给前 3 个 trace 点 👍
    sample = mlflow.search_traces(experiment_ids=[exp_id], max_results=3)
    for _, t in sample.iterrows():
        trace_id = t["trace_id"]
        mlflow.set_trace_tag(trace_id, "user_feedback", "upvote")
        print(f"  trace {trace_id[:8]}... ✓ upvote")

    print("\n" + "=" * 60)
    print("📊 实战场景：")
    print("=" * 60)
    print("""
    1. 客服场景：按 user 看历史会话、按 session 聚合
    2. 质量监控：latency > 5s 的 trace 自动告警
    3. 反馈收集：用户点赞/点踩 → 收集到 eval 数据集
    4. 成本分析：按 user / session 聚合 token 消耗
    5. 调试：当某个用户的回答出问题时，能一键找到他所有相关 trace
    """)


if __name__ == "__main__":
    main()