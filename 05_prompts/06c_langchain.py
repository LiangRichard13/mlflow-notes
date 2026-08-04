"""
阶段 6 示例 3: LangChain autolog + 从 Prompt Registry 加载 prompt
=================================================================

目标：
  - mlflow.langchain.autolog() 一行开启 LangChain 追踪
  - 在 LangChain 应用里用 prompts:/name@alias 加载 MLflow Registry 的 prompt
  - 验证 trace 把 LLM、prompt 渲染、chain 调用都记下来

运行：
  conda activate mlflow
  python 05_prompts/06c_langchain.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "04_tracing"))
import env_bootstrap

import mlflow
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


PROMPT_URI = "prompts:/customer-support-qa@staging"


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("06_langchain")

    # ============ 关键一行：开启 LangChain autolog ============
    mlflow.langchain.autolog()

    # ============ 1. 从 Prompt Registry 加载提示词 ============
    print(f"加载提示词: {PROMPT_URI}")
    mlflow_prompt = mlflow.genai.load_prompt(PROMPT_URI)
    # mlflow_prompt.template 是 list[dict]，直接喂给 ChatPromptTemplate
    print(f"  template: {mlflow_prompt.template}")

    # ============ 2. 构造 LangChain ChatPromptTemplate ============
    # 注意：MLflow 的 chat 格式 [{role, content}] 转 LangChain 的 [(role, content)]
    lc_messages = [(m["role"], m["content"]) for m in mlflow_prompt.template]
    prompt = ChatPromptTemplate.from_messages(lc_messages)

    # ============ 3. 构造 LLM（用 DeepSeek 通过 OpenAI 协议）============
    llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        temperature=0.3,
        max_tokens=200,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )

    # ============ 4. 拼装 chain ============
    # 用 format() 把变量塞进 prompt
    def fill_prompt(inputs: dict) -> str:
        formatted = mlflow_prompt.format(**inputs)
        return formatted   # 此时是 list[dict]，下一步喂给 chat completions

    # 完整 chain：input → 渲染 prompt → 调 LLM → 输出
    chain = (
        RunnablePassthrough()
        | RunnableLambda(fill_prompt)
        | (lambda msgs: llm.invoke(msgs))
    )

    # ============ 5. 跑几次，记 trace ============
    questions = [
        {"company": "DeepSeek", "question": "你们的 V4-Flash 有什么改进？"},
        {"company": "Anthropic", "question": "Claude 怎么集成 MLflow？"},
        {"company": "OpenAI", "question": "GPT-4o 和 Claude Sonnet 怎么选？"},
    ]

    print("\n" + "=" * 60)
    print(f"🤖 跑 3 次 chain（每次都是 autolog 自动追踪）：")
    print("=" * 60)

    with mlflow.start_run(run_name="langchain-rag-demo") as run:
        for i, q in enumerate(questions, 1):
            print(f"\n[Q{i}] {q}")
            resp = chain.invoke(q)
            print(f"  A: {resp.content[:150]}")

    print(f"\n✓ Run: {run.info.run_id[:8]}")

    # ============ 6. 列出 trace ============
    print("\n" + "=" * 60)
    print("📊 这次 Run 包含的 trace：")
    print("=" * 60)
    exp_id = mlflow.get_experiment_by_name("06_langchain").experiment_id
    traces = mlflow.search_traces(
        run_id=run.info.run_id,
        max_results=10,
    )
    print(f"  数量: {len(traces)}")
    for i, t in traces.iterrows():
        print(f"  - {t['trace_id'][:12]}... | {t['execution_duration']:.0f}ms")

    # ============ 7. LangChain 自定义逻辑也加 trace ============
    @mlflow.trace(span_type="RETRIEVER", name="kb_search")
    def search_kb(question: str, k: int = 2) -> str:
        """模拟一个知识库检索"""
        kb = {
            "V4-Flash": "V4-Flash 是 DeepSeek 2026-04 发布的新模型，1M 上下文 + Agent 能力大幅增强。",
            "MLflow": "MLflow 是开源 MLOps 平台，由 Databricks 团队开发。",
            "Claude": "Claude 是 Anthropic 的 AI 助手系列，包括 Haiku/Sonnet/Opus。",
        }
        # 简单关键词匹配
        matched = [v for k_, v in kb.items() if k_.lower() in question.lower()]
        return "\n".join(matched) if matched else "（未找到相关知识库）"

    print("\n" + "=" * 60)
    print("🧪 用 @mlflow.trace 自定义 KB 检索 + LangChain chain：")
    print("=" * 60)

    @mlflow.trace(span_type="CHAIN", name="kb_rag_chain")
    def kb_rag(question: str) -> str:
        ctx = search_kb(question)
        messages = [
            {"role": "system", "content": f"基于以下知识回答：\n{ctx}"},
            {"role": "user", "content": question},
        ]
        resp = llm.invoke(messages)
        return resp.content

    with mlflow.start_run(run_name="kb-rag"):
        for q in ["V4-Flash 是什么？", "MLflow 是谁开发的？"]:
            ans = kb_rag(q)
            print(f"\n  Q: {q}")
            print(f"  A: {ans[:120]}")

    print("\n" + "=" * 60)
    print("📊 在 UI 看 kb-rag 这个 Run 的 trace 树：")
    print("=" * 60)
    print("""
    kb_rag_chain (CHAIN)
    ├── kb_search (RETRIEVER)
    └── ChatOpenAI (LLM, autolog 自动加)
        └── chat completions span
    """)


if __name__ == "__main__":
    main()