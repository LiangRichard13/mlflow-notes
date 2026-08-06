"""
阶段 5 示例 2: @mlflow.trace 自定义 Span
========================================

目标：
  - 用 @mlflow.trace 装饰自己的函数（不只是 LLM 调用）
  - 标记 SpanType：LLM / RETRIEVER / TOOL / CHAIN / AGENT
  - 多层函数嵌套 → 多层 Span 嵌套

运行：
  conda activate mlflow
  python scripts/05_tracing/05c_custom_decorator.py
"""

import env_bootstrap
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _paths

import mlflow
from mlflow.entities.span import SpanType
import os
import time
import random
from openai import OpenAI


# 模拟一些非 LLM 的组件
@mlflow.trace(span_type=SpanType.RETRIEVER, name="retrieve_docs")
def retrieve_docs(query: str, k: int = 3):
    """模拟一个文档检索（向量数据库查 top-k）"""
    time.sleep(random.uniform(0.05, 0.15))   # 模拟 DB 延迟
    fake_corpus = [
        {"id": "doc1", "text": "MLflow 是开源的 ML 生命周期管理平台，由 Databricks 团队开发。"},
        {"id": "doc2", "text": "MLflow 包含 Tracking、Models、Registry、Projects 四大组件。"},
        {"id": "doc3", "text": "MLflow 3 引入了 GenAI 工作流：Tracing、Prompt Registry、Evaluation。"},
        {"id": "doc4", "text": "Traces 是嵌套的 Span 树，每个 Span 是一次操作。"},
    ]
    docs = random.sample(fake_corpus, k)
    return docs


@mlflow.trace(span_type=SpanType.TOOL, name="rerank")
def rerank_with_scores(query: str, docs: list) -> list:
    """模拟一个 rerank 操作"""
    time.sleep(random.uniform(0.02, 0.05))
    return sorted(docs, key=lambda d: len(d["text"]))  # 假 rerank


@mlflow.trace(span_type="PROMPT_TEMPLATE", name="build_prompt")   # 也可以自定义字符串
def build_prompt(question: str, docs: list) -> str:
    """组装最终 prompt"""
    context = "\n".join([f"- {d['text']}" for d in docs])
    return f"""基于以下上下文回答用户问题。只用上下文信息，不要编造。

上下文：
{context}

用户问题：{question}

回答："""


@mlflow.trace(span_type=SpanType.LLM, name="generate_answer")
def generate_answer(prompt: str, model: str) -> str:
    """真正的 LLM 调用（也可以用 mlflow.openai.autolog 自动追踪）"""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3,
    )
    return resp.choices[0].message.content


@mlflow.trace(span_type=SpanType.CHAIN, name="rag_chain")
def rag_chain(question: str, model: str) -> dict:
    """完整的 RAG chain：retrieve → rerank → build_prompt → generate"""
    # 1. 检索
    docs = retrieve_docs(question, k=3)

    # 2. 重排
    reranked = rerank_with_scores(question, docs)

    # 3. 组装 prompt
    prompt = build_prompt(question, reranked)

    # 4. LLM 生成
    answer = generate_answer(prompt, model)

    return {
        "question": question,
        "answer": answer,
        "num_docs": len(reranked),
    }


def main():
    mlflow.set_experiment("05_custom_tracing")

    # autolog 同时启用（这样 generate_answer 内部的 OpenAI 调用也会被追踪）
    mlflow.openai.autolog()

    model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    # ============ 单个 RAG 调用：会形成一棵 Span 树 ============
    print("=" * 60)
    print("Span 树结构（应该看到 5 层嵌套）：")
    print("=" * 60)

    with mlflow.start_run(run_name="rag-q1") as run:
        result = rag_chain("MLflow 是什么？", model)
        print(f"  Q: {result['question']}")
        print(f"  A: {result['answer'][:120]}...")
        print(f"  num_docs: {result['num_docs']}")
        run_id = run.info.run_id

    print(f"\n✓ Run: {run_id[:8]}")

    # ============ 批量调用：每个 RAG 都是独立 Trace ============
    print("\n" + "=" * 60)
    print("批量调用 5 次（每个 RAG 一个独立 Trace）：")
    print("=" * 60)

    questions = [
        "MLflow 的 Tracking 是什么？",
        "如何注册一个模型到 Registry？",
        "什么是 LoggedModel？",
        "怎么评估一个 LLM 应用？",
        "mlflow.genai.evaluate 和 mlflow.models.evaluate 区别？",
    ]
    with mlflow.start_run(run_name="rag-batch-5") as run:
        for i, q in enumerate(questions):
            result = rag_chain(q, model)
            print(f"\n  [{i+1}] Q: {q}")
            print(f"      A: {result['answer'][:80]}...")
            mlflow.log_metric(f"q{i+1}_docs", result["num_docs"])

    print("\n" + "=" * 60)
    print("📊 在 UI 的这个 Run 里你会看到：")
    print("=" * 60)
    print("""
    Traces 标签页：
      - 5 个 Trace（每个 RAG 一个）
      - 每个 Trace 内部嵌套 4 层 Span：
          rag_chain (CHAIN)
            ├── retrieve_docs (RETRIEVER) [k=3, 0.05-0.15s]
            ├── rerank (TOOL)              [0.02-0.05s]
            ├── build_prompt               [瞬时]
            └── generate_answer (LLM)      [OpenAI 调用，自动 autolog]
                └── chat completions span   [实际 LLM 调用]

    所有这些 Span 都共享同一个 Trace ID，可以点开看每一层的：
      - Inputs / Outputs
      - 耗时
      - Attributes（如 model、temperature、k 等）
    """)


if __name__ == "__main__":
    main()