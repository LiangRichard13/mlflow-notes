"""
阶段 7 示例 2: 自定义 Scorer（@scorer + make_judge）
=================================================

目标：
  - 用 @scorer 写 Python 自定义评分器（业务规则）
  - 用 make_judge() 写 LLM-as-judge 评分器（自然语言 rubric）
  - 组合多个 scorer 评估同一个输出

运行：
  conda activate mlflow
  python scripts/07_evaluation/07b_custom_scorer.py
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
import os
import re
import pandas as pd
from openai import OpenAI
from mlflow.genai.scorers import scorer, Correctness, Safety


# ============ 1. 业务规则型 scorer（@scorer）============

@scorer(name="has_citation")
def has_citation(outputs: str) -> bool:
    """回答必须包含至少一个 [source:xxx] 引用（业务硬性要求）"""
    if not isinstance(outputs, str):
        return False
    return bool(re.search(r"\[source:[^\]]+\]", outputs))


@scorer(name="is_concise")
def is_concise(outputs: str) -> bool:
    """回答不超过 100 字"""
    if not isinstance(outputs, str):
        return False
    return len(outputs) <= 100


@scorer(name="mentions_mlflow")
def mentions_mlflow(outputs: str) -> float:
    """回答提到 MLflow 的次数（0-5 区间）"""
    if not isinstance(outputs, str):
        return 0.0
    count = len(re.findall(r"(?i)mlflow", outputs))
    return min(float(count), 5.0)


# ============ 2. LLM-as-judge 型 scorer（make_judge）============

# 自定义 judge：评估"品牌语气是否友好专业"
# ⚠️ instructions 里必须至少有一个变量：{{ inputs }} / {{ outputs }} / {{ trace }} 等
brand_tone_judge = mlflow.genai.make_judge(
    name="brand_tone",
    instructions=(
        "你将评估 {{ outputs }} 的语气是否符合品牌要求：\n"
        "- 友好但不轻浮\n"
        "- 专业但不冷漠\n"
        "- 简洁但不敷衍\n"
        "打分范围 1-5：\n"
        "  1 = 完全不像品牌\n"
        "  2 = 大部分不像\n"
        "  3 = 一般\n"
        "  4 = 像\n"
        "  5 = 非常符合"
    ),
    model="openai:/" + os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
)


# ============ 3. 评估函数 ============

def predict_fn(question: str) -> str:
    """故意有的有 citation 有的没有，用来验证 scorer"""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )
    resp = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        messages=[
            {"role": "system", "content": "你是企业知识库助手。回答要专业、简洁。如果用了知识库信息，末尾加 [source:doc-id]。"},
            {"role": "user", "content": question},
        ],
        max_tokens=200,
        temperature=0.5,   # 高一些产生多样化回答
    )
    return resp.choices[0].message.content


def main():
    mlflow.set_experiment("07_custom_scorer")
    mlflow.openai.autolog()

    EVAL_DATA = pd.DataFrame([
        {"inputs": {"question": "MLflow 的核心组件有哪些？"}},
        {"inputs": {"question": "如何用 mlflow.sklearn.autolog() 自动记录？"}},
        {"inputs": {"question": "Trace 和 Span 的区别？"}},
        {"inputs": {"question": "MLflow Tracking 和 Model Registry 各自负责什么？"}},
    ])

    print("=" * 60)
    print("🔍 跑自定义 scorer 评估（4 个 scorer 并行）：")
    print("=" * 60)
    print("  - Correctness（内置，对比预期答案）")
    print("  - Safety（内置，检测不安全内容）")
    print("  - has_citation（自定义，业务规则）")
    print("  - is_concise（自定义，字数约束）")
    print("  - mentions_mlflow（自定义，关键词计数）")
    print("  - brand_tone（LLM-as-judge）")

    result = mlflow.genai.evaluate(
        data=EVAL_DATA,
        predict_fn=predict_fn,
        scorers=[
            Correctness(),
            Safety(),
            has_citation,
            is_concise,
            mentions_mlflow,
            brand_tone_judge,
        ],
    )

    print(f"\n聚合指标:")
    for metric, value in result.metrics.items():
        print(f"  {metric}: {value:.3f}")

    # ============ 4. 详细对比 ============
    print("\n" + "=" * 60)
    print("📋 自定义 scorer 的特点：")
    print("=" * 60)
    print("""
    @scorer 装饰器：
      - 同步 Python 函数，参数名要符合约定
      - inputs (dict): 原始输入
      - outputs (任意): predict_fn 返回值
      - expectations (dict): 数据集中的期望列
      - 返回：bool / float / int / str

    make_judge：
      - 适合主观/复杂判断（语气、风格、合理性）
      - 用一个 LLM 当裁判（meta-judge）
      - 自然语言 rubric，无需写代码

    实际选择：
      - 硬性规则（必须有引用、不能超过 N 字）→ @scorer
      - 主观判断（语气、合理性、相关性）→ make_judge
      - 需要 ground truth 对比 → Correctness / Equivalence
    """)


if __name__ == "__main__":
    main()