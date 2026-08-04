"""
阶段 7 示例 1: mlflow.genai.evaluate + 内置 scorers
================================================

目标：
  - 构造一个简单的 LLM 应用（用 production prompt）
  - 跑 mlflow.genai.evaluate() 用内置 scorers（Correctness / Safety / RelevanceToQuery）
  - 看每个 scorer 的打分

运行：
  conda activate mlflow
  python 06_evaluation/07a_basic_evaluate.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "04_tracing"))
import env_bootstrap

import mlflow
import os
import pandas as pd
from openai import OpenAI


# ============ 1. 准备评估数据集 ============
# ⚠️ MLflow 要求每行有 inputs 列（dict），其他列是 expectations
# ⚠️ Correctness 需要 expected_response 或 expected_facts
# ⚠️ 同时提供 expected_response 和 expected_facts 可能触发额外 judge 调用（用默认模型）
#    所以要么只用 expected_response，要么只用 expected_facts
EVAL_DATA = pd.DataFrame([
    {
        "inputs": {"question": "什么是 MLflow？"},
        "expectations": {
            "expected_response": "MLflow 是开源的 ML 生命周期管理平台",
        },
    },
    {
        "inputs": {"question": "DeepSeek V4-Flash 是什么时候发布的？"},
        "expectations": {
            "expected_response": "DeepSeek V4-Flash 在 2026 年 4 月 24 日发布 V4 Preview",
        },
    },
    {
        "inputs": {"question": "MLflow 3 的核心新特性是什么？"},
        "expectations": {
            "expected_response": "LoggedModel 一等公民、ResponsesAgent、Prompt Optimization",
        },
    },
    {
        "inputs": {"question": "Python 怎么读取 CSV 文件？"},
        "expectations": {
            "expected_response": "用 pandas 的 read_csv 或标准库 csv 模块",
        },
    },
    {
        "inputs": {"question": "MLflow Tracking 是干什么的？"},
        "expectations": {
            "expected_response": "MLflow Tracking 记录实验参数、指标、artifact",
        },
    },
])


def predict_fn(question: str) -> str:
    """我们的 LLM 应用：用 OpenAI 客户端回答问题"""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )
    resp = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        messages=[
            {"role": "system", "content": "你是一个准确、简洁的 AI 助手。基于事实回答。"},
            {"role": "user", "content": question},
        ],
        max_tokens=200,
        temperature=0.3,
    )
    return resp.choices[0].message.content


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("07_evaluate")
    mlflow.openai.autolog()

    # ============ 2. 用内置 scorer 评估 ============
    from mlflow.genai.scorers import Correctness, Safety, RelevanceToQuery

    # ⚠️ MLflow 内置 scorers 默认用 gpt-4.1-mini 当 judge
    #    国内服务商不支持这个模型名，必须显式传 model= 参数
    #    URI 格式必须是 <provider>:/<model-name>，用 openai:/ 表示 OpenAI 协议
    # ⚠️ 注意：model 参数必须是非空字符串（不能传 None）
    raw_model = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    judge_model = f"openai:/{raw_model}"
    print(f"  judge_model = {judge_model}")

    print("=" * 60)
    print(f"🔍 用内置 scorers 评估（judge_model={judge_model}）：")
    print("=" * 60)

    result = mlflow.genai.evaluate(
        data=EVAL_DATA,
        predict_fn=predict_fn,
        scorers=[
            Correctness(model=judge_model),           # 对比 expectations
            Safety(model=judge_model),                # 检测不安全内容
            RelevanceToQuery(model=judge_model),      # 回答是否切题
        ],
    )

    # 打印结果
    print(f"\n评估完成:")
    print(f"  行数: {len(EVAL_DATA)}")
    print(f"\n聚合指标:")
    for metric, value in result.metrics.items():
        print(f"  {metric}: {value:.3f}")

    # ============ 3. 查看每行的打分（用 mlflow UI 更直观）============
    print("\n" + "=" * 60)
    print("📋 在 UI 看逐行打分：")
    print("=" * 60)
    print("""
    1. mlflow ui --port 5000
    2. 选 experiment '07_evaluate'
    3. 点开最新的 Run
    4. 看 'Evaluation' 标签（如果有）或 Artifacts/eval/
    5. 每行能看到:
       - inputs.question
       - outputs (predict_fn 返回)
       - expectations.expected_facts
       - 各 scorer 的打分 + reasoning

    也可以读 result.tables:
        print(result.tables['eval_results'])
    """)

    # 试试读逐行结果
    try:
        df = result.tables["eval_results"]
        print("\n逐行结果（前 3 行）：")
        cols = [c for c in df.columns if "score" in c or "feedback" in c]
        print(df[["inputs.question", *cols]].head(3).to_string(index=False))
    except Exception as e:
        print(f"\n（读不到逐行结果: {e}）")


if __name__ == "__main__":
    main()