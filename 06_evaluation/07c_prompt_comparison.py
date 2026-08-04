"""
阶段 7 示例 3: 跨提示词版本对比评估
==================================

目标：
  - 用 production 和 staging 两个版本的提示词各跑一遍评估
  - 在 UI 中对比哪个版本更好
  - 演示 MLflow 的 prompt A/B testing 工作流

运行：
  conda activate mlflow
  python 06_evaluation/07c_prompt_comparison.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "04_tracing"))
import env_bootstrap

import mlflow
import os
import pandas as pd
from openai import OpenAI
from mlflow.genai.scorers import Correctness, Safety
from mlflow.genai.scorers import scorer


PROMPT_URI_TEMPLATE = "prompts:/customer-support-qa@{}"


@scorer(name="response_length_ok")
def response_length_ok(outputs: str) -> bool:
    """回答长度在 50-200 字之间"""
    if not isinstance(outputs, str):
        return False
    return 50 <= len(outputs) <= 200


def predict_with_prompt_version(alias: str, question: str) -> str:
    """用指定 alias 的 prompt 回答问题"""
    # 从 Registry 加载 prompt
    prompt_obj = mlflow.genai.load_prompt(PROMPT_URI_TEMPLATE.format(alias))

    # 渲染（chat 格式 / 文本格式都支持）
    # 收集 prompt 模板需要的变量
    variables = prompt_obj.variables or set()
    fmt_kwargs = {"question": question}
    if "company" in variables:
        fmt_kwargs["company"] = "DeepSeek"
    # 给所有模板变量都填个默认值，防止报错
    for v in variables:
        if v not in fmt_kwargs:
            fmt_kwargs[v] = "(默认)"

    if prompt_obj.is_text_prompt:
        # 文本模板
        prompt_text = prompt_obj.format(**fmt_kwargs)
        messages = [{"role": "user", "content": prompt_text}]
    else:
        # chat 格式列表（MLflow 渲染时也会填充变量）
        try:
            messages = prompt_obj.format(**fmt_kwargs)
        except Exception:
            # fallback：手动替换
            messages = []
            for m in prompt_obj.template:
                content = m["content"]
                for k, v in fmt_kwargs.items():
                    content = content.replace("{{ " + k + " }}", str(v))
                    content = content.replace("{{" + k + "}}", str(v))
                messages.append({**m, "content": content})

    # 调 LLM
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )
    resp = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        messages=messages,
        max_tokens=200,
        temperature=0.3,
    )
    return resp.choices[0].message.content


# 评估集（固定，对比两个版本）
# ⚠️ predict_fn 接受一个参数（通常命名 row），参数名要与 inputs 列的 key 对应
EVAL_DATA = pd.DataFrame([
    {"inputs": {"row": {"question": "MLflow 是干什么的？"}},
     "expectations": {"expected_response": "开源 ML 生命周期管理平台"}},
    {"inputs": {"row": {"question": "MLflow 3 的 LoggedModel 是什么？"}},
     "expectations": {"expected_response": "MLflow 3 把模型变成一等公民"}},
    {"inputs": {"row": {"question": "Prompt Registry 怎么用？"}},
     "expectations": {"expected_response": "register_prompt + set_prompt_alias"}},
    {"inputs": {"row": {"question": "怎么评估一个 LLM 应用？"}},
     "expectations": {"expected_response": "mlflow.genai.evaluate + 自定义 scorer"}},
    {"inputs": {"row": {"question": "MLflow Tracing 是什么？"}},
     "expectations": {"expected_response": "记录 LLM 调用的嵌套 span 树"}},
])


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("07_prompt_ab")
    mlflow.openai.autolog()

    judge = f"openai:/{os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-flash')}"

    # ============ 跑 production 版本 ============
    print("=" * 60)
    print("📦 评估 production 版本：")
    print("=" * 60)
    with mlflow.start_run(run_name="production") as run_prod:
        result_prod = mlflow.genai.evaluate(
            data=EVAL_DATA,
            predict_fn=lambda row: predict_with_prompt_version("production", row["question"]),
            scorers=[Correctness(model=judge), Safety(model=judge), response_length_ok],
        )
        prod_metrics = {k: v for k, v in result_prod.metrics.items()}

    # ============ 跑 staging 版本 ============
    print("\n" + "=" * 60)
    print("🧪 评估 staging 版本：")
    print("=" * 60)
    with mlflow.start_run(run_name="staging") as run_stg:
        result_stg = mlflow.genai.evaluate(
            data=EVAL_DATA,
            predict_fn=lambda row: predict_with_prompt_version("staging", row["question"]),
            scorers=[Correctness(model=judge), Safety(model=judge), response_length_ok],
        )
        stg_metrics = {k: v for k, v in result_stg.metrics.items()}

    # ============ 对比 ============
    print("\n" + "=" * 60)
    print("📊 对比结果：")
    print("=" * 60)
    print(f"{'指标':<30} {'production':<15} {'staging':<15} {'赢家':<10}")
    print("-" * 70)
    for metric in prod_metrics:
        p = prod_metrics[metric]
        s = stg_metrics.get(metric, 0)
        winner = "production" if p >= s else "staging"
        print(f"{metric:<30} {p:<15.3f} {s:<15.3f} {winner:<10}")

    # ============ 决策建议 ============
    print("\n" + "=" * 60)
    print("🎯 决策建议：")
    print("=" * 60)
    prod_correct = prod_metrics.get("correctness/mean", 0)
    stg_correct = stg_metrics.get("correctness/mean", 0)
    if stg_correct > prod_correct + 0.1:   # staging 明显更好
        print(f"  ✅ staging 比 production 高 {(stg_correct - prod_correct):.2%}，建议切到 staging")
        print(f"""
        # 切到 staging:
        mlflow.genai.set_prompt_alias(
            "customer-support-qa", "production",
            version=<staging 的 version>,
        )
        """)
    elif prod_correct > stg_correct + 0.1:
        print(f"  ⚠️ production 比 staging 好 {(prod_correct - stg_correct):.2%}，保持 production")
    else:
        print(f"  🟰 两者差不多（差 {(stg_correct - prod_correct):.2%}），继续 A/B 测试更多数据")

    print("\n" + "=" * 60)
    print("💡 在 UI 中查看更详细信息：")
    print("=" * 60)
    print("""
    1. mlflow ui --port 5000
    2. 选 experiment '07_prompt_ab'
    3. 选 production 和 staging 两个 Run
    4. 点 'Compare' → 看 metrics 差异
    5. 看逐行打分：每个 Run 的 "Evaluation" / "Traces" 标签
    """)


if __name__ == "__main__":
    main()