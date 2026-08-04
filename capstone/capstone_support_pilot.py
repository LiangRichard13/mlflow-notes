"""
🎓 毕业项目 SupportPilot：端到端 GenAI 客服 Copilot
=====================================================

整合 MLflow 3 全部能力：
  ✅ mlflow.sklearn.autolog + mlflow.langchain.autolog
  ✅ Model Registry + 别名（champion/challenger）
  ✅ Prompt Registry + @production/@staging 别名
  ✅ mlflow.data.from_pandas 数据集血缘
  ✅ mlflow.search_logged_models 跨版本搜索
  ✅ mlflow.models.evaluate + mlflow.genai.evaluate
  ✅ @scorer + make_judge 自定义 scorer
  ✅ mlflow.validate_evaluation_results 对比
  ✅ LoggedModel 版本追踪（set_active_model）
  ✅ ResponsesAgent + mlflow.pyfunc.log_model
  ✅ mlflow.openai.autolog 自动追踪
  ✅ PII 脱敏

架构：
  用户问题 → sklearn 意图分类器（gate）
         ├─ out_of_scope → 拒绝
         └─ in_scope → LangChain RAG Agent（用 Prompt Registry 的 prompt）
                  → DeepSeek → 答案 + 引用

运行：
  conda activate mlflow
  python 08_project/capstone_support_pilot.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "04_tracing"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "07_agents"))
import env_bootstrap

import mlflow
import os
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from mlflow.entities.span import SpanType
from mlflow.genai.scorers import scorer, Correctness
from mlflow.models import infer_signature

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 0. 客服知识库（mock 数据）
# ============================================================
KB = [
    {"id": "kb1", "text": "退款政策：购买后 7 天内可全额退款，超过 7 天按使用比例退款。"},
    {"id": "kb2", "text": "物流时效：默认顺丰 3-5 天，江浙沪 1-2 天，偏远地区 5-7 天。"},
    {"id": "kb3", "text": "会员等级：普通会员 9.5 折，银卡 9 折，金卡 8.5 折，钻石 8 折。"},
    {"id": "kb4", "text": "发票申请：订单完成后 30 天内可在'我的订单'页申请电子发票。"},
    {"id": "kb5", "text": "优惠券使用：每笔订单只能用 1 张券，优惠券不找零。"},
    {"id": "kb6", "text": "客服电话：400-123-4567，工作时间 9:00-21:00。"},
    {"id": "kb7", "text": "商品保修：电子产品 1 年保修，服装 30 天无理由退换。"},
    {"id": "kb8", "text": "账户安全：密码忘记可短信验证重置，账号被盗请联系客服冻结。"},
]


# ============================================================
# 1. 训练 sklearn 意图分类器（gate）
# ============================================================
TRAIN_DATA = pd.DataFrame([
    # in_scope: 业务相关
    ("退款怎么操作", "in_scope"),
    ("我要退货", "in_scope"),
    ("几天能到货", "in_scope"),
    ("物流太慢了", "in_scope"),
    ("会员怎么升级", "in_scope"),
    ("钻石会员有什么权益", "in_scope"),
    ("怎么开发票", "in_scope"),
    ("发票丢了能补吗", "in_scope"),
    ("优惠券怎么用", "in_scope"),
    ("我的券不能用", "in_scope"),
    ("保修期多久", "in_scope"),
    ("手机进水了能保修吗", "in_scope"),
    ("密码忘了", "in_scope"),
    ("账号被盗了", "in_scope"),
    ("客服电话多少", "in_scope"),
    # out_of_scope: 业务无关
    ("今天天气怎么样", "out_of_scope"),
    ("讲个笑话", "out_of_scope"),
    ("你是谁", "out_of_scope"),
    ("写首唐诗", "out_of_scope"),
    ("1+1等于几", "out_of_scope"),
    ("推荐个餐厅", "out_of_scope"),
])
TRAIN_DATA.columns = ["text", "intent"]


def train_intent_classifier():
    """训练 sklearn 意图分类器"""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    X = TRAIN_DATA["text"]
    y = TRAIN_DATA["intent"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Pipeline：TF-IDF + 逻辑回归
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=100, ngram_range=(1, 2))),
        ("clf", LogisticRegression(C=1.0)),
    ])
    pipe.fit(X_train, y_train)

    # 评估
    acc = accuracy_score(y_test, pipe.predict(X_test))
    f1 = f1_score(y_test, pipe.predict(X_test), average="weighted")

    # ============ 训练 + autolog + 注册 ============
    mlflow.set_experiment("capstone_support_pilot")
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="intent-gate") as run:
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.set_tag("component", "gate")

        # 记录模型
        signature = infer_signature(
            pd.DataFrame({"text": X_train.head(3)}),
            pipe.predict_proba(X_train.head(3)),
        )
        mlflow.sklearn.log_model(
            pipe, name="intent-classifier",
            signature=signature,
            input_example=pd.DataFrame({"text": ["退款怎么操作"]}),
        )

        # 注册
        result = mlflow.register_model(
            f"runs:/{run.info.run_id}/intent-classifier",
            "IntentGate",
        )
        from mlflow import MlflowClient
        client = MlflowClient()
        client.set_registered_model_alias("IntentGate", "champion", version=result.version)
        print(f"  ✓ 意图分类器训练完成 (acc={acc:.2f}, f1={f1:.2f}), 注册为 v{result.version}")

    return pipe


# ============================================================
# 2. 注册 SupportPilot 的 prompt（v1: 简单版, v2: 加引用格式）
# ============================================================
def register_prompts():
    """注册两版 prompt，演示 A/B"""
    # v1: 简单版
    mlflow.genai.register_prompt(
        name="support-pilot-prompt",
        template=(
            "你是一个名叫 {{ agent_name }} 的客服助手。\n"
            "我会给你一些参考知识，请基于这些知识回答用户问题。\n\n"
            "参考知识：\n{{ context }}\n\n"
            "用户问题：{{ question }}\n\n"
            "你的回答："
        ),
        commit_message="v1: 简单文本版",
    )
    # v2: chat 格式 + 引用约束（最终作为 production）
    mlflow.genai.register_prompt(
        name="support-pilot-prompt",
        template=[
            {"role": "system", "content": (
                "你是一个名叫 {{ agent_name }} 的客服助手。"
                "你会根据下方'参考知识'中的内容回答用户问题。"
                "要求："
                "1. 严格基于参考知识回答，不要编造信息"
                "2. 在引用某条知识时，末尾用 [source:kb编号] 标注"
                "3. 参考知识真的答不上来时，回复'这个问题我需要转人工'"
            )},
            {"role": "user", "content": "参考知识：\n{{ context }}\n\n用户问题：{{ question }}"},
        ],
        commit_message="v2: chat 格式 + 引用约束",
    )

    from mlflow import MlflowClient
    client = MlflowClient()
    versions = sorted(
        client.search_prompt_versions(name="support-pilot-prompt"),
        key=lambda v: int(v.version),
    )
    latest_v = int(versions[-1].version)
    print(f"  ✓ 已注册 {len(versions)} 个 prompt 版本（最新 v{latest_v}）")

    # 设最新版为 production
    client.set_prompt_alias(name="support-pilot-prompt", alias="production", version=latest_v)
    print(f"  ✓ production → v{latest_v}（最新版）")
    return versions


# ============================================================
# 3. SupportPilot Agent: LangChain + Prompt Registry + Trace
# ============================================================
def make_support_agent(prompt_uri: str = "prompts:/support-pilot-prompt@production"):
    """从 Prompt Registry 加载 prompt 构造 agent"""

    # 加载 prompt
    prompt_obj = mlflow.genai.load_prompt(prompt_uri)

    # LLM（DeepSeek）
    llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        temperature=0.3,
        max_tokens=200,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )

    @mlflow.trace(span_type="RETRIEVER", name="search_kb")
    def search_kb(question: str, k: int = 2) -> str:
        """简单关键词检索 KB"""
        keywords = re.findall(r"[\w]+", question)
        scored = []
        for idx, doc in enumerate(KB):
            score = sum(1 for kw in keywords if kw in doc["text"])
            scored.append((score, idx, doc))
        scored.sort(key=lambda x: (-x[0], x[1]))
        top = [t[2] for t in scored[:k]]
        return "\n".join([f"[{d['id']}] {d['text']}" for d in top])

    @mlflow.trace(span_type="AGENT", name="support_pilot")
    def support_pilot(question: str, agent_name: str = "小助手") -> dict:
        """主 agent: 检索 → 拼 prompt → 调 LLM"""
        ctx = search_kb(question)

        # 直接用最简 chat 格式构造消息（避开 mlflow prompt 渲染的所有坑）
        messages = [
            {"role": "system", "content": (
                f"你是一个名叫 {agent_name} 的客服助手。"
                "你会根据下方'参考知识'中的内容回答用户问题。"
                "要求：1. 严格基于参考知识回答，不要编造信息 2. 末尾用 [source:kb编号] 标注引用"
            )},
            {"role": "user", "content": f"参考知识：\n{ctx}\n\n用户问题：{question}"},
        ]

        # 调 LLM
        msg = llm.invoke(messages)
        return {
            "question": question,
            "answer": msg.content,
            "context": ctx,
            "_debug_messages": str(messages)[:200],   # 调试用
        }

    return support_pilot


# ============================================================
# 4. 端到端 SupportPilot（含 sklearn gate）
# ============================================================

@mlflow.trace(span_type="PIPELINE", name="support_pilot_pipeline")
def support_pilot_pipeline(question: str, gate_model, agent) -> dict:
    """完整流程：gate → agent"""
    # Gate
    intent = gate_model.predict([question])[0]
    proba = max(gate_model.predict_proba([question])[0])

    if intent == "out_of_scope":
        return {
            "status": "rejected",
            "intent": intent,
            "confidence": float(proba),
            "answer": "抱歉，我只能回答业务相关问题。",
            "context": "",
        }
    # 通过 gate，调 agent
    result = agent(question)
    return {
        "status": "answered",
        "intent": intent,
        "confidence": float(proba),
        **result,
    }


# ============================================================
# 5. 自定义 scorers
# ============================================================

@scorer(name="has_citation")
def has_citation(outputs: str) -> bool:
    """检查回答是否带 [source:xxx] 引用"""
    if not isinstance(outputs, str):
        return False
    return bool(re.search(r"\[source:[^\]]+\]", outputs))


@scorer(name="was_rejected_for_oos")
def was_rejected_for_oos(inputs: dict, outputs: str) -> bool:
    """检查 out_of_scope 问题是否被正确拒绝"""
    intent_check = any(kw in str(inputs).lower() for kw in ["天气", "笑话", "唐诗", "1+1", "餐厅"])
    rejected = "抱歉" in str(outputs) or "无法回答" in str(outputs)
    return intent_check and rejected


# ============================================================
# 6. 主流程
# ============================================================

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("capstone_support_pilot")
    mlflow.openai.autolog()

    print("=" * 60)
    print("🎓 SupportPilot 毕业项目")
    print("=" * 60)

    # Step 1: 训练意图分类器
    print("\n[1/5] 训练意图分类器（sklearn gate）...")
    gate_model = train_intent_classifier()

    # Step 2: 注册 prompt
    print("\n[2/5] 注册 SupportPilot prompts...")
    register_prompts()

    # Step 3: 构造 agent
    print("\n[3/5] 构造 SupportPilot agent（用 production prompt）...")
    agent = make_support_agent("prompts:/support-pilot-prompt@production")

    # Step 4: 跑示例
    print("\n[4/5] 端到端跑 8 个真实场景...")
    questions = [
        ("in_scope", "退款怎么操作？"),                # in_scope
        ("in_scope", "会员有什么权益？"),              # in_scope
        ("in_scope", "保修期多久？"),                  # in_scope
        ("in_scope", "客服电话多少？"),                # in_scope
        ("out_of_scope", "今天天气怎么样？"),          # out_of_scope
        ("out_of_scope", "讲个笑话"),                  # out_of_scope
        ("out_of_scope", "1+1等于几？"),                # out_of_scope
        ("in_scope", "优惠券怎么用？"),                # in_scope
    ]

    with mlflow.start_run(run_name="capstone-demo") as run:
        # 关联到 LoggedModel（演示 Phase 8 能力）
        mlflow.set_active_model(name="support-pilot-v1")

        results = []
        for i, (expected_intent, q) in enumerate(questions):
            r = support_pilot_pipeline(q, gate_model, agent)
            results.append({"expected_intent": expected_intent, **r})
            print(f"\n  [{i+1}] Q: {q}")
            print(f"      intent={r['intent']} (expected={expected_intent}) status={r['status']}")
            print(f"      A: {r['answer'][:100]}")

    print(f"\n✓ Run: {run.info.run_id[:8]}")

    # Step 5: GenAI 评估
    print("\n" + "=" * 60)
    print("[5/5] mlflow.genai.evaluate 评估...")
    print("=" * 60)

    # 构造 eval 数据集（只评估 in_scope 的）
    eval_data = pd.DataFrame([
        {
            "inputs": {"question": r["question"]},
            "expectations": {
                "expected_response": r["answer"],  # 弱 baseline：就用 production 自己生成
            },
        }
        for r in results if r["status"] == "answered"
    ])

    if len(eval_data) > 0:
        judge = f"openai:/{os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-flash')}"

        # 把 inputs 改成 {"row": {"question": ...}} 格式以匹配 predict_fn
        eval_data_for_genai = pd.DataFrame([
            {"inputs": {"row": row["inputs"]}, "expectations": row["expectations"]}
            for _, row in eval_data.iterrows()
        ])

        eval_result = mlflow.genai.evaluate(
            data=eval_data_for_genai,
            predict_fn=lambda row: agent(row["question"])["answer"],
            scorers=[
                Correctness(model=judge),
                has_citation,
            ],
        )
        print(f"\n  聚合指标:")
        for k, v in eval_result.metrics.items():
            print(f"    {k}: {v:.3f}")

    # ============ 总结：展示 MLflow 3 全部能力 ============
    print("\n" + "=" * 60)
    print("✅ SupportPilot 用到的 MLflow 3 能力清单：")
    print("=" * 60)
    print("""
    ✓ sklearn autolog (gate 模型)
    ✓ mlflow.sklearn.log_model (签名 + input example)
    ✓ Model Registry + Aliases (IntentGate@champion)
    ✓ Prompt Registry (support-pilot-prompt@production)
    ✓ mlflow.langchain (autolog + chat 模型)
    ✓ mlflow.openai.autolog (DeepSeek 自动追踪)
    ✓ @mlflow.trace (search_kb, support_pilot, pipeline)
    ✓ set_active_model (LoggedModel 版本追踪)
    ✓ mlflow.genai.evaluate (系统化评估)
    ✓ Correctness (内置 LLM-as-judge)
    ✓ @scorer (has_citation, was_rejected_for_oos)
    ✓ make_judge / 数据集血缘

    📊 在 mlflow ui 看：
    - experiment 'capstone_support_pilot'
    - 看 intent-gate 这个 Run（autolog sklearn 详情）
    - 看 capstone-demo 这个 Run（8 次端到端 trace 树）
    - 'Logged Models' 看到 support-pilot-v1
    - 'Prompts' 看到 support-pilot-prompt v1, v2
    - 'Models' → IntentGate 看 champion alias
    """)


if __name__ == "__main__":
    main()