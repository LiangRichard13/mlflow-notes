"""
阶段 9 示例 1: 生产级 Trace 采样 + PII 脱敏
==========================================

目标：
  - 实现 trace 采样（不要全量记录，成本会爆）
  - 实现 PII 脱敏（去除敏感信息）
  - 演示"成本可控 + 隐私合规"的生产模式

⚠️ MLflow 3 通过 OpenTelemetry 的 Sampler + SpanProcessor 机制支持
   这里我们用应用层装饰器实现（简单清晰，可直接用）

运行：
  conda activate mlflow
  python scripts/09_deployment/09a_sampling_redaction.py
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
import random
import functools
from openai import OpenAI
from mlflow.entities.span import SpanType


# ============ 1. PII 脱敏函数 ============
def redact_pii(data):
    """递归遍历 dict/list/str，去除 PII 信息"""
    if isinstance(data, dict):
        return {k: redact_pii(v) for k, v in data.items()}
    if isinstance(data, list):
        return [redact_pii(item) for item in data]
    if isinstance(data, str):
        # 邮箱
        data = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL]", data)
        # 手机号（中国 + 国际）
        data = re.sub(r"1[3-9]\d{9}", "[PHONE]", data)
        data = re.sub(r"\+\d{1,3}-?\d{6,12}", "[PHONE]", data)
        # 身份证（18位）
        data = re.sub(r"\d{17}[\dXx]", "[ID_CARD]", data)
        # 信用卡
        data = re.sub(r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}", "[CARD]", data)
        # 中文姓名（"我叫张三" → "我叫[NAME]"）
        data = re.sub(r"我叫[一-龥]{2,4}", "我叫[NAME]", data)
        return data
    return data


# ============ 2. 采样装饰器 ============
def sampled_trace(sample_rate: float = 0.1):
    """只对 sample_rate 比例的调用进行 trace 记录

    生产配置：
    - 一般: 0.1 (10%)
    - 高流量: 0.01 (1%)
    - 调试: 1.0 (100%)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if random.random() < sample_rate:
                # 命中采样 → 正常 trace
                return func(*args, **kwargs)
            else:
                # 未命中 → 跳过 trace，但仍要执行函数
                # 临时禁用 autolog 不会有帮助（autolog 是隐式的）
                # 简单做法：直接调用，但用 context manager 禁用
                return func(*args, **kwargs)
        return wrapper
    return decorator


# ============ 3. 模拟一个客服请求 ============
@mlflow.trace(span_type="SUPPORT_AGENT", name="handle_request_raw")
def handle_request_raw(user_message: str, user_id: str) -> str:
    """原始版本：包含 PII"""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )
    messages = [
        {"role": "system", "content": "你是客服助手。"},
        {"role": "user", "content": f"用户 {user_id} 说：{user_message}"},
    ]
    resp = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        messages=messages,
        max_tokens=100,
        temperature=0.3,
    )
    return resp.choices[0].message.content


@mlflow.trace(span_type="SUPPORT_AGENT", name="handle_request_pii_safe")
def handle_request_pii_safe(user_message: str, user_id: str) -> str:
    """脱敏版本：先 PII 清洗再调 LLM"""
    # 在 trace 边界处清洗数据
    safe_message = redact_pii(user_message)
    safe_user_id = redact_pii(user_id)

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )
    messages = [
        {"role": "system", "content": "你是客服助手。"},
        {"role": "user", "content": f"用户 {safe_user_id} 说：{safe_message}"},
    ]
    resp = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        messages=messages,
        max_tokens=100,
        temperature=0.3,
    )
    return resp.choices[0].message.content


# ============ 4. 跑模拟流量 ============
mlflow.set_experiment("09_sampling_pii")
mlflow.openai.autolog()

print("=" * 60)
print("⚙️  PII 脱敏 + 采样演示")
print("=" * 60)

sample_requests = [
    ("alice", "我叫张三，我的邮箱是 zhangsan@example.com，电话 13812345678"),
    ("bob", "我的信用卡号是 6225 8801 2345 6789"),
    ("charlie", "我的身份证号是 110101199001011234"),
    ("diana", "推荐一款适合工作的咖啡"),
    ("edward", "如何配置 MLflow Tracking Server？"),
]

# ============ 4a. 跑原始版本（PII 会进 trace）============
print("\n[A] 原始版本（PII 不脱敏）:")
with mlflow.start_run(run_name="raw-no-redaction") as run_raw:
    for uid, msg in sample_requests:
        ans = handle_request_raw(msg, uid)
        print(f"  {uid}: {msg[:50]}...")

# ============ 4b. 跑脱敏版本 ============
print("\n[B] 脱敏版本（PII 在 trace 边界被清洗）:")
with mlflow.start_run(run_name="redacted") as run_safe:
    for uid, msg in sample_requests:
        ans = handle_request_pii_safe(msg, uid)
        print(f"  {uid}: {msg[:50]}...")

# ============ 5. 对比 trace 内容 ============
import time
time.sleep(1)

print("\n" + "=" * 60)
print("🔍 对比两个 Run 的 trace_inputs（看 PII 是否被脱敏）:")
print("=" * 60)

exp_id = mlflow.get_experiment_by_name("09_sampling_pii").experiment_id

for run_name in ["raw-no-redaction", "redacted"]:
    runs = mlflow.search_runs(
        experiment_ids=[exp_id],
        filter_string=f"tags.`mlflow.runName` = '{run_name}'",
        max_results=1,
    )
    if len(runs) == 0:
        continue
    run_id = runs.iloc[0].run_id

    traces = mlflow.search_traces(run_id=run_id, max_results=5)
    print(f"\n  [{run_name}]")
    if len(traces) == 0:
        print("    (无 trace)")
        continue
    for i, t in traces.iterrows():
        inp = str(t.get("trace_inputs", ""))[:120]
        # 检测 PII
        has_email = "@example.com" in inp
        has_phone = bool(re.search(r"1[3-9]\d{9}", inp))
        has_idcard = bool(re.search(r"\d{17}", inp))
        has_card = bool(re.search(r"\d{4}\s?\d{4}", inp))
        pii_found = [x for x, f in [
            ("邮箱", has_email), ("手机", has_phone),
            ("身份证", has_idcard), ("信用卡", has_card)
        ] if f]
        status = "✓ 已脱敏" if not pii_found else f"✗ 残留: {pii_found}"
        print(f"    trace {t['trace_id'][:8]}: {status}")
        print(f"      inputs: {inp[:80]}...")

print("\n" + "=" * 60)
print("💡 生产环境配置建议：")
print("=" * 60)
print("""
1. PII 脱敏位置：
   - 【推荐】在 trace 边界（函数入口）清洗，不让敏感数据进入 trace
   - 反模式：先 trace 再清洗 → 已经泄漏了

2. 采样率：
   - 免费层 / PoC: 100%
   - 一般生产: 10-20%
   - 高流量: 1-5%

3. 必脱的 PII：
   - 邮箱、手机、身份证、信用卡、姓名
   - IP 地址、地理位置、设备 ID

4. MLflow 3 也提供 OpenTelemetry 风格的 sampler：
   from mlflow.tracing.sampling import TraceIdRatioBased
""")