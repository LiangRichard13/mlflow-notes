"""
阶段 2 示例 2: 注册模型 + 用别名（替代已废弃的 Stage）
====================================================

目标：
  - 注册一个模型到 Model Registry
  - 用别名（champion/challenger）替代已废弃的 stage
  - 学会 set/get/transition Registered Model 的版本元数据

⚠️ MLflow 3 关键变化：
  - Stages（Staging/Production/Archived）已废弃，用 Aliases 替代
  - Champion/Challenger 别名模式：冠军在用、挑战者在测
  - 别名可以热切换，零停机回滚

运行：
  conda activate mlflow
  # 先确保 mlflow.db backend 已启动（如没启动见 02_tracking/03_tracking_server.md）
  python scripts/02_registry/02b_register_alias.py
"""

import mlflow
from mlflow import MlflowClient


REGISTERED_NAME = "WineQualityClassifier"


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("02_model_registry")

    client = MlflowClient()

    # ============ 1. 找到上一个脚本记录的 Run ============
    runs = mlflow.search_runs(
        experiment_names=["02_model_registry"],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError("没找到 02a 的 Run，请先跑 02a_log_model.py")

    latest_run_id = runs.iloc[0].run_id
    model_uri = f"runs:/{latest_run_id}/wine-classifier"
    print(f"准备注册: {model_uri}")

    # ============ 2. 注册模型 ============
    # 同一个 REGISTERED_NAME 可以注册多个版本（v1, v2, v3...）
    result = mlflow.register_model(model_uri, REGISTERED_NAME)
    version = result.version
    print(f"✓ 已注册为 {REGISTERED_NAME} v{version}")
    print(f"  source: {result.source}")
    print(f"  run_id: {result.run_id}")

    # ============ 3. 设置别名（替代已废弃的 Stage）============
    # champion: 当前生产用
    # challenger: 正在测试、准备替换 champion
    # archived: 已弃用但保留
    client.set_registered_model_alias(REGISTERED_NAME, "champion", version=version)
    print(f"✓ 已设置 champion alias → v{version}")

    # ============ 4. 加描述 ============
    client.update_model_version(
        name=REGISTERED_NAME,
        version=version,
        description="StandardScaler + RandomForest(n=200, depth=10) on Wine dataset. accuracy=1.0",
    )

    # ============ 5. 查询所有版本和别名 ============
    print("\n" + "=" * 60)
    print(f"📋 {REGISTERED_NAME} 当前状态：")
    print("=" * 60)

    # 列所有版本
    versions = client.search_model_versions(f"name='{REGISTERED_NAME}'")
    for v in versions:
        aliases = [a.alias for a in v.aliases] if hasattr(v, "aliases") else []
        print(f"  v{v.version} | run={v.run_id[:8]} | aliases={aliases} | status={v.status}")

    # 列所有别名映射
    aliases = client.get_registered_model(REGISTERED_NAME).aliases
    print(f"\n别名映射: {aliases}")

    print("\n" + "=" * 60)
    print("下一步：02c_load_predict.py")
    print("=" * 60)


if __name__ == "__main__":
    main()