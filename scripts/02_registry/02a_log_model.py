"""
阶段 2 示例 1: 训练 + 记录带签名和输入示例的 sklearn Pipeline
================================================================

目标：
  - 理解 MLflow Model 格式（MLmodel YAML + flavors）
  - 用 infer_signature() 自动推断签名
  - 记录 input_example（部署时用于 schema 校验）
  - 用 mlflow.sklearn.log_model() 的 MLflow 3 写法（name 而非 artifact_path）

运行：
  conda activate mlflow
  python scripts/02_registry/02a_log_model.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _paths

import mlflow
from mlflow.models import infer_signature
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np


def main():
    mlflow.set_experiment("02_model_registry")

    # 加载数据（Wine：178 样本 / 13 特征 / 3 类）
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 训练 Pipeline（StandardScaler + RandomForest）
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"模型性能: accuracy={acc:.4f}, f1={f1:.4f}")

    # ============ 记录到 MLflow ============
    with mlflow.start_run(run_name="wine-rf-v1") as run:
        # 1) 记录参数/指标
        mlflow.log_params({
            "n_estimators": 200,
            "max_depth": 10,
            "scaler": "standard",
            "n_features": X_train.shape[1],
        })
        mlflow.log_metrics({"accuracy": acc, "f1": f1})

        # 2) 推断签名（输入 schema + 输出 schema）
        # 签名让部署服务能自动校验请求格式
        signature = infer_signature(X_train, pipe.predict(X_train))

        # 3) input_example（部署时 fallback 用，也用于自动生成 signature）
        input_example = X_train.head(3)

        # 4) 记录模型（MLflow 3 写法：name 而非 artifact_path）
        mlflow.sklearn.log_model(
            pipe,
            name="wine-classifier",  # ⚠️ MLflow 3: 用 name，不是 artifact_path
            signature=signature,
            input_example=input_example,
        )

        # 5) 标记
        mlflow.set_tag("pipeline", "StandardScaler + RandomForest")
        mlflow.set_tag("dataset", "sklearn.wine")

        run_id = run.info.run_id
        print(f"\n✓ Run 已记录: {run_id}")
        print(f"  模型 URI: runs:/{run_id}/wine-classifier")

    print("\n" + "=" * 60)
    print("下一步：02b_register_alias.py")
    print("=" * 60)


if __name__ == "__main__":
    main()