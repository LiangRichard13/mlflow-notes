"""
阶段 4 示例 1: mlflow.models.evaluate - 内置评估
================================================

目标：
  - 用 mlflow.models.evaluate() 对模型做内置评估
  - 自动生成混淆矩阵、ROC 曲线等可视化（写到 artifact）
  - model_type='classifier' vs 'regressor'

运行：
  conda activate mlflow
  python 02_tracking/04a_evaluate_basics.py
"""

import mlflow
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("04_evaluate")

    # 准备数据
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 训练一个模型
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # ============ 用 mlflow.evaluate 做综合评估 ============
    # 把测试数据合成一个 DataFrame（包含 features + label）
    eval_df = X_test.copy()
    eval_df["target"] = y_test

    with mlflow.start_run(run_name="evaluate-baseline") as run:
        # 先记录模型（mlflow.evaluate 要 model_uri）
        mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=X_train.head(3),
        )
        model_uri = f"runs:/{run.info.run_id}/model"

        # 一键评估（自动算 accuracy/F1/precision/recall + 画图）
        result = mlflow.models.evaluate(
            model=model_uri,
            data=eval_df,
            targets="target",                # 哪列是 label
            model_type="classifier",         # 或 "regressor"
            evaluators=["default"],
        )

        print(f"\n✓ 评估完成，指标：")
        for metric_name, metric_value in result.metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")

        print(f"\n✓ 可视化产物（写到 Artifacts/eval/）:")
        for artifact_name in result.artifacts:
            print(f"  - {artifact_name}")

    print("\n" + "=" * 60)
    print("💡 在 UI 的 Run 详情页 Artifacts → eval/ 看生成的图：")
    print("   confusion_matrix.png, roc_curve.png 等")
    print("=" * 60)
    print("下一步：04b_evaluate_custom_metric.py")


if __name__ == "__main__":
    main()