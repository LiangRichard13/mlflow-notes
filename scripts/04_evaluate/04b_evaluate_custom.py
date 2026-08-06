"""
阶段 4 示例 2: 自定义指标 + validate_evaluation_results
====================================================

目标：
  - 用 mlflow.metrics.make_metric() 写自定义指标（如"高价值客户加权 accuracy"）
  - 用 mlflow.validate_evaluation_results() 对比两个模型（MLflow 3 新写法）
  - 替代 MLflow 2 的 baseline_model 参数

运行：
  conda activate mlflow
  python scripts/04_evaluate/04b_evaluate_custom.py
"""

import mlflow
from mlflow.metrics import make_metric, EvaluationMetric
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("04_evaluate_custom")

    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    eval_df = X_test.copy()
    eval_df["target"] = y_test

    # ============ 1. 训练两个模型 ============
    model_a = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    model_a.fit(X_train, y_train)

    model_b = LogisticRegression(C=0.5, max_iter=500, random_state=42)
    model_b.fit(X_train, y_train)

    # ============ 2. 自定义指标：每类加权（class_0 比其他重要）============
    # 业务场景：class_0 是高价值客户，识别错了代价高
    CLASS_WEIGHTS = {0: 5.0, 1: 1.0, 2: 1.0}   # class_0 权重 5 倍

    def weighted_accuracy_fn(predictions, targets):
        """自定义指标：按 class_weights 加权的 accuracy。"""
        preds = np.asarray(predictions)
        targs = np.asarray(targets)
        total = 0.0
        correct = 0.0
        for p, t in zip(preds, targs):
            w = CLASS_WEIGHTS.get(int(t), 1.0)
            total += w
            if p == t:
                correct += w
        return correct / total

    # 用 make_metric 包装（MLflow 3 推荐写法）
    custom_metric = make_metric(
        eval_fn=weighted_accuracy_fn,
        greater_is_better=True,
        name="weighted_accuracy_v1",
    )

    # ============ 3. 评估模型 A ============
    result_a = None
    with mlflow.start_run(run_name="model-A-RF") as run:
        mlflow.sklearn.log_model(model_a, name="model")
        result_a = mlflow.models.evaluate(
            model=f"runs:/{run.info.run_id}/model",
            data=eval_df,
            targets="target",
            model_type="classifier",
            extra_metrics=[custom_metric],   # ← MLflow 3 用 extra_metrics
        )
        print(f"\n[模型 A: RandomForest]")
        print(f"  accuracy_score: {result_a.metrics['accuracy_score']:.4f}")
        print(f"  weighted_accuracy_v1: {result_a.metrics.get('weighted_accuracy_v1', 0):.4f}")

    # ============ 4. 评估模型 B ============
    result_b = None
    with mlflow.start_run(run_name="model-B-LR") as run:
        mlflow.sklearn.log_model(model_b, name="model")
        result_b = mlflow.models.evaluate(
            model=f"runs:/{run.info.run_id}/model",
            data=eval_df,
            targets="target",
            model_type="classifier",
            extra_metrics=[custom_metric],
        )
        print(f"\n[模型 B: LogisticRegression]")
        print(f"  accuracy_score: {result_b.metrics['accuracy_score']:.4f}")
        print(f"  weighted_accuracy_v1: {result_b.metrics.get('weighted_accuracy_v1', 0):.4f}")

    # ============ 5. validate_evaluation_results（MLflow 3 新 API）============
    # 验证模型 B 是不是真的比 A 好
    print("\n" + "=" * 60)
    print("🆚 用 validate_evaluation_results 比较两个模型（MLflow 3 写法）：")
    print("=" * 60)

    from mlflow.models import MetricThreshold

    # 定义阈值：candidate（模型 B）的指标必须满足
    # - 绝对值 >= threshold
    # - 相对 baseline 的提升 >= min_absolute_change / min_relative_change（可选）
    thresholds = {
        "accuracy_score": MetricThreshold(
            threshold=0.9,               # 最低 0.9（绝对值）
            greater_is_better=True,
            # min_absolute_change=0.0 表示允许和 baseline 持平或更差
        ),
        "weighted_accuracy_v1": MetricThreshold(
            threshold=0.85,
            greater_is_better=True,
        ),
    }

    try:
        mlflow.validate_evaluation_results(
            validation_thresholds=thresholds,
            candidate_result=result_b,
            baseline_result=result_a,
        )
        print("✓ 模型 B 通过验证（在阈值范围内）")
    except Exception as e:
        print(f"✗ 验证失败：{type(e).__name__}: {e}")

    # ============ 6. 与 MLflow 2 对比 ============
    print("\n" + "=" * 60)
    print("💡 MLflow 2 vs 3 评估 API 差异：")
    print("=" * 60)
    print("""
    MLflow 2:
        result = mlflow.evaluate(
            model, data, targets, model_type,
            baseline_model=baseline_uri,    # ← 旧写法
        )

    MLflow 3:
        result_a = mlflow.evaluate(...)   # 旧模型
        result_b = mlflow.evaluate(...)   # 新模型
        mlflow.validate_evaluation_results(
            candidate_result=result_b,
            baseline_result=result_a,      # ← 新写法
        )
    """)


if __name__ == "__main__":
    main()