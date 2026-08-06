"""
阶段 1 示例 2: 用 sklearn 跑一个真实 ML 任务
============================================

目标：理解 MLflow 怎么记录真实 ML 流程
  - 用 sklearn 训练一个分类器
  - 演示 mlflow.sklearn.autolog()（一行代码自动记录一切）
  - 对比不同超参数下的 Run
  - 看模型签名（signature）、输入示例

运行：
  conda activate mlflow
  python scripts/01_basics/01b_sklearn_basics.py
"""

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np


def train_one_model(model, model_name: str, params: dict, X_train, X_test, y_train, y_test):
    """训练一个模型并记录到 MLflow，返回最终 accuracy。"""
    with mlflow.start_run(run_name=model_name) as run:
        # 1. 记录参数
        mlflow.log_params(params)

        # 2. 训练
        model.set_params(**params)
        model.fit(X_train, y_train)

        # 3. 预测
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        # 4. 计算并记录指标
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # 5. 记录模型（带 signature 和 input example，部署必备）
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train[:3]
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )

        # 6. 记录数据集信息（用于追溯）
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])

        # 7. 标记
        mlflow.set_tag("dataset", "iris")
        mlflow.set_tag("task", "classification")

        print(f"[{model_name}] accuracy={acc:.4f}  f1={f1:.4f}  run_id={run.info.run_id[:8]}")
        return acc, f1


def main():
    # 设置实验
    mlflow.set_experiment("01_sklearn_iris")

    # 加载数据
    print("加载 iris 数据集...")
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print("-" * 60)

    # 实验 1: 逻辑回归 + 强正则
    acc1, f1_1 = train_one_model(
        LogisticRegression(max_iter=200),
        "logreg_strong_reg",
        {"C": 0.1, "solver": "lbfgs"},
        X_train, X_test, y_train, y_test,
    )

    # 实验 2: 逻辑回归 + 弱正则
    acc2, f1_2 = train_one_model(
        LogisticRegression(max_iter=200),
        "logreg_weak_reg",
        {"C": 10.0, "solver": "lbfgs"},
        X_train, X_test, y_train, y_test,
    )

    # 实验 3: 随机森林（更深）
    acc3, f1_3 = train_one_model(
        RandomForestClassifier(random_state=42),
        "rf_deep",
        {"n_estimators": 200, "max_depth": 10, "min_samples_split": 2},
        X_train, X_test, y_train, y_test,
    )

    # 实验 4: 随机森林（更浅）
    acc4, f1_4 = train_one_model(
        RandomForestClassifier(random_state=42),
        "rf_shallow",
        {"n_estimators": 50, "max_depth": 3, "min_samples_split": 10},
        X_train, X_test, y_train, y_test,
    )

    # 汇总
    print("\n" + "=" * 60)
    print("结果对比：")
    print(f"  logreg_strong_reg: acc={acc1:.4f}")
    print(f"  logreg_weak_reg:   acc={acc2:.4f}")
    print(f"  rf_deep:           acc={acc3:.4f}")
    print(f"  rf_shallow:        acc={acc4:.4f}")
    print("=" * 60)
    print("\n下一步: mlflow ui --port 5000 查看")
    print("  - '01_sklearn_iris' 实验的 4 个 Run")
    print("  - 切换 metric（如 accuracy/f1）查看对比")
    print("  - 点开任一 Run，看 'Artifacts/model/' 下的 pickle 文件 + signature")


if __name__ == "__main__":
    main()
