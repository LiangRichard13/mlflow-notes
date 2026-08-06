"""
阶段 2 示例 3: 用别名加载模型 + 推理
====================================

目标：
  - 用 models:/name@alias 加载模型
  - 体验"零停机"切换：换 alias 就切换生产模型
  - 用 mlflow.sklearn.load_model 直接拿可调用对象

运行：
  conda activate mlflow
  python scripts/02_registry/02c_load_predict.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _paths

import mlflow
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


REGISTERED_NAME = "WineQualityClassifier"


def main():
    # ============ 1. 用别名加载 ============
    model_uri = f"models:/{REGISTERED_NAME}@champion"
    print(f"加载模型: {model_uri}")

    model = mlflow.sklearn.load_model(model_uri)
    print(f"✓ 模型类型: {type(model).__name__}")
    print(f"  Pipeline steps: {[s[0] for s in model.steps]}")

    # ============ 2. 准备测试数据 ============
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ============ 3. 推理 ============
    sample = X_test.head(5)
    preds = model.predict(sample)
    pred_labels = [f"class_{p}" for p in preds]
    print(f"\n前 5 个预测: {pred_labels}")
    print(f"真实标签:    {[f'class_{y}' for y in y_test[:5]]}")

    # ============ 4. 模拟"切换生产模型"演示 ============
    print("\n" + "=" * 60)
    print("💡 演示别名热切换：")
    print("=" * 60)
    print("""
    假设新模型已注册为 v2 并测试通过：

        client.set_registered_model_alias("WineQualityClassifier", "champion", version=2)

    此时所有 'models:/WineQualityClassifier@champion' 加载都会自动用 v2，
    无需重启服务。这就是 alias 替代 stage 的核心好处。
    """)


if __name__ == "__main__":
    main()