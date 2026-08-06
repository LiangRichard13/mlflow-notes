"""
阶段 3 示例 1: 数据集血缘追踪
==============================

目标：
  - 用 mlflow.data.from_pandas() 把数据集注册到 MLflow
  - 用 mlflow.log_input() 记录"这个 Run 用了哪个数据集"
  - 理解为什么这很重要：可追溯、可复现、可审计

运行：
  conda activate mlflow
  python scripts/03_tracking/03b_dataset_lineage.py

⚠️ 需要 mlflow.db 后端（sqlite 即可），不需要远程 server。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _paths

import mlflow
import pandas as pd
import tempfile
import os
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main():
    # 本地 sqlite 后端（不需要启 server）
    mlflow.set_experiment("03_dataset_lineage")

    # ============ 1. 准备数据集 ============
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["target"] = wine.target
    df["target_name"] = df["target"].map(
        dict(enumerate(wine.target_names))
    )

    # 把数据写到临时文件，模拟"数据集来自某个文件"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as f:
        df.to_csv(f.name, index=False)
        data_path = f.name
    print(f"数据集已保存: {data_path} ({df.shape})")

    # ============ 2. 注册数据集到 MLflow ============
    # mlflow.data.from_pandas() 创建一个 Dataset 对象
    # source: 数据来源（路径/URL/库名），记录在元数据里
    # name: 数据集名字（自己起）
    # targets: 目标列名（用于评估时知道哪列是 label）
    dataset = mlflow.data.from_pandas(
        df,
        source=data_path,
        name="wine_dataset",
        targets="target",
    )
    print(f"\n数据集元数据:")
    print(f"  name: {dataset.name}")
    print(f"  source: {dataset.source}")
    print(f"  digest: {dataset.digest}")   # 哈希，用于检测数据集是否变了
    print(f"  schema: {dataset.schema}")

    # ============ 3. 训练 + 记录 Run + 关联数据集 ============
    X = df[wine.feature_names]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="rf-with-dataset") as run:
        # 训练
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        mlflow.log_metric("accuracy", acc)

        # 关键：把数据集关联到这个 Run
        # context 可以是 "training" / "testing" / "validation"
        mlflow.log_input(dataset, context="training")

        # 再加一个测试集
        test_df = X_test.copy()
        test_df["target"] = y_test
        test_dataset = mlflow.data.from_pandas(
            test_df,
            source=data_path,
            name="wine_test_split",
            targets="target",
        )
        mlflow.log_input(test_dataset, context="testing")

        # ⚠️ 演示数据集变更检测
        # 如果后面有人改了数据集，digest 会变，MLflow 会提醒
        mlflow.set_tag("data_snapshot", dataset.digest)

        print(f"\n✓ Run 已记录: {run.info.run_id[:8]}")
        print(f"  accuracy: {acc:.4f}")
        print(f"  dataset digest: {dataset.digest}")
        print(f"  关联的数据集: training + testing")

    # ============ 4. 反向追溯：哪些 Run 用了某个数据集？============
    print("\n" + "=" * 60)
    print("📋 反向追溯：通过 Client API 找 Run 关联的数据集")
    print("=" * 60)

    # MLflow 3: 数据集信息存储在 runs 的 datasets 字段里
    # 通过 client.get_run(run_id) 可以拿到
    runs = mlflow.search_runs(
        experiment_names=["03_dataset_lineage"],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs.empty:
        latest_run_id = runs.iloc[0].run_id
        client = mlflow.MlflowClient()
        run_detail = client.get_run(latest_run_id)

        print(f"Run {latest_run_id[:8]} 的关联数据集：")
        # run_detail.inputs.dataset_inputs 是关联的数据集列表
        for ds_input in run_detail.inputs.dataset_inputs:
            ds = ds_input.dataset
            # tags 是 list[InputDatasetTag]，需要遍历找
            ctx = "-"
            for t in (ds_input.tags or []):
                if t.key == "mlflow.dataset_context":
                    ctx = t.value
            print(f"  - name: {ds.name}")
            print(f"    digest: {ds.digest}")
            print(f"    source: {ds.source}")
            print(f"    context: {ctx}")

    # ============ 5. 展示数据集重要性 ============
    print("\n" + "=" * 60)
    print("💡 数据集血缘的价值：")
    print("=" * 60)
    print("""
    1. 可追溯：模型出问题时，能立刻找到它训练用的数据
    2. 可复现：用同一个 digest + source 就能重新生成数据集
    3. 可审计：监管要求"模型用了什么数据"时一键回答
    4. 防"偷偷换数据"：digest 变了会立刻发现
    """)

    # 清理临时文件
    os.unlink(data_path)


if __name__ == "__main__":
    main()