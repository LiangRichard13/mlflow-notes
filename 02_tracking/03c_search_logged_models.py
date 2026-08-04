"""
阶段 3 示例 2: MLflow 3 新增 search_logged_models（跨实验搜索模型）
==================================================================

目标：
  - 用 mlflow.search_logged_models() 跨实验搜索模型
  - SQL 风格 filter_string，按 metrics/params/属性筛选
  - 找到最佳模型后加载推理

⚠️ 这是 MLflow 3 才有的新 API。MLflow 2 没有。

运行：
  conda activate mlflow
  python 02_tracking/03c_search_logged_models.py
"""

import mlflow
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


def train_and_log(name, model, params, X_train, X_test, y_train, y_test):
    """训练一个模型并 log 到 MLflow，返回 LoggedModel 信息。"""
    with mlflow.start_run(run_name=name) as run:
        # 训练
        model.set_params(**params)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))

        # 记录
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.set_tag("model_name", name)

        # MLflow 3 写法：name 而非 artifact_path
        mlflow.sklearn.log_model(model, name="classifier")

        return run.info.run_id, acc


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("03_search_demo")

    # 加载数据
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ============ 训练 5 个不同模型 ============
    print("训练 5 个不同模型...")
    experiments = [
        ("logreg-C0.1", LogisticRegression(max_iter=200), {"C": 0.1, "solver": "lbfgs"}),
        ("logreg-C1.0", LogisticRegression(max_iter=200), {"C": 1.0, "solver": "lbfgs"}),
        ("logreg-C10", LogisticRegression(max_iter=200), {"C": 10.0, "solver": "lbfgs"}),
        ("rf-shallow", RandomForestClassifier(random_state=42), {"n_estimators": 50, "max_depth": 3}),
        ("rf-deep", RandomForestClassifier(random_state=42), {"n_estimators": 200, "max_depth": 15}),
    ]

    results = []
    for name, model, params in experiments:
        run_id, acc = train_and_log(name, model, params, X_train, X_test, y_train, y_test)
        results.append({"name": name, "run_id": run_id, "accuracy": acc, **params})
        print(f"  {name}: accuracy={acc:.4f}")

    # ============ search_logged_models 演示 ============
    # 先拿 experiment_id（这个 API 接收的是 id 不是 name）
    exp_id = mlflow.get_experiment_by_name("03_search_demo").experiment_id
    print("\n" + "=" * 60)
    print("🔍 MLflow 3 新 API: search_logged_models()")
    print("=" * 60)

    def get_metric(metrics_list, key):
        """从 metrics 列表里取某个 key 的值（metrics 列实际是 list[Metric]）"""
        for m in metrics_list:
            if m.key == key:
                return m.value
        return None

    def show(models, cols=None):
        """格式化打印模型列表"""
        rows = []
        for m in models:
            rows.append({
                "name": m.name,
                "model_id": m.model_id[:12],
                "accuracy": get_metric(m.metrics, "accuracy"),
                "n_est": m.params.get("n_estimators", "-"),
                "max_d": m.params.get("max_depth", "-"),
                "C": m.params.get("C", "-"),
            })
        df = pd.DataFrame(rows)
        if cols:
            df = df[cols]
        print(df.to_string(index=False))

    # 查询 1: 所有 accuracy > 0.95 的模型
    print("\n[1] accuracy > 0.95 的模型：")
    models = mlflow.search_logged_models(
        experiment_ids=[exp_id],
        filter_string="metrics.accuracy > 0.95",
        order_by=[{"field_name": "metrics.accuracy", "ascending": False}],
        max_results=10,
        output_format="list",  # 用 list 格式方便遍历
    )
    show(models)

    # 查询 2: 只看名字含特定字符的模型
    print("\n[2] 模型名包含 'classifier' 且 accuracy > 0.95：")
    models = mlflow.search_logged_models(
        experiment_ids=[exp_id],
        filter_string="name = 'classifier' AND metrics.accuracy > 0.95",
        order_by=[{"field_name": "metrics.accuracy", "ascending": False}],
        output_format="list",
    )
    show(models, cols=["name", "accuracy", "n_est", "max_d", "C"])

    # 查询 3: 复杂条件：accuracy > 0.95 AND n_estimators = 50
    # ⚠️ params 是字符串，数值比较（< > <= >=）只对 metrics 有效
    #    params 只支持 =, !=, LIKE, ILIKE, IN, NOT IN
    print("\n[3] accuracy > 0.95 且 n_estimators = 50：")
    models = mlflow.search_logged_models(
        experiment_ids=[exp_id],
        filter_string="metrics.accuracy > 0.95 AND params.n_estimators = '50'",
        order_by=[{"field_name": "metrics.accuracy", "ascending": False}],
        output_format="list",
    )
    show(models, cols=["name", "accuracy", "n_est", "max_d", "C"])

    # ============ 找到最佳模型并加载 ============
    print("\n" + "=" * 60)
    print("🏆 找到最佳模型并加载：")
    print("=" * 60)

    models = mlflow.search_logged_models(
        experiment_ids=[exp_id],
        order_by=[{"field_name": "metrics.accuracy", "ascending": False}],
        max_results=1,
        output_format="list",
    )
    best = models[0]
    print(f"最佳模型 name: {best.name}")
    print(f"  accuracy: {get_metric(best.metrics, 'accuracy'):.4f}")
    print(f"  model_id: {best.model_id}")

    # 加载模型（MLflow 3 写法：用 model_id）
    model_uri = f"models:/{best.model_id}"
    loaded = mlflow.sklearn.load_model(model_uri)
    pred = loaded.predict(X_test.head(3))
    print(f"  加载成功，类型: {type(loaded).__name__}")
    print(f"  推理测试: {pred}")

    # ============ filter_string 语法速查 ============
    print("\n" + "=" * 60)
    print("📖 filter_string 语法速查（SQL 风格）")
    print("=" * 60)
    print("""
    # 比较运算
    metrics.accuracy > 0.9               # ✓ 数值比较 OK
    metrics.loss <= 0.1                  # ✓
    params.lr = 0.01                     # ⚠️ params 是字符串，只支持 = != LIKE 等

    # 逻辑运算
    metrics.accuracy > 0.9 AND params.max_depth = '5'    # params 用 = 不用 <=
    metrics.accuracy > 0.95 OR name = 'classifier'

    # 排序（注意是 list[dict]，不是 list[str]）
    order_by=[{"field_name": "metrics.accuracy", "ascending": False}]
    """)


if __name__ == "__main__":
    main()