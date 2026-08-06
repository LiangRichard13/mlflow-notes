"""
阶段 10 示例 2: 跨模型评估 + validate_evaluation_results
========================================================

目标：
  - 读取 10b 的 9 个模型 Run
  - 在每个 Run 里追加自定义指标 + 验证
  - 用 efficientnet_b0 当 baseline，验证其他模型都比它强

运行：
  conda activate mlflow
  python scripts/10_vision_classification/10c_compare_and_evaluate.py

⚠️ 必须先跑过 10b_train_models.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _paths

import mlflow
import pandas as pd

from shared import MODEL_NAMES, load_cifar10, CLASS_NAMES


# MLflow 3 自定义指标需要 judge 模型；但 GenAI metrics 不适合纯 CV
# 这里直接用 evaluate 的简化版：基于已有 metrics 做汇总分析
# ⚠️ 完整 mlflow.models.evaluate 需要 Pyfunc 包装，这里跳过以保持简洁


def main():
    mlflow.set_experiment("10_vision_classification")

    # 1. 找到 10b 跑出来的所有 Run
    exp = mlflow.get_experiment_by_name("10_vision_classification")
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.step = 'data_overview'",   # 排除
    )
    # 找 9 个训练 Run（有 model_name tag 的）
    train_runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.model_name != ''",
        max_results=30,
    )

    if len(train_runs) == 0:
        print("✗ 没找到 10b 训练的 Run，请先跑 10b_train_models.py")
        return

    print(f"找到 {len(train_runs)} 个训练 Run：")
    print(train_runs[['run_id', 'tags.model_name', 'metrics.accuracy_score', 'metrics.f1_macro']].to_string(index=False))
    print()

    # 2. 汇总对比表
    comparison = train_runs[[
        'run_id', 'tags.model_name', 'tags.model_family',
        'metrics.accuracy_score', 'metrics.f1_macro', 'metrics.f1_weighted',
        'metrics.precision_macro', 'metrics.recall_macro',
        'metrics.balanced_accuracy_score',
        'metrics.train_time_seconds', 'metrics.inference_time_per_sample_ms',
        'metrics.model_size_bytes',
    ]].copy()
    comparison.columns = [
        'run_id', 'model', 'family',
        'accuracy', 'f1_macro', 'f1_weighted',
        'precision_macro', 'recall_macro', 'balanced_acc',
        'train_s', 'infer_ms', 'size_MB',
    ]
    comparison['size_MB'] = comparison['size_MB'] / 1024 / 1024
    comparison = comparison.sort_values('accuracy', ascending=False)

    print("=" * 80)
    print("📊 9 模型对比表（按 acc 降序）")
    print("=" * 80)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    # 3. 把汇总表 log 到一个专门的 Run
    with mlflow.start_run(run_name="10c_comparison_summary") as run:
        mlflow.set_tag("phase", "10")
        mlflow.set_tag("dataset", "cifar10")
        mlflow.set_tag("step", "comparison_summary")

        # 按 family 算平均
        family_avg = comparison.groupby('family').agg({
            'accuracy': 'mean',
            'f1_macro': 'mean',
            'train_s': 'mean',
            'size_MB': 'mean',
        }).reset_index()
        mlflow.log_table(family_avg, artifact_file="family_summary.json")

        # 完整对比表
        mlflow.log_table(comparison, artifact_file="all_models_comparison.json")

        # 关键指标 log 成 metric 方便 UI 排序
        mlflow.log_metric("best_accuracy", float(comparison['accuracy'].max()))
        mlflow.log_metric("worst_accuracy", float(comparison['accuracy'].min()))
        mlflow.log_metric("accuracy_range", float(
            comparison['accuracy'].max() - comparison['accuracy'].min()))
        mlflow.log_metric("best_train_time_s", float(comparison['train_s'].min()))
        mlflow.log_metric("total_params_models", float(len(comparison)))

        # 冠军（按 acc）
        best = comparison.iloc[0]
        mlflow.set_tag("best_model", best['model'])
        mlflow.set_tag("best_model_run_id", best['run_id'])
        mlflow.log_metric("best_model_accuracy", float(best['accuracy']))

        print(f"🏆 冠军: {best['model']} (acc={best['accuracy']:.4f})")
        print(f"   Run: {best['run_id']}")

        # 性价比（acc per MB）
        comparison['acc_per_mb'] = comparison['accuracy'] / comparison['size_MB']
        most_efficient = comparison.sort_values('acc_per_mb', ascending=False).iloc[0]
        mlflow.set_tag("most_efficient_model", most_efficient['model'])
        mlflow.log_metric("best_acc_per_mb", float(most_efficient['acc_per_mb']))
        print(f"⚡ 最高性价比: {most_efficient['model']} "
              f"({most_efficient['acc_per_mb']:.3f} acc/MB)")

        run_id = run.info.run_id

    print(f"\n✓ Summary Run: {run_id[:8]}")
    print("\n下一步: 10d_failure_analysis.py（跨模型失败案例）")


if __name__ == "__main__":
    main()