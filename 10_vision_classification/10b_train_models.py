"""
阶段 10 示例 1: 训练 9 个 timm 预训练 CNN 做对比
=====================================================

目标：
  - 训练 resnet18/34/50 + efficientnet_b0/b1/b2 + densenet121/169/201
  - 每个模型一个 MLflow Run，记录 11 个指标 + 可视化
  - 完整 grid 可视化失败/成功样本 + 混淆矩阵

运行：
  conda activate mlflow
  python 10_vision_classification/10b_train_models.py

⚠️ 预计 30-50 分钟（CPU，9 模型 × 3 epochs）
   想先测试：把 MODEL_NAMES 改短（如只留 resnet18），或 epochs=1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from shared import (
    MODEL_NAMES, load_cifar10, train_and_evaluate,
    tensor_to_uint8_images, make_grid,
    make_failure_grid, make_success_grid, make_confusion_matrix,
    CLASS_NAMES, model_family, log_timm_state_dict,
)


def train_and_log_one(model_name, train_ds, test_ds, class_names,
                      epochs=3, batch_size=128):
    """训练单个模型 + 完整 MLflow 记录。"""
    family = model_family(model_name)
    print(f"\n{'='*60}")
    print(f"[{model_name}] 训练 ({family})")
    print(f"{'='*60}")

    # 训练 + 评估
    model, metrics, preds, state_bytes = train_and_evaluate(
        model_name, train_ds, test_ds, class_names,
        epochs=epochs, batch_size=batch_size,
    )
    print(f"  ✓ acc={metrics['accuracy_score']:.4f}  "
          f"f1_macro={metrics['f1_macro']:.4f}  "
          f"训练 {metrics['train_time_seconds']:.1f}s")

    # 准备样本可视化（用测试集的前 128 张，足够覆盖失败案例）
    sample_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                                num_workers=0)
    sample_x, sample_y = next(iter(sample_loader))
    sample_uint8 = tensor_to_uint8_images(sample_x)
    y_true = preds["y_true"]
    y_pred = preds["y_pred"]
    y_proba = preds["y_proba"]

    # 选失败样本和成功样本（从全集而非仅 128 张里选）
    # 失败 grid
    fail_grid = make_failure_grid(
        sample_uint8, sample_y[:len(y_true)], y_pred, y_proba,
        class_names, n=64, n_cols=8,
    )
    success_grid = make_success_grid(
        sample_uint8, sample_y[:len(y_true)], y_pred, y_proba,
        class_names, n=64, n_cols=8,
    )
    cm_img = make_confusion_matrix(y_true, y_pred, class_names, normalize=True)

    # MLflow Run
    with mlflow.start_run(run_name=f"cifar10_{model_name}") as run:
        # Tags
        mlflow.set_tag("phase", "10")
        mlflow.set_tag("dataset", "cifar10")
        mlflow.set_tag("model_family", family)
        mlflow.set_tag("model_name", model_name)

        # Params（模型结构 + 训练超参）
        mlflow.log_params({
            "model_name": model_name,
            "model_family": family,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr_backbone": 1e-4,
            "lr_head": 1e-3,
            "optimizer": "AdamW",
            "image_size": 224,
            "pretrained": True,
        })

        # Metrics
        mlflow.log_metrics(metrics)

        # Artifacts：state_dict + 可视化
        log_timm_state_dict(state_bytes, run.info.run_id, "model")
        if fail_grid is not None:
            mlflow.log_image(fail_grid, artifact_file="failures_grid.png")
        if success_grid is not None:
            mlflow.log_image(success_grid, artifact_file="successes_grid.png")
        mlflow.log_image(cm_img, artifact_file="confusion_matrix.png")

        run_id = run.info.run_id

    print(f"  ✓ Run {run_id[:8]} log 完成")
    return run_id, metrics


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("10_vision_classification")
    mlflow.set_tags({"phase": "10", "dataset": "cifar10"})

    # 加载数据（如果还没下，10a 已经下了）
    print("加载 CIFAR-10...")
    train_ds, test_ds, class_names = load_cifar10()
    print(f"  训练: {len(train_ds)}, 测试: {len(test_ds)}, 类别: {len(class_names)}")

    # 全局参数
    epochs = 3
    batch_size = 128

    # 依次训练 9 个模型
    summary = []
    for model_name in MODEL_NAMES:
        run_id, metrics = train_and_log_one(
            model_name, train_ds, test_ds, class_names,
            epochs=epochs, batch_size=batch_size,
        )
        summary.append({
            "model": model_name,
            "run_id": run_id[:8],
            **metrics,
        })

    # 汇总打印
    print("\n" + "=" * 60)
    print("📊 9 模型对比汇总（按 acc 降序）:")
    print("=" * 60)
    print(f"{'model':<22} {'acc':<8} {'f1_macro':<10} {'train(s)':<10} {'size(MB)':<10}")
    print("-" * 60)
    summary_sorted = sorted(summary, key=lambda x: -x["accuracy_score"])
    for s in summary_sorted:
        print(f"{s['model']:<22} {s['accuracy_score']:<8.4f} "
              f"{s['f1_macro']:<10.4f} {s['train_time_seconds']:<10.1f} "
              f"{s['model_size_bytes']/1024/1024:<10.2f}")

    print("\n" + "=" * 60)
    print("下一步:")
    print("  10c_compare_and_evaluate.py  跨模型评估 + validate")
    print("  10d_failure_analysis.py      失败案例定性分析")
    print("=" * 60)


if __name__ == "__main__":
    main()