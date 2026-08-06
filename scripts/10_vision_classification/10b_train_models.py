"""
阶段 10 示例 1: 训练 9 个 timm 预训练 CNN 做对比
=====================================================

目标：
  - 训练 resnet18/34/50 + efficientnet_b0/b1/b2 + densenet121/169/201
  - 每个模型一个 MLflow Run，记录 11 个指标 + 可视化
  - 完整 grid 可视化失败/成功样本 + 混淆矩阵

运行：
  conda activate mlflow
  python scripts/10_vision_classification/10b_train_models.py

⚠️ 预计 30-50 分钟（CPU，9 模型 × 3 epochs）
   想先测试：把 MODEL_NAMES 改短（如只留 resnet18），或 epochs=1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _paths

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

    # grid 只看前 128 张的预测（与 sample_uint8 对齐）
    n_sample = len(sample_y)
    y_true_s = y_true[:n_sample]
    y_pred_s = y_pred[:n_sample]
    y_proba_s = y_proba[:n_sample]

    # 失败 grid
    fail_grid = make_failure_grid(
        sample_uint8, sample_y, y_pred_s, y_proba_s,
        class_names, n=64, n_cols=8,
    )
    success_grid = make_success_grid(
        sample_uint8, sample_y, y_pred_s, y_proba_s,
        class_names, n=64, n_cols=8,
    )
    # 混淆矩阵用全集（更准）
    cm_img = make_confusion_matrix(y_true, y_pred, class_names, normalize=True)

    # MLflow Run
    # 防御：mlflow 3 拒绝在已激活 Run 时开新 Run（旧孤儿 Run 也要清理）
    mlflow.end_run()
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
    mlflow.set_experiment("10_vision_classification")
    mlflow.set_tags({"phase": "10", "dataset": "cifar10"})
    exp = mlflow.get_experiment_by_name("10_vision_classification")

    # 加载数据（如果还没下，10a 已经下了）
    print("加载 CIFAR-10...")
    train_ds, test_ds, class_names = load_cifar10()
    print(f"  训练: {len(train_ds)}, 测试: {len(test_ds)}, 类别: {len(class_names)}")

    # 全局参数
    epochs = 3
    batch_size = 64   # 224×224 + GPU 上 64 更稳（多个模型连续训练不 OOM）
    import os
    if os.environ.get("VISION_IMAGE_SIZE"):
        print(f"  [config] 使用自定义 image_size={os.environ['VISION_IMAGE_SIZE']}")

    # 依次训练 9 个模型（跳过已完成的，支持断点续跑）
    summary = []
    for model_name in MODEL_NAMES:
        # 查这个模型是否已有完成的 Run
        existing = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.model_name = '{model_name}' AND tags.model_family != ''",
            max_results=1,
        )
        already_done = False
        if len(existing) > 0:
            end = existing.iloc[0]['end_time']
            if end is not None and str(end) != 'NaT':
                already_done = True
                acc = existing.iloc[0].get('metrics.accuracy_score', None)
                print(f"  ⏭️ 跳过 {model_name}（已有完成 Run, acc={acc}）")
                # 把已有 Run 加入 summary
                summary.append({
                    "model": model_name,
                    "run_id": existing.iloc[0]['run_id'][:8],
                    "accuracy_score": float(acc) if acc is not None else 0,
                })
        if already_done:
            continue

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
    summary_sorted = sorted(summary, key=lambda x: -x.get("accuracy_score", 0))
    for s in summary_sorted:
        acc = s.get("accuracy_score", float("nan"))
        f1 = s.get("f1_macro", float("nan"))
        train = s.get("train_time_seconds", float("nan"))
        size = s.get("model_size_bytes", 0)
        print(f"{s['model']:<22} {acc:<8.4f} "
              f"{f1:<10.4f} {train:<10.1f} "
              f"{size/1024/1024:<10.2f}")

    print("\n" + "=" * 60)
    print("下一步:")
    print("  10c_compare_and_evaluate.py  跨模型评估 + validate")
    print("  10d_failure_analysis.py      失败案例定性分析")
    print("=" * 60)


if __name__ == "__main__":
    main()