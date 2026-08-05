"""
阶段 10 示例 3: 跨模型失败案例定性分析
========================================

目标：
  - 加载 9 个模型的预测结果
  - 找出"所有模型都错"的最难样本（hard examples）
  - 分析各模型族（ResNet / EfficientNet / DenseNet）失败模式差异
  - 演示 ensemble 投票的效果
  - 每个样本的失败 metadata CSV

运行：
  conda activate mlflow
  python 10_vision_classification/10d_failure_analysis.py

⚠️ 必须先跑过 10b_train_models.py（会在 mlflow.db 里查 9 个 Run）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow
import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared import (
    MODEL_NAMES, load_cifar10,
    tensor_to_uint8_images, make_grid, CLASS_NAMES, model_family,
    create_model, predict_full,
)


def load_all_predictions_from_artifacts(exp, model_names):
    """每个模型 Run 都保存了 state_dict artifact。
    这里直接重新加载模型，对测试集重新预测（确保数据一致）。
    """
    train_ds, test_ds, class_names = load_cifar10()
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False,
                               num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [device] {device}")

    preds_per_model = {}
    y_true = None
    for name in model_names:
        run = find_run_for_model(exp, name)
        if run is None:
            print(f"  ⚠️ 找不到 {name} 的 Run，跳过")
            continue

        # 下载 state_dict（文件名可能是临时名，先下目录再找 .pt）
        model_dir = mlflow.artifacts.download_artifacts(
            run_id=run.run_id, artifact_path="model"
        )
        import glob
        pt_files = glob.glob(os.path.join(model_dir, "*.pt"))
        if not pt_files:
            print(f"  ⚠️ {name}: model/ 下没有 .pt 文件，跳过")
            continue
        state_path = pt_files[0]
        state = torch.load(state_path, map_location="cpu")

        model = create_model(name, num_classes=10, pretrained=False)
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()

        y_true, y_pred, y_proba, latencies = predict_full(model, test_loader, device)
        preds_per_model[name] = {
            "y_pred": y_pred,
            "y_proba": y_proba,
            "metrics_run_id": run.run_id,
        }
        print(f"  ✓ {name}: 加载完成 (run {run.run_id[:8]})")

    if y_true is None:
        print("  ⚠️ 没有任何模型加载成功，检查模型 Run 是否存在")
        return {}, test_ds, class_names
    preds_per_model["y_true"] = y_true
    return preds_per_model, test_ds, class_names


def find_run_for_model(exp, model_name):
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.model_name = '{model_name}'",
        max_results=1,
    )
    if len(runs) == 0:
        return None
    return runs.iloc[0]


def find_hard_examples(preds_per_model, y_true, min_wrong=5):
    """找所有模型都错（或大部分错）的样本。"""
    n_models = len([k for k in preds_per_model if k != "y_true"])
    model_keys = [k for k in preds_per_model if k != "y_true"]

    n_wrong_per_sample = np.zeros(len(y_true), dtype=int)
    for k in model_keys:
        n_wrong_per_sample += (preds_per_model[k]["y_pred"] != y_true).astype(int)

    return n_wrong_per_sample, n_models


def make_hard_examples_grid(images_uint8, n_wrong, n_models, y_true,
                            preds_per_model, class_names, n=16):
    """画 n 张"n_models 全错"的最难样本 + 每个模型的预测。"""
    # images_uint8 可能只有前 N 张图（sample），但 n_wrong 是全集的
    n_imgs = len(images_uint8)
    # 选全错的（n_wrong == n_models），如果不够就选错误最多的
    all_wrong_idx = np.where(n_wrong == n_models)[0]
    # 只保留在前 n_imgs 张内的索引
    all_wrong_idx = all_wrong_idx[all_wrong_idx < n_imgs]
    if len(all_wrong_idx) < n:
        # 降级：选 n_wrong 最大的 n 张（也要在 sample 范围内）
        candidates = np.argsort(-n_wrong)
        candidates = candidates[candidates < n_imgs]
        all_wrong_idx = candidates[:n]
    all_wrong_idx = all_wrong_idx[:n]

    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 2.5, n_rows * 2.5))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i < len(all_wrong_idx):
            idx = all_wrong_idx[i]
            ax.imshow(images_uint8[idx])
            lines = [f"TRUE: {class_names[y_true[idx]]}"]
            for k in [k for k in preds_per_model if k != "y_true"]:
                pred_n = class_names[preds_per_model[k]["y_pred"][idx]]
                conf = preds_per_model[k]["y_proba"][idx, preds_per_model[k]["y_pred"][idx]]
                ok = "✓" if pred_n == lines[0].split(": ")[1] else "✗"
                lines.append(f"{k}: {pred_n} ({conf:.2f}) {ok}")
            ax.set_title("\n".join(lines), fontsize=6.5)
        ax.axis("off")
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return arr, all_wrong_idx


def make_disagreement_matrix(preds_per_model, y_true, n_models):
    """画每个样本被多少模型错的矩阵（按 disagreement 排序）。"""
    model_keys = [k for k in preds_per_model if k != "y_true"]
    n_wrong = np.zeros(len(y_true), dtype=int)
    for k in model_keys:
        n_wrong += (preds_per_model[k]["y_pred"] != y_true).astype(int)

    # 按 disagreement 降序
    order = np.argsort(-n_wrong)
    n_show = min(200, len(y_true))
    mat = np.zeros((n_show, n_models), dtype=np.float32)
    for i, idx in enumerate(order[:n_show]):
        for j, k in enumerate(model_keys):
            mat[i, j] = preds_per_model[k]["y_pred"][idx] != y_true[idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(mat, cmap="Reds", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("model")
    ax.set_ylabel("sample (sorted by disagreement)")
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_keys, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Disagreement matrix (top {n_show} samples)")
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return arr, n_wrong


def make_per_class_difficulty(preds_per_model, y_true, class_names):
    """画每个类被每个模型识别的 F1（heatmap）。"""
    from collections import defaultdict
    model_keys = [k for k in preds_per_model if k != "y_true"]
    n_models = len(model_keys)
    n_classes = len(class_names)

    # 每模型每类的 F1
    f1_mat = np.zeros((n_classes, n_models))
    for j, k in enumerate(model_keys):
        y_pred = preds_per_model[k]["y_pred"]
        for c in range(n_classes):
            tp = ((y_pred == c) & (y_true == c)).sum()
            fp = ((y_pred == c) & (y_true != c)).sum()
            fn = ((y_pred != c) & (y_true == c)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_mat[c, j] = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(f1_mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_models))
    ax.set_xticklabels(model_keys, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_title("Per-class F1 score heatmap")
    for i in range(n_classes):
        for j in range(n_models):
            ax.text(j, i, f"{f1_mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black" if f1_mat[i, j] > 0.5 else "white")
    fig.tight_layout()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return arr


def make_family_comparison(preds_per_model, y_true):
    """同族不同深度对比曲线（每个族一张图）。"""
    model_keys = [k for k in preds_per_model if k != "y_true"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    families = ["resnet", "efficientnet", "densenet"]
    for ax, fam in zip(axes, families):
        fam_models = [k for k in model_keys if k.startswith(fam)]
        accs = [(preds_per_model[k]["y_pred"] == y_true).mean() for k in fam_models]
        ax.plot(range(len(fam_models)), accs, "o-", linewidth=2, markersize=10)
        ax.set_xticks(range(len(fam_models)))
        ax.set_xticklabels(fam_models, rotation=0)
        ax.set_ylim(0.85, 1.0)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{fam} family")
        ax.grid(alpha=0.3)
    fig.suptitle("Accuracy vs Model Depth (per family)")
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return arr


def compute_ensemble(preds_per_model, y_true):
    """Hard voting ensemble。"""
    model_keys = [k for k in preds_per_model if k != "y_true"]
    # 收集所有预测 → shape (n_samples, n_models)
    all_preds = np.stack([preds_per_model[k]["y_pred"] for k in model_keys], axis=1)
    # 投票
    from scipy.stats import mode
    ensemble_pred, _ = mode(all_preds, axis=1, keepdims=False)
    acc = (ensemble_pred == y_true).mean()
    return float(acc), ensemble_pred


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    exp = mlflow.get_experiment_by_name("10_vision_classification")

    # 1. 加载 9 个模型的预测
    print("=" * 60)
    print("加载 9 个模型...")
    print("=" * 60)
    preds_per_model, test_ds, class_names = load_all_predictions_from_artifacts(exp, MODEL_NAMES)
    y_true = preds_per_model["y_true"]
    n_models = len([k for k in preds_per_model if k != "y_true"])
    print(f"\n  共 {n_models} 个模型，{len(y_true)} 个测试样本")

    if n_models < 2:
        print("✗ 至少需要 2 个模型的预测，跳过")
        return

    # 2. 计算 ensemble
    ensemble_acc, ensemble_pred = compute_ensemble(preds_per_model, y_true)
    print(f"\n  Ensemble (hard voting) acc: {ensemble_acc:.4f}")

    # 3. 加载测试集图片（取前 256 张够做 grid）
    print("\n加载测试集图片...")
    sample_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                                num_workers=0)
    sample_x, sample_y = next(iter(sample_loader))
    sample_uint8 = tensor_to_uint8_images(sample_x)

    # 4. 找 hard examples + 画图
    n_wrong, n_models_total = find_hard_examples(preds_per_model, y_true)
    print(f"\n  所有 {n_models_total} 模型都错的样本数: {(n_wrong == n_models_total).sum()}")
    print(f"  至少错一半的样本数: {(n_wrong > n_models_total / 2).sum()}")

    hard_grid, hard_idx = make_hard_examples_grid(
        sample_uint8, n_wrong, n_models_total, y_true,
        preds_per_model, class_names, n=16
    )

    disagreement_grid, _ = make_disagreement_matrix(
        preds_per_model, y_true, n_models_total
    )

    per_class_grid = make_per_class_difficulty(
        preds_per_model, y_true, class_names
    )

    family_grid = make_family_comparison(preds_per_model, y_true)

    # 5. Failures metadata CSV
    print("\n生成 failures.csv...")
    rows = []
    model_keys = [k for k in preds_per_model if k != "y_true"]
    for idx in np.where(y_true != preds_per_model[model_keys[0]]["y_pred"])[0][:200]:
        row = {
            "image_idx": int(idx),
            "true_label": class_names[int(y_true[idx])],
            "n_models_correct": int(n_models_total - n_wrong[idx]),
            "n_models_wrong": int(n_wrong[idx]),
        }
        for k in model_keys:
            row[f"{k}_pred"] = class_names[int(preds_per_model[k]["y_pred"][idx])]
            row[f"{k}_conf"] = float(
                preds_per_model[k]["y_proba"][idx, preds_per_model[k]["y_pred"][idx]])
        rows.append(row)
    df_failures = pd.DataFrame(rows)
    csv_path = "/tmp/failures.csv"
    df_failures.to_csv(csv_path, index=False)

    # 6. MLflow Run
    print("\nLog 到 MLflow...")
    with mlflow.start_run(run_name="10d_failure_analysis") as run:
        mlflow.set_tag("phase", "10")
        mlflow.set_tag("dataset", "cifar10")
        mlflow.set_tag("step", "failure_analysis")

        mlflow.log_params({
            "n_models": n_models_total,
            "n_samples": len(y_true),
            "models": ",".join(model_keys),
        })

        mlflow.log_metrics({
            "ensemble_acc": ensemble_acc,
            "best_single_model_acc": float(max(
                (preds_per_model[k]["y_pred"] == y_true).mean() for k in model_keys)),
            "n_samples_all_wrong": int((n_wrong == n_models_total).sum()),
            "n_samples_majority_wrong": int((n_wrong > n_models_total / 2).sum()),
            "disagreement_rate": float((n_wrong > 0).mean()),
        })

        mlflow.log_image(hard_grid, artifact_file="hard_examples_grid.png")
        mlflow.log_image(disagreement_grid, artifact_file="disagreement_matrix.png")
        mlflow.log_image(per_class_grid, artifact_file="per_class_f1_heatmap.png")
        mlflow.log_image(family_grid, artifact_file="family_comparison.png")
        mlflow.log_artifact(csv_path, artifact_path="")

    print(f"\n✓ Run: {run.info.run_id[:8]}")
    print("\n" + "=" * 60)
    print("📊 关键发现:")
    print("=" * 60)
    print(f"  Ensemble acc: {ensemble_acc:.4f}")
    best_single = max((preds_per_model[k]["y_pred"] == y_true).mean() for k in model_keys)
    print(f"  最佳单模型 acc: {best_single:.4f}")
    print(f"  Ensemble vs 最佳单模型: {ensemble_acc - best_single:+.4f}")
    print(f"  全员都错的样本: {(n_wrong == n_models_total).sum()}")
    print(f"  至少一半错的样本: {(n_wrong > n_models_total / 2).sum()}")


if __name__ == "__main__":
    main()