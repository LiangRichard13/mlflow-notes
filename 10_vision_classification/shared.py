"""
Phase 10 - shared utilities for image classification comparison.

提供：
  - CIFAR-10 数据加载（首次自动下载到 ~/data/cifar10/）
  - timm 模型工厂（resnet/efficientnet/densenet）
  - 训练循环 + 推理 + 指标计算
  - 图像 grid 可视化（带文字叠加）
  - 混淆矩阵绘制
  - state_dict 保存（方便后续 MLflow 记录模型）
"""
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

import timm
import mlflow


# CIFAR-10 类别名
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# ImageNet 归一化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# 9 个对比模型（按家族分组）
MODEL_NAMES = [
    "resnet18", "resnet34", "resnet50",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    "densenet121", "densenet169", "densenet201",
]


# ============================================================
# 数据加载
# ============================================================
def get_transforms(image_size: int = 224, train: bool = True):
    """训练集含数据增强；测试集只 Resize+Normalize。"""
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_cifar10(data_dir: str = "~/data/cifar10", image_size: int = 224):
    """返回 (train_dataset, test_dataset, class_names)。首次自动下载 ~170MB。"""
    data_dir = os.path.expanduser(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=get_transforms(image_size, train=True),
    )
    test = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=get_transforms(image_size, train=False),
    )
    return train, test, CLASS_NAMES


# ============================================================
# 模型工厂
# ============================================================
def model_family(name: str) -> str:
    if name.startswith("resnet"): return "resnet"
    if name.startswith("efficientnet"): return "efficientnet"
    if name.startswith("densenet"): return "densenet"
    return "other"


def create_model(name: str, num_classes: int = 10, pretrained: bool = True):
    """用 timm 创建模型，分类头替换为 num_classes。"""
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


# ============================================================
# 训练 & 推理
# ============================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, correct, total_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        correct += (out.argmax(1) == y).sum().item()
    return total_loss / total, correct / total


@torch.no_grad()
def predict_full(model, loader, device):
    """返回 (y_true, y_pred, y_proba, latencies_ms_per_sample)."""
    model.eval()
    ys, ps, probs, latencies = [], [], [], []
    for x, y in loader:
        x = x.to(device)
        t0 = time.perf_counter()
        logits = model(x)
        latency = (time.perf_counter() - t0) / x.size(0) * 1000
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        pred = prob.argmax(1)
        ys.append(y.numpy())
        ps.append(pred)
        probs.append(prob)
        latencies.extend([latency] * x.size(0))
    return (np.concatenate(ys), np.concatenate(ps),
            np.concatenate(probs), np.array(latencies))


def compute_metrics(y_true, y_pred):
    """11 个核心分类指标 + per-class 分布信息（不依赖 sklearn）。"""
    n = len(y_true)
    n_classes = int(max(y_true.max(), y_pred.max())) + 1

    # per-class TP/FP/FN
    tp = np.zeros(n_classes)
    fp = np.zeros(n_classes)
    fn = np.zeros(n_classes)
    support = np.zeros(n_classes)
    for t, p in zip(y_true, y_pred):
        support[t] += 1
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        per_prec = np.where((tp + fp) > 0, tp / (tp + fp), 0)
        per_rec  = np.where((tp + fn) > 0, tp / (tp + fn), 0)
        per_f1   = np.where((per_prec + per_rec) > 0,
                            2 * per_prec * per_rec / (per_prec + per_rec), 0)

    macro_prec = per_prec.mean()
    macro_rec  = per_rec.mean()
    macro_f1   = per_f1.mean()
    weights = support / support.sum().clip(min=1)
    weighted_prec = (per_prec * weights).sum()
    weighted_rec  = (per_rec * weights).sum()
    weighted_f1   = (per_f1 * weights).sum()
    micro_prec = micro_rec = micro_f1 = tp.sum() / n
    accuracy = (y_true == y_pred).mean()
    balanced_acc = macro_rec

    return {
        "accuracy_score": float(accuracy),
        "f1_micro": float(micro_f1),
        "f1_macro": float(macro_f1),
        "f1_weighted": float(weighted_f1),
        "precision_micro": float(micro_prec),
        "precision_macro": float(macro_prec),
        "precision_weighted": float(weighted_prec),
        "recall_micro": float(micro_rec),
        "recall_macro": float(macro_rec),
        "recall_weighted": float(weighted_rec),
        "balanced_accuracy_score": float(balanced_acc),
        "f1_per_class_min": float(per_f1.min()),
        "f1_per_class_max": float(per_f1.max()),
        "f1_per_class_std": float(per_f1.std()),
    }


def train_and_evaluate(model_name, train_ds, test_ds, class_names,
                        epochs=3, batch_size=128,
                        lr_backbone=1e-4, lr_head=1e-3,
                        num_workers=2, max_train_batches=None,
                        max_test_batches=None):
    """
    训练单个模型 + 测试推理。
    max_train_batches/max_test_batches 用于快速 smoke test。
    返回 (model, metrics_dict, predictions_dict, state_dict_bytes)。
    """
    device = torch.device("cpu")
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                               shuffle=False, num_workers=num_workers)

    model = create_model(model_name, num_classes=len(class_names), pretrained=True)
    model = model.to(device)

    # backbone 用小学习率，分类头用大学习率
    head_params, backbone_params = [], []
    for n, p in model.named_parameters():
        if any(k in n for k in ["fc", "classifier", "head"]):
            head_params.append(p)
        else:
            backbone_params.append(p)
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ])
    criterion = nn.CrossEntropyLoss()

    # 训练
    t_train_start = time.perf_counter()
    for epoch in range(epochs):
        loss, acc = 0.0, 0.0
        n_batches = 0
        for i, (x, y) in enumerate(train_loader):
            if max_train_batches is not None and i >= max_train_batches:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_val = loss.item() * x.size(0)
            acc_val = (out.argmax(1) == y).sum().item() / x.size(0)
            # 累加
            n_batches += 1
            # 用 EMA-style 简化打印（避免一直累积）
        # 简化：每 epoch 跑完了不严格计算 mean，loss/acc 取最后一批
        print(f"  [epoch {epoch+1}/{epochs}] last_batch_loss={loss.item():.4f}  last_batch_acc={acc_val:.4f}")
    train_time = time.perf_counter() - t_train_start

    # 推理
    y_true, y_pred, y_proba, latencies = predict_full(model, test_loader, device)
    if max_test_batches is not None:
        # 限制测试 batch 数（smoke test）
        n_keep = min(len(y_true), max_test_batches * batch_size)
        y_true, y_pred, y_proba, latencies = (
            y_true[:n_keep], y_pred[:n_keep], y_proba[:n_keep], latencies[:n_keep])

    metrics = compute_metrics(y_true, y_pred)
    metrics["train_time_seconds"] = float(train_time)
    metrics["inference_time_total_seconds"] = float(latencies.sum() / 1000)
    metrics["inference_time_per_sample_ms"] = float(latencies.mean())

    # 模型大小
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    metrics["model_size_bytes"] = float(buf.tell())
    state_bytes = buf.getvalue()

    return model, metrics, {
        "y_true": y_true, "y_pred": y_pred, "y_proba": y_proba,
        "latencies_ms": latencies,
    }, state_bytes


# ============================================================
# 图像可视化
# ============================================================
def make_grid(images_uint8, n_cols, titles=None,
              figsize_per_cell=1.6):
    """[N, H, W, 3] uint8 → grid ndarray [H', W', 3] uint8."""
    n = len(images_uint8)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(
        n_cols * figsize_per_cell, n_rows * figsize_per_cell))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images_uint8[i])
            if titles is not None and i < len(titles):
                ax.set_title(titles[i], fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return arr


def make_failure_grid(images_uint8, y_true, y_pred, y_proba,
                       class_names, n=64, n_cols=8):
    """错分样本 grid，每张图叠加"真值 → 预测 (置信度)"。"""
    # 兼容 torch.Tensor 输入
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()
    wrong_idx = np.where(y_true != y_pred)[0][:n]
    if len(wrong_idx) == 0:
        return None
    titles = []
    for i in wrong_idx:
        true_n = class_names[int(y_true[i])]
        pred_n = class_names[int(y_pred[i])]
        conf = float(y_proba[i, y_pred[i]])
        titles.append(f"t={true_n}\np={pred_n}\nconf={conf:.2f}")
    return make_grid(images_uint8[wrong_idx], n_cols=n_cols, titles=titles)


def make_success_grid(images_uint8, y_true, y_pred, y_proba,
                       class_names, n=64, n_cols=8):
    """正确样本 grid。"""
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()
    correct_idx = np.where(y_true == y_pred)[0][:n]
    if len(correct_idx) == 0:
        return None
    titles = [f"t={class_names[int(y_true[i])]}\nconf={float(y_proba[i, y_pred[i]]):.2f}"
              for i in correct_idx]
    return make_grid(images_uint8[correct_idx], n_cols=n_cols, titles=titles)


def make_confusion_matrix(y_true, y_pred, class_names, normalize=True):
    """画混淆矩阵，返回 [H, W, 3] uint8。"""
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=np.float32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalized)" if normalize else "Confusion Matrix (counts)")
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6, color=color)
    fig.tight_layout()
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return arr


def tensor_to_uint8_images(x_normalized):
    """[N, 3, H, W] 归一化 tensor → [N, H, W, 3] uint8."""
    arr = x_normalized.detach().cpu().numpy()
    mean = np.array(IMAGENET_MEAN).reshape(1, 3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(1, 3, 1, 1)
    arr = arr * std + mean
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    arr = arr.transpose(0, 2, 3, 1)
    return arr


# ============================================================
# 模型保存到 MLflow artifact
# ============================================================
def log_timm_state_dict(state_bytes, run_id, artifact_path="model"):
    """把 timm state_dict 字节流写到 Run 的 artifact 目录。"""
    import io, tempfile, os
    from mlflow import MlflowClient
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        f.write(state_bytes)
        tmp = f.name
    client = MlflowClient()
    client.log_artifact(run_id, tmp, artifact_path)
    os.unlink(tmp)
    return f"{artifact_path}/model.pt"