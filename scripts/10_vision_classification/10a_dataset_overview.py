"""
阶段 10 示例 0: CIFAR-10 数据集概览
=======================================

目标：
  - 首次下载 CIFAR-10 到 ~/data/cifar10/（约 170MB）
  - 在 MLflow 里记下数据集元信息 + 样本 grid
  - 用 mlflow.log_input 关联数据集到 Run（数据血缘）

运行：
  conda activate mlflow
  python scripts/10_vision_classification/10a_dataset_overview.py

⚠️ 首次运行会下载 CIFAR-10，需联网。之后离线可用。
   预训练权重（10b 用）首次也会从 huggingface 下载，
   国内可设 HF_ENDPOINT=https://hf-mirror.com 加速。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlflow
import pandas as pd
import numpy as np
import torch

from shared import (
    load_cifar10,
    tensor_to_uint8_images,
    make_grid,
    CLASS_NAMES,
)


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("10_vision_classification")

    with mlflow.start_run(run_name="dataset_overview_cifar10") as run:
        mlflow.set_tag("phase", "10")
        mlflow.set_tag("dataset", "cifar10")
        mlflow.set_tag("step", "data_overview")

        print("加载 CIFAR-10（首次自动下载）...")
        train_ds, test_ds, class_names = load_cifar10()
        n_train = len(train_ds)
        n_test = len(test_ds)
        n_classes = len(class_names)
        print(f"  训练集: {n_train} 张")
        print(f"  测试集: {n_test} 张")
        print(f"  类别: {n_classes} 个")

        mlflow.log_params({
            "n_train": n_train,
            "n_test": n_test,
            "n_classes": n_classes,
            "image_shape": "(3, 224, 224)",   # resize 后
            "raw_image_shape": "(3, 32, 32)",  # 原始
            "class_names": ",".join(class_names),
        })

        # 类别分布
        train_labels = np.array(train_ds.targets)
        class_counts = {class_names[i]: int((train_labels == i).sum())
                        for i in range(n_classes)}
        mlflow.log_dict(class_counts, "class_distribution.json")

        # ---- 样本 grid：随机 32 张 ----
        print("生成随机样本 grid...")
        from torch.utils.data import DataLoader
        sample_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                   num_workers=0)
        sample_images_uint8 = None
        sample_labels = None
        for x, y in sample_loader:
            sample_images_uint8 = tensor_to_uint8_images(x)
            sample_labels = y.numpy()
            break   # 只取第一批

        titles = [class_names[int(y)] for y in sample_labels]
        grid_img = make_grid(sample_images_uint8, n_cols=8, titles=titles,
                              figsize_per_cell=1.6)
        mlflow.log_image(grid_img, artifact_file="dataset_samples.png")
        print("  ✓ log dataset_samples.png (32 张随机样本)")

        # ---- 类别 grid：每类 5 张 ----
        print("生成类别代表 grid...")
        per_class_imgs = []
        per_class_titles = []
        # 按类别找前 5 张
        for cls_idx in range(n_classes):
            cls_mask = np.where(train_labels == cls_idx)[0][:5]
            if len(cls_mask) == 0:
                continue
            # 直接用 __getitem__ 拿图
            imgs = [train_ds[int(i)][0] for i in cls_mask]
            per_class_imgs.append(tensor_to_uint8_images(torch.stack(imgs)))
            per_class_titles.extend([class_names[cls_idx]] * 5)
        per_class_imgs_arr = np.concatenate(per_class_imgs, axis=0)
        grid_per_class = make_grid(per_class_imgs_arr, n_cols=5,
                                    titles=per_class_titles,
                                    figsize_per_cell=2.0)
        mlflow.log_image(grid_per_class, artifact_file="class_grid.png")
        print(f"  ✓ log class_grid.png (10 类 × 5 张 = 50 张)")

        # ---- log_input 数据血缘 ----
        # 把 train set 的 (image_index, label) 转成 DataFrame，log_input
        print("记录数据集血缘...")
        input_df = pd.DataFrame({
            "image_idx": list(range(n_train)),
            "label": [int(l) for l in train_labels],
            "label_name": [class_names[int(l)] for l in train_labels],
        })
        train_dataset = mlflow.data.from_pandas(
            input_df, source="torchvision.datasets.CIFAR10",
            name="cifar10_train",
            targets="label",
        )
        mlflow.log_input(train_dataset, context="training")

        test_labels = np.array(test_ds.targets)
        test_df = pd.DataFrame({
            "image_idx": list(range(n_test)),
            "label": [int(l) for l in test_labels],
            "label_name": [class_names[int(l)] for l in test_labels],
        })
        test_dataset = mlflow.data.from_pandas(
            test_df, source="torchvision.datasets.CIFAR10",
            name="cifar10_test",
            targets="label",
        )
        mlflow.log_input(test_dataset, context="testing")
        print("  ✓ log_input train + test 数据集")

    print(f"\n✓ Run: {run.info.run_id[:8]}")
    print("\n下一步: 10b_train_models.py（~30-50 分钟，9 个模型）")


if __name__ == "__main__":
    main()