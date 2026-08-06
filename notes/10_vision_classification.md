# 阶段 10 学习笔记：图像分类深度学习对比实验

> 对应脚本：`scripts/10_vision_classification/10a_dataset_overview.py`、`10b_train_models.py`、`10c_compare_and_evaluate.py`、`10d_failure_analysis.py`
> 需要 API Key：否（纯本地训练）
> 需要环境变量：可选 `HF_ENDPOINT=https://hf-mirror.com`（国内加速 huggingface 权重下载）

## 🎯 这篇笔记做什么

前 9 个 phase 都用 sklearn 在小表格数据上跑。现实世界的图像识别问题，必须上深度学习 CNN。这一阶段就是用 **9 个 timm 预训练 CNN** 在 CIFAR-10 上做端到端对比实验，演示 MLflow 完整能力：
- 完整指标记录（accuracy / F1 / precision / recall 三种平均 + 训练/推理时间 + 模型大小）
- 失败案例可视化（错分样本 grid + 混淆矩阵）
- 跨模型对比（不同家族 + 不同深度 + ensemble 投票）

打个比方：这一阶段就像办一场"汽车拉力赛"——9 款不同车型（ResNet / EfficientNet / DenseNet）跑同一段路（CIFAR-10），记录每辆车的速度、油耗、故障次数，最后告诉你"谁最适合你"。

**跑完的产出**：MLflow UI 里能看到 9 个 Run 的并排对比，每个 Run 有 failures_grid.png、successes_grid.png、confusion_matrix.png 和一个跨模型分析 Run（hard examples + ensemble）。

### 你会学到什么

- 用 timm 一行代码加载 9 种 ImageNet 预训练 CNN（ResNet/EfficientNet/DenseNet 各 3 个深度变体）
- 用 torchvision transforms + DataLoader 加载 CIFAR-10（首次自动下载 ~170MB）
- 微调预训练模型到 10 类（CIFAR-10）
- 计算 11 个分类指标（accuracy + F1/precision/recall × 3 平均 + balanced_accuracy + per-class 分布）
- 用 `mlflow.log_image()` 记录失败/成功样本 grid 和混淆矩阵
- 用 `mlflow.log_input()` 给 Run 关联数据集血缘
- 跨模型分析：disagreement matrix + per-class F1 heatmap + family comparison
- Hard voting ensemble 的实现和效果

### 对应脚本清单

| 脚本 | 一句话作用 | 是否必跑 | 前置 |
|------|-----------|---------|------|
| `10a_dataset_overview.py` | 下载 CIFAR-10 + 样本 grid + log_input | ✓ 必跑 | 无 |
| `10b_train_models.py` | 训练 9 个 timm 预训练 CNN（~30-50 分钟） | ✓ 必跑 | 跑过 10a |
| `10c_compare_and_evaluate.py` | 跨模型指标汇总 + 性价比分析 | 推荐 | 跑过 10b |
| `10d_failure_analysis.py` | 跨模型失败案例 + ensemble 实验 | 推荐 | 跑过 10b |

### 前置知识

- 已完成 Phase 1-9（会用 MLflow 各种 API）
- 装好 `torch` + `torchvision` + `timm`（脚本前会装）
- 本地有 `mlflow.db`（前 9 个 phase 已创建）
- 第一次跑会从 huggingface 下载 9 个预训练权重（约 100-300MB），国内可设 `HF_ENDPOINT=https://hf-mirror.com` 加速

### 跑完必看（UI）

1. 启动 UI：`mlflow ui --port 5000`
2. 选 experiment `10_vision_classification`
3. 看到 11 个 Run（1 数据集 + 9 训练 + 1 对比汇总，可选 1 失败分析）
4. 点开任意训练 Run（如 `cifar10_resnet50`），看：
   - **Metrics 标签**：`accuracy_score` / `f1_macro` / `precision_macro` / `recall_macro` / `balanced_accuracy_score` / `train_time_seconds` / `inference_time_per_sample_ms` / `model_size_bytes` 等 14 个
   - **Artifacts → model/model.pt**：PyTorch state_dict（state_dict 不是完整模型，要用 `create_model()` 重建才能 load）
   - **Artifacts → failures_grid.png**：8×8 = 64 张错分样本，每张标注"真值 → 预测 (置信度)"
   - **Artifacts → successes_grid.png**：8×8 = 64 张正确样本（对比看）
   - **Artifacts → confusion_matrix.png**：10×10 混淆矩阵
5. 用 search_runs 找 9 个模型 Run，点 Compare 并排对比所有指标
6. `10c` 和 `10d` 跑完后看：
   - `10c_comparison_summary` Run → 看汇总表（family_summary.json + all_models_comparison.json）
   - `10d_failure_analysis` Run → 看 hard_examples_grid.png + per_class_f1_heatmap.png + family_comparison.png

---

## 一、核心概念：用人话讲清楚

### 1. CIFAR-10：经典图像分类 benchmark

10 类 32×32 彩色图（飞机/汽车/鸟/猫/鹿/狗/青蛙/马/船/卡车），每个类 6000 张。深度学习的"Hello World"数据集。

```
50000 训练 + 10000 测试，每张 32×32×3（RGB）
```

我们用 **timm 预训练的 ImageNet 模型微调**，因为从零训练小数据集容易过拟合，预训练模型已经学到了"边缘、纹理、形状"等通用特征。

### 2. timm：预训练模型的"应用商店"

`timm`（PyTorch Image Models）是一个模型库，统一接口访问 300+ SOTA 视觉模型。我们用它加载 ResNet/EfficientNet/DenseNet 变体，避免重复造轮子：

```python
import timm
model = timm.create_model("resnet18", pretrained=True, num_classes=10)
# ↓ 一次性得到：加载 ImageNet 权重 + 替换分类头为 10 类
```

### 3. 9 个模型怎么分？

按"家族 × 深度"二维分组：

| 家族 | 深度变体 | 共同特点 |
|------|---------|---------|
| **ResNet** | 18 / 34 / 50 | 残差连接，深度递增 |
| **EfficientNet** | B0 / B1 / B2 | 复合缩放（深度+宽度+分辨率同时调）|
| **DenseNet** | 121 / 169 / 201 | 密集连接，每层接收前面所有层的特征 |

9 个模型覆盖"经典 / 轻量 / 密集连接"三大主流 CNN 架构，每个家族 3 个深度变体 → 既能比"哪个家族好"，也能比"深度增加有没有用"。

### 4. 微调策略

只训练 3 个 epoch，用 ImageNet 预训练权重 + 替换分类头。学习率分两组：
- backbone（小 lr=1e-4）：已学到的特征别破坏
- head（大 lr=1e-3）：新分类头快速适配 10 类

### 5. 评估指标全景

11 个核心指标，三种平均方式（micro / macro / weighted）都给：

| 指标 | 含义 | 何时看 |
|------|------|--------|
| `accuracy_score` | 整体准确率 | 第一眼看 |
| `f1_macro` | 各类 F1 平均（不权重） | 类别均衡时 |
| `f1_weighted` | 各类 F1 按样本数加权 | 类别不均衡时 |
| `f1_micro` | 等价于 accuracy | 与 sklearn 一致 |
| `precision_macro/weighted` | 各类预测为正类的准确度 | 关注误报 |
| `recall_macro/weighted` | 各类识别正类的能力 | 关注漏报 |
| `balanced_accuracy_score` | 每类 recall 平均 | 类别不均衡时 |
| `f1_per_class_min/std` | 最差类的 F1 / 标准差 | 看是否偏科 |
| `train_time_seconds` | 训练耗时 | 部署决策 |
| `inference_time_per_sample_ms` | 单样本推理时间 | 实时性要求 |
| `model_size_bytes` | state_dict 大小 | 边缘部署 |

### 6. 失败案例可视化

每个模型训练完都生成 3 张图：
- **`failures_grid.png`**：8×8 = 64 张错分样本，每张图上标"真值 → 预测 (置信度)"，让你一眼看出"模型把猫认成狗，但很自信"
- **`successes_grid.png`**：8×8 = 64 张正确样本，对比看"模型做对的题长什么样"
- **`confusion_matrix.png`**：10×10 混淆矩阵（归一化），看"猫 vs 狗"这种容易混的类

### 7. 跨模型对比（10d）

加载 9 个模型的 state_dict，对同一测试集重新预测，分析：
- **所有模型都错的"hard examples"**：网格化展示，每张图叠加 9 个模型的预测+置信度
- **disagreement matrix**：横轴=模型，纵轴=样本，红=预测错，白=对，一眼看出"哪些样本 9 个模型意见分歧"
- **per-class F1 heatmap**：横轴=模型，纵轴=类别，看哪个类被哪个家族识别得最好/最差
- **family comparison**：同族不同深度曲线（ResNet 18→50 的 acc 变化）
- **ensemble (hard voting)**：9 个模型投票，看是否能超过最佳单模型

---

## 二、代码模式：可复用的模板

### 模板 1：加载 CIFAR-10

```python
import torchvision
from torchvision import transforms

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),    # 适配 ImageNet 预训练
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])
train = torchvision.datasets.CIFAR10(
    root="~/data/cifar10", train=True, download=True,
    transform=transform_train,
)
```

### 模板 2：用 timm 训练单个模型

```python
import timm, torch
model = timm.create_model("resnet18", pretrained=True, num_classes=10)

# backbone 用小学习率，head 用大学习率
backbone_params, head_params = [], []
for n, p in model.named_parameters():
    (head_params if any(k in n for k in ["fc", "classifier"]) else backbone_params).append(p)
optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 1e-4},
    {"params": head_params, "lr": 1e-3},
])

# 标准 PyTorch 训练循环（参考 shared.py 的 train_one_epoch）
```

### 模板 3：计算全套分类指标（不依赖 sklearn）

```python
# shared.py 的 compute_metrics() 用 numpy 实现
# 涵盖 accuracy / precision / recall / F1 三种平均 + balanced_accuracy
metrics = shared.compute_metrics(y_true, y_pred)
```

### 模板 4：MLflow 记录完整 Run

```python
import mlflow
with mlflow.start_run(run_name=f"cifar10_{model_name}"):
    mlflow.set_tags({"phase": "10", "model_family": "resnet"})
    mlflow.log_params({...})
    mlflow.log_metrics(metrics)
    mlflow.log_image(failures_grid, artifact_file="failures_grid.png")
    mlflow.log_artifact("model.pt", artifact_path="model")
```

---

## 三、实战步骤：按顺序照做

### Step 0：装依赖（如还没装）

```bash
conda run -n mlflow pip install \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu
conda run -n mlflow pip install timm
```

### Step 1：跑数据概览

```bash
python scripts/10_vision_classification/10a_dataset_overview.py
```

会下载 CIFAR-10（~170MB），生成 `dataset_overview_cifar10` Run。

### Step 2：训练 9 个模型（最耗时，30-50 分钟）

```bash
python scripts/10_vision_classification/10b_train_models.py
```

**想先快速测试**：打开 10b 顶部 `MODEL_NAMES` 列表，临时注释掉大部分，只留 `resnet18`，`epochs` 改成 `1`。

### Step 3：跨模型对比

```bash
python scripts/10_vision_classification/10c_compare_and_evaluate.py
```

### Step 4：失败案例分析

```bash
python scripts/10_vision_classification/10d_failure_analysis.py
```

### Step 5：UI 看效果

```bash
mlflow ui --port 5000
```

→ experiment `10_vision_classification` → 11 个 Run 并排对比

---

## 四、避坑清单

- ⚠️ **CIFAR-10 首次下载慢**：~170MB；下载完就缓存到 `~/data/cifar10/`，以后离线可用。如果网络不稳可手动下载后放到该目录。
- ⚠️ **timm 预训练权重下载**：第一次跑每个模型都会从 huggingface 下载（resnet18 ~45MB、efficientnet_b0 ~20MB、densenet201 ~80MB），9 个合计约 350MB。国内可设：
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```
- ⚠️ **CPU 微调慢**：每模型 3-5 分钟，9 个合计 30-50 分钟。如果嫌慢，先跑 1-2 个模型看效果。
- ⚠️ **state_dict 不是完整模型**：`mlflow.log_artifact("model.pt")` 只存权重，加载时需要 `timm.create_model(name, pretrained=False)` + `load_state_dict()`
- ⚠️ **`mlflow.models.evaluate` 跳过**：因为 timm 模型需要 Pyfunc 包装成复杂代码，10c 简化为直接用 MLflow 记录指标汇总，不调 evaluate API
- ⚠️ **CIFAR-10 类不均衡**：每类正好 6000 张训练样本，不算严重不均衡，但 `balanced_accuracy_score` 仍然值得看
- ⚠️ **预测时显示的 cm 中有 NaN**：如果某个类一个都没预测对，对应行会全 0，PNG 仍能生成（值都是 NaN 但 matplotlib 会显示 0）

---

## 五、小结：5 个 take-aways

- **timm 让加载 9 种 SOTA CNN 像 `pip install` 一样简单**——一行 `create_model(name, pretrained=True, num_classes=10)` 拿到 ImageNet 预训练权重
- **同族不同深度 + 跨家族对比**是图像实验的标准模式：ResNet 18/34/50 看深度收益，ResNet vs EfficientNet vs DenseNet 看架构差异
- **11 个指标全覆盖**：3 种平均方式（micro/macro/weighted）+ balanced accuracy + per-class 分布 + 训练/推理时间 + 模型大小
- **失败案例可视化是 MLflow 的杀手锏**：错分样本 grid + 混淆矩阵 + 跨模型 hard examples 让 debug 不再"猜"
- **Ensemble 投票常能再 +1-2%**：9 模型 hard voting 通常比最佳单模型高，比单模型更稳