# MLflow 3 作业指导书（QUICKSTART）

目标：1-2 天内从环境安装到跑通第一个实验，再到用 coding agent 在 vibecoding 场景下用 MLflow

范围：覆盖 MLFlowLearning/ 项目全部 10 个 phase + capstone 毕业项目 + 12 个 mlflow_skills

> **路径约定**：本文档所有命令里的 `<project-root>` 替换为你 clone 项目的实际路径（如 `~/projects/MLFlowLearning`、`/Users/you/code/MLFlowLearning`）。命令格式 `cd <project-root> && <cmd>` 意味着"先切到项目根目录，再执行命令"——**这只是建议**，脚本实际可以从**任何目录**运行（项目根、IDE 当前文件、phase 子目录都行）。每个 phase 脚本顶部都 `import _paths`，自动把 tracking URI 锚定到项目根的 `mlflow.db`——无论你在哪儿跑，UI 里看到的都是同一个数据库。

## 📖 如何使用本文档

**建议阅读顺序**：

```
1. 本开篇（你正在看）              ← 5 分钟，决定路线
2. Chapter 0-1（前置 + 安装）        ← 30 分钟，必须做
3. Chapter 2-4（核心传统 ML）        ← 2 小时，必学
4. Chapter 5-7（追踪 + GenAI + vibecoding）← 3 小时，必学
5. Chapter 8-11（提示词 + 评估 + 部署 + Agent）← 选学
6. Chapter 12-13（生产 + Debug）    ← 选学
7. Chapter 14（速查）+ Skill 段       ← 日常参考
```

**速查表**：

| 章节                       | 时间   | 难度       | 必学 | API Key |
| -------------------------- | ------ | ---------- | ---- | ------- |
| 0 前置认知                 | 5 min  | ★☆☆☆☆ | ✓   | 否      |
| 1 环境安装                 | 20 min | ★★☆☆☆ | ✓   | 否      |
| 2 核心对象与追踪           | 30 min | ★★☆☆☆ | ✓   | 否      |
| 3 Model 格式               | 30 min | ★★★☆☆ | ✓   | 否      |
| 4 注册表与别名             | 30 min | ★★★☆☆ | ✓   | 否      |
| 5 Tracking Server + 血缘   | 30 min | ★★★☆☆ | △   | 否      |
| 6 GenAI 追踪               | 30 min | ★★☆☆☆ | ✓   | ✅      |
| 7**vibecoding 集成** | 30 min | ★★★☆☆ | ✓   | -       |
| 8 Prompt Registry          | 30 min | ★★★☆☆ | △   | ✅      |
| 9 GenAI 评估               | 45 min | ★★★★☆ | △   | ✅      |
| 10 模型评估与部署          | 30 min | ★★★☆☆ | ✓   | 否      |
| 11 Agent Tracing           | 45 min | ★★★★☆ | △   | ✅      |
| 12 生产部署                | 30 min | ★★★★☆ | △   | 否      |
| 13 Debug 指南              | 20 min | ★★★☆☆ | ✓   | 否      |
| 14 速查表                  | 5 min  | ★☆☆☆☆ | ✓   | -       |
| Skill 段（12 个 skill）    | 30 min | ★★☆☆☆ | ✓   | -       |

**总计**：必学部分约 6 小时，选学 4 小时。

**约定**：

- 路径全部用 `<project-root>/`（项目根目录）
- 命令用 `$` 前缀（`$ conda activate mlflow`）
- 所有路径都用 `cd <project_root> && ...` 完整写

---

## 📋 目录

- [Chapter 0：前置认知——MLflow 是什么、解决什么问题](#chapter-0)
- [Chapter 1：环境安装——从零到第一个 UI](#chapter-1)
- [Chapter 2：核心对象与实验追踪](#chapter-2)
- [Chapter 3：Model 格式与 Model Registry](#chapter-3)
- [Chapter 4：注册表与冠军/挑战者模式](#chapter-4)
- [Chapter 5：Tracking Server + 数据血缘](#chapter-5)
- [Chapter 6：用一行代码追踪 LLM 调用](#chapter-6)
- [Chapter 7：Vibecoding 集成](#chapter-7)
- [Chapter 8：Prompt Registry](#chapter-8)
- [Chapter 9：GenAI 评估与 LLM-as-judge](#chapter-9)
- [Chapter 10：模型评估与本地部署](#chapter-10)
- [Chapter 11：Agent Tracing + LoggedModel](#chapter-11)
- [Chapter 12：生产级部署（选学）](#chapter-12)
- [Chapter 13：常见错误 Debug 指南](#chapter-13)
- [Chapter 14：参考速查](#chapter-14)
- [Skill 段：12 个 mlflow_skills 介绍](#skill-段)

---

<a id="chapter-0"></a>

# Chapter 0：前置认知——MLflow 是什么、解决什么问题

> ⏱️ 预计时间：5 分钟
> 🔑 是否需 API Key：否
> 📚 前置知识：无

## 🎯 这章做什么

读完这一章，你会理解：

- MLflow 是什么、解决什么问题
- 本项目能让你学到什么
- 应该按什么顺序学习

## 0.1 一句话定义

**MLflow 是一个机器学习实验管理平台**——它帮你把"调一次超参数、跑一次训练、产出一个模型"的全过程都记录下来，方便事后回看、对比、复现、部署。

如果你把每一次模型训练当成"做一道菜"，那 MLflow 就是你的**厨房日志**——

| 厨房概念        | MLflow 概念                  | 说明                                      |
| --------------- | ---------------------------- | ----------------------------------------- |
| 厨房            | **Experiment（实验）** | 一组相关训练的容器                        |
| 做菜的过程      | **Run（运行）**        | 一次训练的执行                            |
| 用什么食材/火候 | **Param（参数）**      | `learning_rate=0.01`, `batch_size=32` |
| 出品评分        | **Metric（指标）**     | `accuracy=0.95`, `loss=0.05`          |
| 成品照片        | **Artifact（产物）**   | 模型文件、图表、配置文件                  |
| 备注标签        | **Tag（标签）**        | `stage=baseline`, `team=alice`        |

## 0.2 MLflow 解决什么问题

没有 MLflow 时，ML 研究者的典型痛苦：

```
~/projects/awesome-model/
├── models/
│   ├── model_v1_acc0.91.pkl       # 哪个超参？忘了
│   ├── model_v2_acc0.93.pkl       # lr 是多少？0.001 还是 0.01？
│   ├── model_v3_acc0.89.pkl       # 训练集是不是这个？
│   └── ...                            # 老板问"上次那个 0.95 怎么训的？"
├── logs/
│   ├── train.log                  # terminal 输出，乱七八糟
│   └── ...                         # 时间戳混乱
└── notes.txt                       # 手写笔记，几个月后看不懂
```

有 MLflow 后：

```
MLflow UI:
Experiment: awesome-model
├─ Run: v1 (lr=0.01, batch=32)  acc=0.91
├─ Run: v2 (lr=0.001, batch=64) acc=0.93  ← 冠军
└─ Run: v3 (lr=0.01, batch=64)  acc=0.89
```

老板的问题 5 秒能答：哪个超参、什么指标、用了什么数据。

## 0.3 本项目能让你学到什么

完成本项目后，你能：

- ✅ 在 5 分钟内装好 MLflow 并跑通第一个实验
- ✅ 训练一个 sklearn 模型并自动记录所有参数/指标
- ✅ 在 UI 里对比 10 个模型的效果
- ✅ 把训练好的模型注册到 Registry，跨项目共享
- ✅ 用 `mlflow.sklearn.autolog()` 一行代码追踪 PyTorch / XGBoost / LightGBM
- ✅ 用 `mlflow.openai.autolog()` 追踪 LLM 调用
- ✅ 用 `mlflow.evaluate()` 自动算指标 + 出图（混淆矩阵、ROC）
- ✅ 用 `mlflow models serve` 部署模型为 REST API
- ✅ 用 `mlflow.genai.evaluate()` 系统化评估 LLM agent
- ✅ 用 coding agent（vibecoding）通过 mlflow_skills 自动化 MLflow 操作

## 0.4 学习路径

**如果你只做传统 ML（sklearn/xgboost/pytorch）**：

```
Chapter 1 → 2 → 3 → 4 → 5 → 10 → 13（必学）
Chapter 11 → 12（选学）
```

**如果你主要做 LLM/Agent**：

```
Chapter 1 → 2 → 6 → 7（必学）
Chapter 8 → 9 → 11（选学）
```

**如果你两边都做**：

```
完整 14 章节按顺序
```

## 0.5 MLflow 3 vs 2：必须知道的变化

MLflow 3（2025 年发布）相对 MLflow 2 做了**大量破坏性变化**。本项目**只用 MLflow 3 写法**。常见坑：

| MLflow 2（不要用）                                    | MLflow 3（本项目用）                                                        |
| ----------------------------------------------------- | --------------------------------------------------------------------------- |
| `mlflow.sklearn.log_model(..., artifact_path="m")`  | `mlflow.sklearn.log_model(..., name="m")`                                 |
| `transition_model_version_stage(..., "Production")` | `client.set_registered_model_alias(..., "champion", version)`             |
| `mlflow.evaluate(..., baseline_model=uri)`          | 两个`mlflow.models.evaluate()` + `mlflow.validate_evaluation_results()` |
| 模型在 Run 下（`runs:/<id>/<path>`）                | 模型独立（`models:/<model_id>`）                                          |
| Stage（Staging/Production）                           | Alias（champion/challenger）                                                |

⚠️ 如果你从 MLflow 2 文档复制代码，几乎必踩坑。

## 📖 下一步

→ [Chapter 1：环境安装——从零到第一个 UI](#chapter-1)

---

<a id="chapter-1"></a>

# Chapter 1：环境安装——从零到第一个 UI

> ⏱️ 预计时间：20 分钟
> 🔑 是否需 API Key：否
> 📚 前置知识：Chapter 0

## 🎯 这章做什么

跑完这一章，你会：

- 装好 conda env `mlflow`（包含 MLflow 3.x + 必要依赖）
- 启动 `mlflow ui` 看到空界面
- 知道整个项目的目录结构

## 1.1 装 conda（如果还没装）

```bash
# Linux/Mac 推荐装 Miniconda（轻量，~100MB）
# 下载：https://docs.conda.io/en/latest/miniconda.html
# 或用 mamba（更快）：https://mamba.readthedocs.io

# 验证
$ conda --version
# 期望：conda 23.x 或更新
```

## 1.2 创建专用 conda 环境

```bash
$ conda create -n mlflow python=3.11 -y
$ conda activate mlflow
```

⚠️ **不要装在 base 环境**——会污染系统 Python。每个项目独立 env 是好习惯。

## 1.3 安装 MLflow 和基础依赖

**推荐用项目自带的依赖文件**（一次性装全，不用记包名）：

```bash
# 方式 A：用 requirements.txt（pip 装）
$ pip install -r requirements.txt

# 方式 B：用 environment.yml（conda 一键建环境，更省心）
$ conda env create -f environment.yml
$ conda activate mlflow
```

装完验证：

```bash
# 验证
$ mlflow --version
# 期望：mlflow, version 3.15.1（或更新）
$ python -c "import mlflow; print(mlflow.__version__)"
# 期望：3.15.1
```

⚠️ **如果版本 < 3.0**：本项目用了 MLflow 3 的新 API，请升级：

```bash
$ pip install --upgrade mlflow
```

## 1.4 克隆项目

```bash
# 在你自己的工作目录下 clone
$ git clone https://github.com/LiangRichard13/mlflow-notes.git MLFlowLearning
$ cd MLFlowLearning

# 或如果你已经有项目目录
$ cd <project-root>
$ git pull  # 拉最新
```

## 1.5 启动 MLflow UI

```bash
$ mlflow ui --port 5000
# 看到 "Listening at: http://5000" 就 OK
```

打开浏览器：[http://localhost:5000](http://localhost:5000)

你应该看到一个空界面（"No experiments found"），因为还没有跑任何实验。

⚠️ **MLflow 3.5+ 必须配 `--allowed-hosts`**（防 DNS rebinding）：

```bash
$ mlflow ui --port 5000 --allowed-hosts "localhost,127.0.0.1"
```

⚠️ **端口被占用**：换端口：

```bash
$ mlflow ui --port 5001
```

## 1.6 项目结构

```
MLFlowLearning/
├── README.md                      ← 项目概览
├── .env.example                   ← API Key 模板
├── notes/00_quickstart.md         ← 你正在读的文件
├── scripts/01_basics/                     ← Phase 1 脚本：传统 ML 基础
├── scripts/02_registry/                   ← Phase 2：模型注册
├── scripts/03_tracking/                   ← Phase 3：Tracking Server
├── scripts/04_evaluate/                   ← Phase 4：评估与部署
├── scripts/05_tracing/                    ← Phase 5：GenAI 追踪
├── scripts/06_prompts/                    ← Phase 6：Prompt Registry
├── scripts/07_evaluation/                 ← Phase 7：GenAI 评估
├── scripts/08_agents/                     ← Phase 8：Agent + 版本化
├── scripts/09_deployment/                 ← Phase 9：生产部署
├── scripts/10_vision_classification/      ← Phase 10：图像分类（深度学习）
├── scripts/capstone/                      ← 毕业项目 SupportPilot
├── mlflow_skills/                  ← 12 个 MLflow skill（vibecoding 用）
└── notes/                         ← 学习笔记（11 个 phase + 3 个 appendix）
    ├── 01_basics.md
    ├── ...
    └── appendix/
        ├── roadmap.md
        ├── api_cheatsheet.md
        └── mlflow3_breaking_changes.md
```

## 1.7 （可选）Phase 10 额外依赖

如果你想做 Phase 10 图像分类实验，需要 torch + timm：

```bash
# 用项目里的 requirements-phase10.txt 一键装
$ pip install -r requirements-phase10.txt

# 或者手动装（国内加速：先设 huggingface 国内镜像）
$ export HF_ENDPOINT=https://hf-mirror.com
$ pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
$ pip install timm
```

⚠️ 这些是可选的，Phase 1-9 + capstone 不需要。

## 🛠️ 动手做

1. **跑通**上述 1.1-1.5 全部步骤
2. **验证**：浏览器能打开 `http://localhost:5000`，看到空 UI
3. **跑通项目第一个脚本**（热身，为 Chapter 2 做准备）：
   ```bash
   $ cd <project-root>
   $ conda run -n mlflow python scripts/01_basics/01_hello_mlflow.py
   ```

   应该看到 3 个 Run 创建（experiment 叫 `01_basics_demo`）

## 避坑清单

- ⚠️ **`mlflow` 命令找不到**：conda 环境没激活。检查 `which mlflow` 是否在 ``conda env list` 显示的 env 路径` 路径下
- ⚠️ **版本 < 3.0**：升级 `pip install --upgrade mlflow`
- ⚠️ **端口 5000 被占用**：换端口（`--port 5001`），但记得 UI 地址也要换
- ⚠️ **conda activate 后仍用系统 python**：检查 `which python` 应该指向 ``conda env list` 显示的 env 路径python`

## 📖 下一步

UI 装好 + 第一个脚本跑通后 → [Chapter 2：核心对象与实验追踪](#chapter-2)

# Chapter 2：MLflow 核心对象与实验追踪

> ⏱️ 预计时间：30 分钟
> 🔑 是否需 API Key：否
> 📚 前置知识：Chapter 1（环境安装）

## 🎯 这章做什么

调了一晚上超参数，第二天却想不起"上次 lr=0.01 的效果到底是多少、模型文件扔哪了"——这一章就是解决这个问题的。MLflow 像一个**实验记账本**，自动把每次训练的参数、指标、模型文件、备注标签归档。跑完这一章，你会得到两个实验（`01_basics_demo`、`01_sklearn_iris`）下的 7 个 Run，并学会用 UI 的 Compare 复盘调参。

### 你会学到什么

- 用 `log_param` / `log_metric` / `log_artifact` 手动记录一次训练
- 用 `mlflow.sklearn.autolog()` 一行代码自动记录 sklearn 训练
- 理解 6 个核心对象：Experiment / Run / Param / Metric / Artifact / Tag
- 启动 `mlflow ui`，用 Compare 功能对比多个 Run 的效果

### 前置

- 已完成 Chapter 1（环境安装），`mlflow` 环境可用
- 需要 API Key：否
- 已安装 `mlflow` 和 `scikit-learn`（`pip install mlflow scikit-learn`）

### 必跑脚本清单

| 脚本 | 一句话作用 | 前置 |
|------|-----------|------|
| `scripts/01_basics/01_hello_mlflow.py` | 纯手动 log，演示 6 个核心对象 | 无 |
| `scripts/01_basics/01b_sklearn_basics.py` | 用 autolog 记录 4 个真实 sklearn 模型 | 建议先跑 01 |

## 核心概念（一句话版）

- **Experiment（实验）**：一组相关 Run 的容器，就像项目文件夹（文件系统下对应 `mlruns/<exp_id>/` 目录）。
- **Run（运行）**：单次训练的日记，每次 `start_run()` 产生一个，有唯一 `run_id`。
- **Param（参数）**：字符串型配置，如 `lr=0.01`；同一 key 只能记一次（"一次定型"）。
- **Metric（指标）**：数值型效果，如 `accuracy=0.92`；可带 step，能画出训练曲线。
- **Artifact（产物）**：任意文件（模型、图表、配置），统一收在 Run 的 `Artifacts/` 下。
- **Tag（标签）**：任意备注文本，用于 UI 过滤/搜索，可随时改。
- **autolog**：`mlflow.<框架>.autolog()` 一行开启，自动记录 sklearn/pytorch/xgboost 等框架的参数、指标、模型文件与签名。

记住一句话：**Experiment 是文件夹，Run 是文件，Param/Metric/Artifact/Tag 是字段。**

## 🛠️ 动手做

1. **跑两个必跑脚本并进 UI 复盘**。依次执行：

```bash
cd <project-root> && python scripts/01_basics/01_hello_mlflow.py
cd <project-root> && python scripts/01_basics/01b_sklearn_basics.py
cd <project-root> && mlflow ui --port 5000
```

浏览器打开 `http://localhost:5000`，进 `01_sklearn_iris`，勾选全部 4 个 Run，点 **Compare**，看不同超参（C / max_depth / n_estimators）下 accuracy、f1 的差异。验证：能看到 4 个 Run 的并排对比表，差异列被高亮。

2. **改超参重跑对比**。把 `scripts/01_basics/01b_sklearn_basics.py` 里第 103 行附近的 `{"n_estimators": 200, ...}` 改成 `{"n_estimators": 50, ...}`，重新跑：

```bash
cd <project-root> && python scripts/01_basics/01b_sklearn_basics.py
```

再进 UI Compare 新老两个 `rf_deep` Run。验证：`01_sklearn_iris` 里出现 5 个 Run，Compare 视图下 `n_estimators` 列被高亮为差异列。

> 避坑速记：永远用 `with mlflow.start_run() as run:` 管理生命周期（异常自动结束 Run）；Param 同 key 不能覆盖，会变的值改用 `set_tag`；UI 和脚本都必须在 `<project-root>` 目录下跑，否则看不到数据（默认 backend 是 `file:./mlruns`）。完整清单见笔记。

## 📖 深入阅读（关键！）

完整原理、代码模式、全部避坑细节见对应笔记：

> 📚 [`notes/01_basics.md`](01_basics.md)（Phase 1：核心对象与实验追踪）

## 📖 下一步

→ [Chapter 3：MLflow Model 格式与 Model Registry](#chapter-3)

---
# Chapter 3：MLflow Model 格式与 Model Registry

> ⏱️ 预计时间：35 分钟
> 🔑 是否需 API Key：否
> 📚 前置知识：Chapter 2（核心对象与实验追踪）

## 🎯 这章做什么

Chapter 2 学会了"记录"，但训练出来的模型本身去哪了？这一章解决两个问题：**MLflow Model 格式**（把模型 + 依赖 + 输入输出说明打包成自描述目录，能脱离训练代码部署）和 **Model Registry**（模型的"版本仓库"，像 Git 之于代码）。跑完这一章，你会把 Wine 数据集训练的 Pipeline 记录为带签名和 `input_example` 的模型，并在 UI 里亲眼看到它的 `MLmodel` YAML。

### 你会学到什么

- 读懂 MLflow Model 目录结构，尤其是 `MLmodel` YAML 元数据在说什么
- 用 `infer_signature()` 自动推断模型输入输出 schema
- 理解 `input_example` 的作用（自动推断签名 + UI 展示 + 部署冒烟测试）
- 理解 MLflow 3 的变化：LoggedModel 独立于 Run、`name=` 取代 `artifact_path=`

### 前置

- 已完成 Chapter 2，理解 6 个核心对象
- 需要 API Key：否
- 已安装 `mlflow`（3.x）、`scikit-learn`、`pandas`
- **关键环境**：Registry 必须有数据库后端，本项目用 `sqlite:///mlflow.db`（文件系统 `file:./mlruns` 不支持 Registry）；脚本可以从任何目录运行（`import _paths` 自动锚定）

### 必跑脚本清单

| 脚本 | 一句话作用 | 前置 |
|------|-----------|------|
| `scripts/02_registry/02a_log_model.py` | 训练 Pipeline，推断签名 + input_example，用 `name=` 记录模型 | Chapter 2 |

## 核心概念（一句话版）

- **MLmodel YAML**：模型目录里的"身份证"，记录 flavors（加载方式）、依赖、签名、run_id，是模型能脱离训练代码部署的关键。
- **Signature（签名）**：模型输入输出 schema，`infer_signature(X_train, model.predict(X_train))` 一行生成；部署时自动校验请求，也是给人看的接口文档。
- **input_example**：几行真实输入样本；忘传 signature 时自动推断、UI 展示"请求长什么样"、部署后做冒烟测试。
- **LoggedModel（MLflow 3 新概念）**：模型成为独立一等公民，有自己的 `model_id`，不再寄生在 Run 下——删 Run 不误删模型，UI 的 `Models` 页可直接列出。

## 🛠️ 动手做

1. **跑 02a 记录带签名的模型**：

```bash
cd <project-root> && python scripts/02_registry/02a_log_model.py
```

验证：输出含 `模型性能: accuracy=1.0000, f1=1.0000` 和一行 `模型 URI: runs:/<run_id>/wine-classifier`。

2. **去 UI 看 model artifact 的 MLmodel YAML（必做）**：

```bash
cd <project-root> && mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

打开 `http://localhost:5000` → `Experiments` → `02_model_registry` → 打开 Run → `Artifacts` → 点开 `wine-classifier` 目录，**亲眼看一遍 `MLmodel` 文件**。验证：能看到 `flavors.python_function` 与 `flavors.sklearn` 两段、`signature`（13 列输入 + long 输出）、`run_id`；同目录下还有 `input_example.json`、`conda.yaml`、`requirements.txt`。

> 避坑速记：MLflow 3 用 `name=` 而不再用 `artifact_path=`；Stage API 已废弃，用 Alias 代替；版本号只增不减；URI 里的路径名必须和 `log_model` 的 `name` 完全一致。完整清单见笔记。

## 📖 深入阅读（关键！）

完整原理、三种 model URI 写法、pyfunc vs sklearn 加载区别见笔记；MLflow 3 模型侧破坏性变化见附录：

> 📚 [`notes/02_registry.md`](02_registry.md)（Phase 2：模型格式与注册表）
> 📚 [`notes/appendix/mlflow3_breaking_changes.md`](appendix/mlflow3_breaking_changes.md)（Model 相关部分：LoggedModel 一等公民、API 改名）

## 📖 下一步

→ [Chapter 4：模型注册表与冠军/挑战者模式](#chapter-4)
# Chapter 4：模型注册表与冠军/挑战者模式

> ⏱️ 预计时间：45 分钟
> 🔑 是否需 API Key：否
> 📚 前置知识：第 3 章（Phase 1：记录参数与指标）

## 🎯 这章做什么

第 3 章你学会了"记录"——把参数、指标写到 MLflow，训练完能回头查。但模型本身呢？三个月后同事问你"线上跑的是哪个模型、当时用什么数据训的、输入要几列",你多半答不上来。更糟的是，你想换个更好的模型上线，得手动改代码里的文件路径，再重启服务。

**这一章解决的就是"模型怎么管"的问题。** MLflow 提供了 **Model Registry**——一个版本化的"产品货架"。日记里写几百次实验（Run），货架上只放你精挑细选、贴好标签的那几个。**Alias（别名）** 就是货架上的标签牌——"champion"（现役冠军）这块牌子今天挂在 v1 上，明天可以挂到 v3 上，所有来取货的人（加载模型的服务）自动拿到新版本，**不用改一行代码、不用重启**。

**学完这章你会的硬技能：**

- 把 Run 里的模型注册到 Model Registry，理解版本号怎么自动累加
- 用 champion/challenger alias 做"线上冠军/灰度挑战者"
- 解释清楚为什么 MLflow 3 抛弃了 Stage、新接 Alias
- 三种 model URI 写法选哪个

### 你会学到什么

- 把 Run 里的模型"提升"成 Registered Model（`mlflow.register_model`）
- 用 `set_registered_model_alias()` 替代已废弃的 `transition_model_version_stage`
- 区分 champion / challenger / archived 三种角色
- 掌握三种 model URI：`runs:/`、`models:/<name>/<version>`、`models:/<name>@<alias>`
- 体验"一秒钟切生产模型"的零停机回滚

### 对应脚本清单

| 脚本                      | 作用                                               | 是否必跑     | 前置     |
| ------------------------- | -------------------------------------------------- | ------------ | -------- |
| `02a_log_model.py`      | 训练 sklearn 模型 + 推断签名 + 记到 Run            | 必跑（上章） | Phase 1  |
| `02b_register_alias.py` | 把 Run 提升为 Registered Model + 设 champion alias | 必跑         | 跑过 02a |
| `02c_load_predict.py`   | 用`models:/name@champion` 加载并推理             | 必跑         | 跑过 02b |

---

## 核心概念

### 1. Model Registry：模型的"Git 仓库"

Run 里的模型是**实验产物**——你可能跑了 200 次，其中 199 次都是垃圾。Registry 是**发布通道**——你从那 200 次里挑出好的，给它起个正式名字（`WineQualityClassifier`），它就有了 v1、v2、v3 的版本序列。

打个比方：

| 概念             | 类比                    | 特点                                               |
| ---------------- | ----------------------- | -------------------------------------------------- |
| Run 里的模型     | 本地的一次 commit       | 数量多，随手产生，用`runs:/<run_id>/<name>` 引用 |
| Registered Model | 一个 Git 仓库           | 有名字，是一个逻辑上的"产品线"                     |
| Model Version    | 打的 tag（v1、v2）      | 注册一次自动 +1，**不可变**                  |
| Alias            | 指向某个 tag 的分支指针 | 可以随时改指向，`champion` → v3                 |

**版本号只增不减**，删了 v2 之后，下次注册是 v3 而不是补上 v2。所以版本号可以放心当唯一标识用。

### 2. 为什么 Model Registry 必须有数据库后端

Registry 不是文件，是一堆带关系的表（哪个名字有哪些版本、版本挂了哪些别名、版本跟哪个 Run 关联）。这必须用数据库存。

```
✗ file:./mlruns       → 纯文件系统，调 register_model 直接报错
✓ sqlite:///mlflow.db → 小团队、个人项目（本章方案）
✓ postgresql://...    → 生产、跨机器
```

如果你用 `file:./mlruns` 跑 `register_model`，会直接看到 `RESOURCE_DOES_NOT_EXIST` 或 `registry` 相关的报错。**这是 MLflow 3 的硬性要求**，不是 bug。

### 3. ⭐ Stage 退役，Alias 上位（新手最困惑的点）

**先说结论：MLflow 2 时代的 Stage（`None` / `Staging` / `Production` / `Archived`）在 MLflow 3 里已经废弃，取而代之的是 Alias。** 如果你在网上搜到 `transition_model_version_stage(...)` 的教程，那是旧写法——别学。

**Stage 的三个硬伤：**

1. **写死的四个值，改不了**。现实里团队的流程五花八门：有人要 `dev` / `qa` / `canary` / `prod` 四级，有人做 A/B 测试要同时上两个模型。Stage 只给你四个固定选项，全都塞不下。Alias 是**自定义字符串**，你想叫什么叫什么。
2. **一个 stage 只能挂一个版本，一个版本只能有一个 stage**。这个 1 对 1 的死限制让 A/B 测试特别难做——你没法说"v2 和 v3 同时是生产模型"。Alias 是**多对多**的：一个版本可以同时挂 `champion` 和 `stable`，你也可以加 `challenger` 挂到 v3 上做灰度。
3. **语义模糊**。"Production" 到底是"正在生产环境跑"还是"通过了测试可以上生产"？不同团队理解不一样。Alias 强迫你自己命名，反而更明确。

**Alias 的核心好处——热切换：**

生产服务里你的加载代码写死一行：

```python
model = mlflow.sklearn.load_model("models:/WineQualityClassifier@champion")
```

新模型 v2 上线时，你**不改代码、不重启服务**，只要执行：

```python
client.set_registered_model_alias("WineQualityClassifier", "champion", version=2)
```

下一次加载就自动是 v2 了。要回滚？把别名指回 v1，一秒钟的事。这个切换是**原子操作**，不存在"改到一半"的中间状态。

### 4. 冠军/挑战者模式（Champion / Challenger）

最简单的模型发布模式：**一个在用、一个在测**。

| Alias          | 含义                               | 跑在线上吗         |
| -------------- | ---------------------------------- | ------------------ |
| `champion`   | 当前生产在用的冠军模型             | 是                 |
| `challenger` | 正在评测、准备替换 champion 的候选 | 灰度流量或离线评估 |
| `baseline`   | 用于对比的基准模型                 | 否                 |
| `archived`   | 已下线但保留，方便回溯             | 否                 |

**完整发布流程：**

1. 训练出新模型 v2，跑完离线评估觉得"有戏"
2. 把 v2 注册到 Registry，注册时贴 `challenger` alias
3. 在小流量（5%）上对请求做 A/B：5% 走 `challenger`，95% 走 `champion`
4. 线上指标稳定优于 champion → 把 `champion` 从 v1 移到 v2
5. 老的 v1 不删，挂 `archived` 留痕迹

**全程没有服务重启、没有代码改动。** 改 alias 指向就等于换线上模型。

### 5. 三种 Model URI：什么时候用哪个

```python
# 1) runs:/<run_id>/<path> —— 从 Run 加载
# 调试期用，URI 里的 run_id 很长且不好记
model = mlflow.sklearn.load_model(f"runs:/abc123def456/wine-classifier")

# 2) models:/<name>/<version> —— 用版本号加载
# 需要精确复现某个版本时用（debug 线上问题）
model = mlflow.sklearn.load_model("models:/WineQualityClassifier/1")

# 3) models:/<name>@<alias> —— 用别名加载（⭐ 生产推荐）
# 代码写死不用改，alias 指向变了，自动用新版本
model = mlflow.sklearn.load_model("models:/WineQualityClassifier@champion")
```

**速记口诀**：调试用 `runs:/`，复现用 `models:/n/v`，生产用 `models:/n@alias`。

---

## 实战步骤

### Step 1 — 训练并记录模型（如果没跑过 02a）

```bash
conda activate mlflow
cd <project-root> && python scripts/02_registry/02a_log_model.py
```

预期输出里有 `模型性能: accuracy=1.0000, f1=1.0000`（Wine 数据集很简单，满分正常），还会打印一行 `模型 URI: runs:/<run_id>/wine-classifier`，**把这个 run_id 记下来**。

> ⚠️ **必须确认数据库后端**：02a 通过 `import _paths` 自动把 tracking URI 设成 `sqlite:///<project-root>/mlflow.db`（绝对路径，不依赖 cwd）。如果你看到 `mlruns/` 目录被创建到了子目录里而不是项目根，说明 `_paths` 没起作用，**Registry 一会儿会报错**。

### Step 2 — 注册 + 设别名

```bash
cd <project-root> && python scripts/02_registry/02b_register_alias.py
```

预期看到两行关键输出：

```
✓ 已注册为 WineQualityClassifier v1
✓ 已设置 champion alias → v1
```

最后会打印版本列表和别名映射（`{'champion': '1'}` 这种）。

**脚本都干了什么：**

```python
# 1. 找到最近一次 Run
runs = mlflow.search_runs(
    experiment_names=["02_model_registry"],
    order_by=["start_time DESC"],
    max_results=1,
)
model_uri = f"runs:/{runs.iloc[0].run_id}/wine-classifier"

# 2. 注册：同名重复注册 = 自动创建新版本
result = mlflow.register_model(model_uri, "WineQualityClassifier")
version = result.version   # 第一次注册 = 1

# 3. 设别名（替代已废弃的 stage）
client.set_registered_model_alias("WineQualityClassifier", "champion", version=version)

# 4. 加描述，UI 上能看到
client.update_model_version(
    name="WineQualityClassifier",
    version=version,
    description="StandardScaler + RandomForest(n=200, depth=10) on Wine dataset. accuracy=1.0",
)
```

### Step 3 — 用别名加载并推理

```bash
cd <project-root> && python scripts/02_registry/02c_load_predict.py
```

预期看到 `Pipeline steps: ['scaler', 'clf']`，以及 5 个预测标签和真实标签的对比（Wine 简单，应该全对）。

**核心代码就一行：**

```python
model = mlflow.sklearn.load_model("models:/WineQualityClassifier@champion")
preds = model.predict(X_test.head(5))
```

加载回来的是**完整的 Pipeline**，StandardScaler 和 RandomForest 两步都在，直接喂原始数据即可。

### Step 4 — 去 UI 验证（别跳过）

```bash
cd <project-root> && mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

打开 `http://localhost:5000`，看：

1. **左侧导航栏点 `Models`** → 看到 `WineQualityClassifier`
2. **点进去看 Version 1**，重点看三处：
   - **Aliases** 那一栏显示 `champion`
   - **Description** 是 02b 里 `update_model_version` 写进去的那段文字
   - **Source Run** 链接能跳回 02a 的那次 Run
3. 从 `Experiments` → `02_model_registry` → 打开 Run → `Artifacts` tab → 点开 `wine-classifier` 目录，亲眼看一眼 `MLmodel` 文件

---

## 🛠️ 动手做：完整跑一遍"上线新版本 + 回滚"

光看代码不够，把这个流程亲手动一遍才能体会 alias 的威力。

**Step 1：自己生成 v2**

```bash
conda activate mlflow
# 用 sed 改 02a 里的 n_estimators: 200 → 50
cd <project-root> && sed -i 's/n_estimators=200/n_estimators=50/' scripts/02_registry/02a_log_model.py
cd <project-root> && python scripts/02_registry/02a_log_model.py
cd <project-root> && python scripts/02_registry/02b_register_alias.py
```

预期看到：

```
✓ 已注册为 WineQualityClassifier v2
✓ 已设置 champion alias → v2
```

**v2 出现了，冠军自动跳到了 v2。** 此时 `champion` 指向 v2，v1 上没有任何 alias。

**Step 2：手动把 champion 切回 v1，模拟回滚**

新开一个终端（或在 Python 里手动跑）：

```bash
cd <project-root> && python -c "
from mlflow import MlflowClient
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow.db')
MlflowClient().set_registered_model_alias('WineQualityClassifier', 'champion', version=1)
print('已回滚: champion → v1')
"
```

**Step 3：跑 02c，看预测用的是哪个版本**

```bash
cd <project-root> && python scripts/02_registry/02c_load_predict.py
```

代码里的 `models:/WineQualityClassifier@champion` **完全没改**，但自动加载到了 v1。再跑一次 Step 2 把 champion 改回 v2，02c 又会加载 v2。

**这就是零停机切换的核心演示。** 你的服务代码里没有版本号、没有 `if version == 1`、`if version == 2`，全靠 alias 跳转。

**Step 4（可选）：跑一次灰度——同时挂两个 alias**

```python
from mlflow import MlflowClient
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()
client.set_registered_model_alias("WineQualityClassifier", "champion", version=1)
client.set_registered_model_alias("WineQualityClassifier", "challenger", version=2)
```

现在 v1 同时挂着 `champion`，v2 同时挂着 `challenger`。你的 A/B 测试代码就可以：

```python
if user_id % 100 < 5:
    model = mlflow.sklearn.load_model("models:/WineQualityClassifier@challenger")  # 5% 灰度
else:
    model = mlflow.sklearn.load_model("models:/WineQualityClassifier@champion")    # 95% 主流量
```

灰度没问题后，把 `champion` 移到 v2 就完成上线。整个过程 `models:/name@champion` 这串代码一个字没改。

---

## 避坑清单

- ⚠️ **坑 1：Registry 必须有数据库后端**。用 `file:./mlruns` 时调 `register_model` 会直接报错。本项目统一 `sqlite:///mlflow.db`。脚本通过 `import _paths` 自动锚定到项目根 db，所以**可以从任何目录运行**，不会在别处生成孤儿 db。
- ⚠️ **坑 2：Stage 相关 API 全部废弃**。`transition_model_version_stage()`、`stage="Production"` 这类写法不要再用，一律换成 `set_registered_model_alias()`。MLflow 3 的 UI 里也已经看不到 Stage 下拉框了。如果你在网上看到 2024 年之前的教程，全部按"过期"对待。
- ⚠️ **坑 3：02b 用 `search_runs` 取"最近一次 Run"有隐患**。如果你在跑完 02a 之后又在 `02_model_registry` 这个实验里跑了别的 Run，02b 会注册错的那个。稳妥做法是显式指定 run_id，或者加上 run_name 过滤：

  ```python
  runs = mlflow.search_runs(
      experiment_names=["02_model_registry"],
      filter_string="attributes.run_name = 'wine-rf-v1'",
      order_by=["start_time DESC"],
      max_results=1,
  )
  ```
- ⚠️ **坑 4：model URI 里的路径名必须和 `log_model` 的 `name` 完全一致**。02a 写的是 `name="wine-classifier"`，02b 就必须拼 `runs:/{run_id}/wine-classifier`。写错一个字符就是 `RESOURCE_DOES_NOT_EXIST`，错误信息不会告诉你"你是不是拼错了名字"。
- ⚠️ **坑 5：别名区分大小写，且不能用作纯数字**。`Champion` 和 `champion` 是两个不同的别名；别名也不能起成 `1`、`2` 这种，会和版本号语法冲突。
- ⚠️ **坑 6：签名太严格也会咬人**。如果推断签名时用的是 DataFrame（有列名），那推理时也必须传 DataFrame，传 numpy 数组会因为缺列名而校验失败。保持训练和推理的数据形态一致。
- ⚠️ **坑 7：第一次用 sqlite + Registry 时跑一次 `mlflow db upgrade`**。如果 `register_model` 报 `no such table: registered_models`，原因是 mlflow.db 是旧版本 schema。执行：

  ```bash
  cd <project-root> && mlflow db upgrade sqlite:///mlflow.db
  ```
- ⚠️ **坑 8：02b/02c 文件头注释里写的是 `python 03_registry/...`，这是笔误**，正确目录是 `scripts/02_registry/`。脚本本身能跑，但别照抄错路径。

---

## 下一步

到这里你已经会注册模型、用别名管理版本了。但有个问题没解决：**怎么在团队里共享？** 现在的 MLflow 还是你本地的 SQLite 文件，别人连不上。

**Chapter 5** 我们会启动一个 **Tracking Server**——把 MLflow 升级成 HTTP 服务，让团队所有人往同一个后端写。同时还会讲**数据集血缘**——给训练数据发"身份证"，让任何一个模型都能追溯"它吃了什么数据"。

详细理论参考：`notes/02_registry.md`

---

# Chapter 5：追踪服务器与数据集血缘

> ⏱️ 预计时间：50 分钟
> 🔑 是否需 API Key：否
> 📚 前置知识：第 4 章（Model Registry + Alias）

## 🎯 这章做什么

第 4 章你学会了注册模型、用 alias 切换生产版本。但所有这一切都跑在你自己电脑的 SQLite 文件上——团队里张三、李四、CI 流水线、调度平台呢？他们怎么共享同一个 MLflow？

**这一章解决两个问题：**

1. **团队共享**：把 MLflow 升级成 HTTP 服务（Tracking Server），让所有人往同一个后端写
2. **数据血缘**：训练数据是模型的"食材"，出问题时要能立刻回答"这个模型吃了什么数据"。MLflow 让每个 Run 跟输入的数据集绑死，谁、什么时间、用哪份数据、跑出多少分，全留痕

> **类比**：第 4 章的 MLflow 像你一个人用笔记本写实验记录；这一章的 MLflow 像公司给你配了 **GitLab**（追踪服务器）+ **数据血缘**（像 Git 里每行代码都带作者和 commit）。

**学完这章你会的硬技能：**

- 启动一个 Tracking Server，让多个客户端往同一个 SQLite 库写
- 区分 Backend Store（元数据）和 Artifact Store（文件）
- 用 `mlflow.data.from_pandas` + `mlflow.log_input` 给 Run 关联训练数据
- 用 MLflow 3 的 `search_logged_models` 跨实验 SQL 风格搜索模型

### 你会学到什么

- 能启动一个 Tracking Server，让团队共享
- 知道 Backend Store 三种选型（file/sqlite/postgres）的取舍
- 理解 **file:// 致命问题**：纯文件后端 + Registry 直接报错
- 用 `mlflow.data.from_pandas()` 给数据发"身份证"（含 digest 哈希）
- 用 MLflow 3 的 `search_logged_models` 跨实验、跨参数、跨指标筛选模型

---

## 核心概念

### 1. Tracking Server：团队协作的"实验中心"

之前的脚本都是直接写本地文件（`./mlruns/` 目录）。一旦团队大于 1 人，就要一个 **HTTP 服务** 让所有人往同一个地方写。这就是 **Tracking Server**：一个 FastAPI 进程，对外暴露 REST API 和 UI。

启动后，所有 `mlflow.log_*` 调用都自动走 HTTP，根本不用改业务代码——只要环境变量 `MLFLOW_TRACKING_URI` 指过去就行。

```
┌──────────────┐         ┌──────────────────┐
│  你的脚本    │ ──HTTP─>│  Tracking Server │ ──> mlflow.db (SQLite)
│  (客户端)    │         │  (5000 端口)     │ ──> mlruns/   (Artifact)
└──────────────┘         └──────────────────┘
                              │
                              │
┌──────────────┐              │
│ 同事的脚本    │ ──HTTP──────┘
│ (另一台机器)  │              同一个 server，写同一个 db
└──────────────┘
```

### 2. Backend Store vs Artifact Store：账本 vs 仓库

Tracking Server 自己不存数据，它把数据分两家：

| 概念                     | 类比                   | 存什么                                            | 选型                                 |
| ------------------------ | ---------------------- | ------------------------------------------------- | ------------------------------------ |
| **Backend Store**  | 餐厅的点菜单（数据库） | experiments、runs、metrics、params、tags、aliases | SQLite（小团队）、PostgreSQL（生产） |
| **Artifact Store** | 餐厅的仓库（文件系统） | 模型文件、图片、配置、特征文件                    | 本地路径（学习）、S3/MinIO（生产）   |

为什么要分？因为数据库擅长"频繁小写入"（每次 metric 都写一行），文件系统擅长"大文件顺序读写"（模型权重几百 MB）。混在一起两者都做不好。

**三种 Backend Store 选型对比：**

```
file://（纯本地文件）
  ✗ 只支持 mlflow.log_* 等基本 API
  ✗ Model Registry 直接报错（致命问题！）
  ✗ 跨机器无法共享
  适用：单次实验性的 explore

sqlite://（文件数据库）
  ✓ 团队在小范围共享（同一台机器 / NFS）
  ✓ 完全支持 Registry
  ✓ 零运维，存一个文件
  ✗ 写并发有限（多人同时写偶尔锁表）
  适用：学习、小团队（< 5 人）

postgresql:// / mysql://（服务数据库）
  ✓ 高并发、支持几百客户端同时写
  ✓ 跨服务器、跨机房
  ✗ 需要运维：一台 PG 实例 + 备份
  适用：生产环境
```

### 3. ⚠️ file:// 的致命问题：Registry 不能用

**这是本章最重要的一条规则：**

> **用了 Model Registry → 必须用 sqlite 或 postgres，不能用 file://**

如果你用 `mlflow.set_tracking_uri("file:./mlruns")` 然后调 `register_model`，会直接报：

```
MlflowException: No suitable backend store to use for the registry
```

或者 `RESOURCE_DOES_NOT_EXIST` 类似的错误。本项目从第 4 章开始统一用 `sqlite:///mlflow.db`（通过 `scripts/_paths.py` 锚定到项目根）。所有脚本顶部都是 `import _paths`，不需要在脚本里手写 `set_tracking_uri`。**这一行不是装饰，是功能要求。**

### 4. 数据集血缘：给训练数据发"身份证"

每次训练时，不光要记录参数和指标，还要告诉 MLflow "我用了这份数据"。`mlflow.data.from_pandas(df, source=..., name=..., targets=...)` 创建一个 `Dataset` 对象，里面带四样东西：

| 字段             | 含义                     | 举例                                                  |
| ---------------- | ------------------------ | ----------------------------------------------------- |
| **source** | 数据来自哪个文件/URL/库  | `"data/wine.csv"`、`"sklearn.datasets.load_wine"` |
| **name**   | 你给数据集起的名         | `"wine_dataset"`                                    |
| **digest** | 数据集内容的哈希（指纹） | `"a3f5..."`                                         |
| **schema** | 列名 + 类型              | `[{"name": "alcohol", "type": "double"}, ...]`      |

**digest 是关键防伪**：一模一样的数据 → 同一个 digest；只要改一个字节 → digest 全变。这就是"防偷偷换数据"的硬证据。

`mlflow.log_input(dataset, context="training")` 把这个"身份证"挂在 Run 上，从此 Run 知道自己吃了什么数据长大。

**反向追溯**（出问题时用）：

```python
client = mlflow.MlflowClient()
run = client.get_run(run_id)
for ds_input in run.inputs.dataset_inputs:
    ds = ds_input.dataset
    print(ds.name, ds.digest, ds.source)
```

### 5. LoggedModel：MLflow 3 把"模型"独立成"一等公民"

MLflow 2 里，模型是 Run 下的一个 artifact（`runs:/<run_id>/model`）。MLflow 3 把模型抽出来当独立对象，叫 **LoggedModel**，有自己的 `model_id`，可以跨 Run 跨 Experiment 引用（`models:/<model_id>`）。

为什么这事重要？因为**搜索模型**不该受 Run 的束缚。一个超参搜索可能产生几百个 Run，但模型本身才是你关心的产物。`search_logged_models` 就是为了这个——SQL 风格筛选，秒级返回。

### 6. MLflow 3 vs 2 关键差异（速查）

| 维度               | MLflow 2                           | MLflow 3                            |
| ------------------ | ---------------------------------- | ----------------------------------- |
| 模型 URI           | `runs:/<id>/<path>`              | 加`models:/<model_id>` 跨实验引用 |
| `log_model` 参数 | `artifact_path="..."`            | `name="..."`（强制改名）          |
| 阶段切换           | `transition_model_version_stage` | `set_registered_model_alias`      |
| 搜索模型           | 只能用`search_runs`              | 新增`search_logged_models`        |
| UI 入口            | 模型藏在 Run 里                    | 左侧栏独立`Logged Models`         |
| 服务器             | --                                 | 3.5+ 必须配`--allowed-hosts`      |

---

## 实战步骤

### Step 1：启动 Tracking Server（新开一个终端）

```bash
conda activate mlflow
cd <project-root> && bash scripts/03_tracking/03a_start_server.sh
```

脚本会执行：

```bash
mlflow server \
  --backend-store-uri sqlite:///$(pwd)/mlflow.db \
  --default-artifact-root $(pwd)/mlruns \
  --host 0.0.0.0 \
  --port 5000
```

启动后你会在当前目录看到两个东西被创建/使用：

- `mlflow.db`（SQLite 数据库，账本）
- `mlruns/`（artifact 文件夹，仓库）

看到 `Listening at: http://0.0.0.0:5000` 就 OK。

> **MLflow 3.5+ 必加 `--allowed-hosts "localhost,127.0.0.1"`**，不然浏览器可能打不开（防 DNS rebinding 攻击）。如果跑不通，先停掉，然后用以下命令手启：
>
> ```bash
> cd <project-root> && mlflow server \
>   --backend-store-uri sqlite:///$(pwd)/mlflow.db \
>   --default-artifact-root $(pwd)/mlruns \
>   --host 0.0.0.0 --port 5000 \
>   --allowed-hosts "localhost,127.0.0.1"
> ```

### Step 2：让客户端连上 server（再开一个终端）

```bash
conda activate mlflow
export MLFLOW_TRACKING_URI=http://localhost:5000
cd <project-root>
```

**不设这个变量，脚本会默认写本地 `./mlruns`，跟 server 不通。** 这一行是连接 server 的钥匙。

> 备选：如果你不想 export 环境变量，也可以在每个脚本里写 `mlflow.set_tracking_uri("http://localhost:5000")`——但环境变量更省事，全局生效。

### Step 3：跑数据集血缘脚本

回到有 `MLFLOW_TRACKING_URI` 的终端：

```bash
cd <project-root> && python scripts/03_tracking/03b_dataset_lineage.py
```

控制台会打印：

```
数据集已保存: /tmp/xxx.csv (178, 15)
数据集元数据:
  name: wine_dataset
  source: /tmp/xxx.csv
  digest: a3f5e8d2c9b1...   ← 数据指纹
  schema: ...

✓ Run 已记录: 7c8d9e0f
  accuracy: 1.0000
  dataset digest: a3f5e8d2c9b1...
  关联的数据集: training + testing
```

**注意：03b 在脚本里显式写了 `mlflow.set_tracking_uri("sqlite:///mlflow.db")`，所以即使你没设 `MLFLOW_TRACKING_URI` 它也能跑（写本地 mlflow.db）。但要想真正尝试连 server，去掉那行或改成 `http://localhost:5000`。**

### Step 4：跑搜索 LoggedModel 脚本

```bash
cd <project-root> && python scripts/03_tracking/03c_search_logged_models.py
```

它训练 5 个不同模型（不同 C、深度），然后用 `search_logged_models` 筛三种条件。控制台会打印三张表，对应：

- "accuracy > 0.95" 的所有模型
- "name 含 classifier 且 accuracy > 0.95"
- "accuracy > 0.95 且 n_estimators = 50"

最后会找到最佳模型并 `mlflow.sklearn.load_model(f"models:/{best.model_id}")` 加载推理。

### Step 5：UI 验证（必看）

浏览器开 `http://localhost:5000`，按顺序看：

1. **`Models` 页面**：之前在第 4 章注册的 `WineQualityClassifier` 还在，champion alias 指向 v1（或 v2，看你切了几次）
2. **左侧导航栏的 `Logged Models`**（MLflow 3 新增的栏目）→ 看跨实验的所有 LoggedModel 卡片
3. **选 experiment `03_dataset_lineage`** → 点开 Run `rf-with-dataset`（训练那个）
4. **Datasets 标签**（关键新栏目）：列出 `wine_dataset (training)` 和 `wine_test_split (testing)` 两行，每行带 digest
5. **Tags 标签**：`data_snapshot = <digest>` 我们用 `set_tag` 留了一份快照
6. **进 experiment `03_search_demo`**（`03c` 跑完才有）→ 左边 `Logged Models` → 看 5 个模型按 accuracy 排序的卡片

---

## 🛠️ 动手做：完整验证数据集血缘

**目标：亲手验证 digest 在 UI 上能帮你发现"数据被偷换"**

**Step 1：跑 03b，看 digest**

```bash
cd <project-root> && python scripts/03_tracking/03b_dataset_lineage.py
```

记下控制台输出的 `dataset.digest` 一长串哈希（比如 `a3f5e8d2c9b1...`）。

**Step 2：去 UI 看**

打开 `http://localhost:5000` → Experiments → `03_dataset_lineage` → Run `rf-with-dataset` → **Datasets** 标签。

你看到 `wine_dataset (training)` 那一行，右边的 digest 字段就是刚才记下的那串哈希。

**Step 3：手动改数据，验证 digest 会变**

打开 03b 脚本，找到：

```python
df = pd.DataFrame(wine.data, columns=wine.feature_names)
```

把它改成：

```python
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["alcohol"] = df["alcohol"] + 0.0001   # 偷偷改一列
```

重新跑：

```bash
cd <project-root> && python scripts/03_tracking/03b_dataset_lineage.py
```

**digest 大概率变了**（除非你幸运地没改到哈希敏感的字段，但基本不可能）。这就是"防偷偷换数据"的硬证据——审计部只要对一下两个版本的 digest，立刻知道你改了数据。

**Step 4：故意绕开 digest**

把 03b 里的 `mlflow.log_input(dataset, context="training")` **注释掉**——再跑一次。UI 的 Datasets 标签就空了。这就是"不挂身份证"的代价：模型不知道自己吃了什么数据长大。

---

## 避坑清单

- ⚠️ **坑 1：忘了 `export MLFLOW_TRACKING_URI`** → 写本地 `mlruns`，server 看不到。脚本里没看到就在 Python 里加 `mlflow.set_tracking_uri("http://localhost:5000")`。
- ⚠️ **坑 2：想用 Model Registry 但用的是文件 backend** → `register_model` 直接报错。必须先切到 sqlite 或 postgres。**这就是 file:// 的致命问题**——本项目以 `sqlite:///mlflow.db` 为基线。
- ⚠️ **坑 3：`search_logged_models` 用 `params.lr <= 0.01`** → 报错。`params` 是字符串，只支持 `=`、`!=`、`LIKE`、`IN`；数值比较（`<`、`>`、`<=`、`>=`）只对 `metrics` 有效。要筛数值超参，先用 `log_metric` 把它记成 metric，或者用 `params.xxx = '50'` 当字符串比对。
- ⚠️ **坑 4：`experiment_ids=["03_search_demo"]` 传名字** → 返回空。`search_logged_models` 接收的是 id 不是 name，必须先用 `mlflow.get_experiment_by_name(...).experiment_id` 拿到 id。
- ⚠️ **坑 5：改了数据但忘了 digest 检测** → 每次跑新实验 MLflow 都会重算 digest，变了就说明数据被改过——别绕开它。如果发现 digest 变了但你"没改数据"，八成是数据源本身在变（CSV 被覆盖、数据库连接拿的是不同快照），这是排查训练-推理不一致的关键线索。
- ⚠️ **坑 6：环境变量作用域**。`export MLFLOW_TRACKING_URI=...` 只在当前终端窗口生效。新开一个终端要重新 export，或者把它加到 `~/.bashrc`。
- ⚠️ **坑 7：MLflow 3.5+ 浏览器打不开**。原因是没有 `--allowed-hosts`。新版本出于安全考虑默认拒绝外部 host。启动 server 时加上 `--allowed-hosts "localhost,127.0.0.1"`。
- ⚠️ **坑 8：scripts 里 `mlflow.set_tracking_uri` 覆盖了环境变量**。如果你既 export 了又脚本里写，**脚本里的生效**（因为它后调用）。所以 03b 的 `set_tracking_uri("sqlite:///mlflow.db")` 会让脚本绕过 server 直接写本地 db——这是该脚本的设计（"不需要启 server"），但你要清楚。

---

## 下一步

到这里你已经会：

- 启动 Tracking Server，让团队共享同一个 MLflow
- 选 Backend Store（file/sqlite/postgres）并知道 file 用不了 Registry
- 用 `mlflow.data.from_pandas` + `mlflow.log_input` 给 Run 关联训练数据
- 用 `search_logged_models` 跨实验 SQL 风格搜索模型

**但模型还没真正上线。** 现在你加载模型用的是 Python 脚本调用，生产环境通常是 HTTP 服务（线上系统 POST 请求过来，模型返回预测）。

**Chapter 6** 会讲：

- `mlflow models serve` 启动一个 REST API
- 用 `curl` 调模型
- 打 Docker 镜像、部署到生产环境
- 评估模型：混淆矩阵、ROC、AUC 等指标在 MLflow 里怎么记

详细理论参考：`notes/03_tracking.md`

# Chapter 6：用一行代码追踪 LLM 调用——理解 Trace / Span / SpanType

> ⏱️ 预计时间：45 分钟
> 🔑 是否需 API Key：是（DeepSeek 或其他 OpenAI 协议兼容的服务商）
> 📚 前置知识：第 1-5 章（必须会 `mlflow ui` 和 `mlflow.set_experiment`）

## 🎯 这章做什么

从这一章开始我们进入 GenAI / LLM 应用的世界。前面几章你可能在训 scikit-learn 模型——一次训练就是一个明确的过程。LLM 应用不一样：一次"推理"往往是一连串步骤——查向量库、组装 prompt、调大模型、解析输出，有时候还要循环或重试。出了 bug 你根本不知道是哪一步慢、哪一步错。

MLflow 的 **Tracing（追踪）** 就是为了解决这个问题的。它能把你代码里**每一次调用**都自动记下来（不管是 LLM 调用、向量库检索、还是你自己写的函数），形成一棵"调用树"（Span 树）。出问题的时候点开 UI 就能看到底是哪一步出错、花了多少时间、花了多少 token。

这一章你会做的是：用一个最简单的脚本实现"一行代码追踪所有 LLM 调用"，并在 UI 里看一棵 Span 树。重点不是写新代码，而是**学会看 trace 里的信息**——怎么从一堆 UI 元素里找到"哪一步花了 5 秒"、"这条调用花了多少 token"、"是哪个用户的请求"。

### 你会学到什么

- 能用 `mlflow.openai.autolog()` 一行开启追踪，所有 LLM 调用自动留痕
- 能用 `@mlflow.trace` 装饰自己的 Python 函数，把任何业务步骤也纳入追踪
- 理解 **Trace / Span / SpanType** 三个核心概念，看 UI 时不会迷路
- 能给 trace 打 user / session 元数据，做"按用户查历史""按会话聚合"
- 能用 `mlflow.search_traces()` 像查数据库一样程序化搜索历史 trace

### 对应脚本清单

这一章对应 `scripts/05_tracing/` 目录下的 4 个脚本：

| 脚本                        | 一句话作用                                                 | 是否必跑                 | 前置               |
| --------------------------- | ---------------------------------------------------------- | ------------------------ | ------------------ |
| `env_bootstrap.py`        | 自动把国内 LLM（DeepSeek 等）的 key 桥接成 OpenAI 协议     | 必跑（其他脚本都依赖它） | 无                 |
| `05a_env_test.py`         | 验证 MLflow 能联通 DeepSeek，发一次最简单的请求            | 必跑                     | 跑过 env_bootstrap |
| `05b_basic_tracing.py`    | `mlflow.openai.autolog()` 实战 + 多轮对话追踪            | 必跑                     | 跑过 05a           |
| `05c_custom_decorator.py` | `@mlflow.trace` 自定义 Span，搭一个 RAG 链看嵌套 Span 树 | 推荐                     | 跑过 05b           |
| `05d_metadata_search.py`  | 给 trace 打 user/session +`search_traces` 查询实战       | 推荐                     | 跑过 05b           |

## 核心概念

在写代码之前，我们先把 3 个最关键的概念搞清楚。它们就像做菜的"菜单 → 工序 → 单步"三级关系。

### Trace（一次完整的调用链）

Trace 就是"用户从发起请求到拿到结果"这一次完整的过程。在聊天应用里，一次问答或一次对话就是一个 Trace。Trace 有一个独一无二的 `trace_id`，UI 上每条 Trace 都对应一行。

> 类比：厨房日志里一整份**完整的点菜单**。菜还没做，但菜单已经登记——"客人 A 第 3 次来，点了酸菜鱼"。

### Span（一个具体操作）

一个 Trace 内部会被拆成若干个 Span。每个 Span 代表**一个具体做的事**：调一次 LLM、查一次向量库、跑一次重排函数。

Span 是嵌套的——比如一个 RAG 应用里最外层的 `rag_chain` 父 Span 调了 `retrieve_docs`、`rerank`、`generate_answer`，那后三者就是它的子 Span。**最外层那个 Span 叫 root span**。

> 类比：菜单里的一道大菜（酸菜鱼）会拆成若干**工序**——杀鱼、片鱼、炒料、熬汤、装盘。每道工序都是 Span，大菜本身是父 Span，里面每道工序是子 Span。

### SpanType（Span 的种类）

每个 Span 都有一个"类型"标签，告诉 MLflow 这个 Span 在干什么。常用值：

| SpanType      | 含义       | 用在哪儿                                  |
| ------------- | ---------- | ----------------------------------------- |
| `LLM`       | 大模型调用 | 包装调大模型的函数                        |
| `RETRIEVER` | 检索       | 包装向量查询、文档检索                    |
| `TOOL`      | 工具调用   | 包装外部 API / 工具函数                   |
| `CHAIN`     | 编排链     | 包装整个流程（最常用的"外层"装饰）        |
| `AGENT`     | Agent 决策 | 包装 Agent 主循环                         |
| 任意字符串    | 自定义     | 你自己随便起名，比如`"PROMPT_TEMPLATE"` |

```python
from mlflow.entities.span import SpanType

@mlflow.trace(span_type=SpanType.RETRIEVER, name="retrieve_docs")
def my_retriever(query):
    ...
```

UI 上不同类型会显示不同的小图标。**类型不影响功能**，主要是让你（和未来的你）一眼看出这个 Span 在干什么。

### 为什么这一章用 DeepSeek 而不是 OpenAI

直接连 OpenAI 在国内可能会卡。所以这一章我们用 `env_bootstrap.py` 把 DeepSeek 桥接成 OpenAI 协议——你用 OpenAI SDK 写代码，调的是 DeepSeek 的接口。这样你不用改代码，但能跑通。

> 类比：你点外卖用的是美团 App，但商家其实是同一家——平台帮你把"想吃的"翻译成"能下单的"。

## 实战步骤

### Step 0：准备 `.env` 文件

在项目根目录（`MLFlowLearning/`）下创建 `.env` 文件：

```bash
cd <project-root>
cp .env.example .env
```

然后用编辑器打开 `.env`，**至少填一行**：

```env
# 推荐：DeepSeek（国内直连，便宜）
DEEPSEEK_API_KEY=sk-xxxxxxxx
DEEPSEEK_MODEL=deepseek-chat

# 如果你有 OpenAI key 也可以直接用：
# OPENAI_API_KEY=sk-xxxxxxxx
# OPENAI_API_BASE=https://api.openai.com/v1
```

> 💡 **怎么拿 DeepSeek key？** 上 https://platform.deepseek.com/ → 注册 → 点 API Keys → 创建。DeepSeek 现在送几块钱额度，足够你跑完这章。
>
> 💡 **填完之后保存**（不是保存到 `.env.example`，而是 **File → Save As → 改名为 `.env`**——注意那个点不能漏）。

### Step 1：跑连通性测试（必跑）

这一步验证"你的 key 是对的、能联通 DeepSeek"。如果这步都跑不通，后面别浪费时间：

```bash
cd <project-root>/05_tracing
conda activate mlflow        # 或你装 mlflow 的环境
python 05a_env_test.py
```

应该看到这样的输出（中文回答是 DeepSeek 生成的，不影响理解）：

```
环境检查
  OPENAI_API_KEY:    已设置 (sk-xxx...)
  OPENAI_API_BASE:   https://api.deepseek.com/v1
  DEEPSEEK_MODEL:    deepseek-chat

调用 deepseek-chat ...
模型响应：
  我是 DeepSeek，一个由深度求索公司开发的 AI 助手...
  tokens: 78 (prompt=42, completion=36)
  model: deepseek-chat

✓ Phase 5 环境就绪！可以开始追踪了。
```

- 如果看到 `✓ Phase 5 环境就绪！` —— 进入下一步。
- 如果看到 `⚠️ 没找到任何 LLM API key` —— 回头检查 `.env`：填了 key 没？保存没？文件名叫 `.env`（不是 `.env.example`）？
- 如果连通了但报错（如 401、403、超时）——常见原因：key 复制错有空格、base URL 错、余额用完。

### Step 2：跑自动追踪 demo（必跑）

这步跑真正的追踪——一行代码开启 + 多轮对话：

```bash
python 05b_basic_tracing.py
```

跑完另开一个终端：

```bash
cd <project-root>
mlflow ui --port 5000
```

浏览器打开 `http://localhost:5000`，你会看到一个 **Experiments** 列表：

1. 选 `05_basic_tracing` experiment
2. 应该能看到至少 **2 个 Run**：一个单次调用的，一个多轮对话的 `multi-turn-chat`
3. 点开 `multi-turn-chat` 这个 Run
4. 切换到 **Traces** 标签页——你会看到 **3 条 trace**（每次 LLM 调用一条）
5. 任选一条 trace 点开——你应该看到一棵 **Span 树**，可能长得像这样：

```
chat_turn (root span, CHAT)
  └─ chat completions (OpenAI autolog 自动抓)
       Attributes: model=deepseek-chat, temperature=0.3, max_tokens=150
       Inputs: messages=[...]
       Outputs: content="..."
       Usage: prompt_tokens=42, completion_tokens=78
       Latency: 1.2 秒
```

**重点看**：

- **Attributes** 标签：点开看，model、temperature、max_tokens 这些参数是自动填的
- **Inputs / Outputs** 标签：能看到完整的 messages（user/system/assistant 三种角色）和生成的回复
- **Latency**：实际耗时（毫秒）
- **Usage**：token 用量（这就是为什么这章需要 key——你花的 token 是真实的钱）

> 💡 **如果没看到 Traces 标签**：切换到 Run 详情页，右侧栏有没有 `Traces` 字样？看不了的话说明 version 太老，`pip install -U mlflow` 试试。

### Step 3：自定义 Span 的 RAG demo（推荐）

这步展示更复杂的 Span 树——一个 RAG（检索增强生成）应用：

```bash
python 05c_custom_decorator.py
```

回 UI → experiment `05_custom_tracing` → Run `rag-q1` → Traces 标签 → 点开：

你应该看到一棵 **5 层嵌套**的 Span 树：

```
rag_chain (CHAIN)
  ├─ retrieve_docs (RETRIEVER)        [模拟向量库检索，耗时 50-150ms]
  ├─ rerank (TOOL)                    [重排，耗时 20-50ms]
  ├─ build_prompt                     [组装 prompt，瞬时]
  └─ generate_answer (LLM)
       └─ chat completions            [autolog 自动抓的 OpenAI 调用]
```

**重点看**：

- 每一层的颜色/图标可能不一样（按 SpanType 区分）
- 每一层都有自己的 Latency——你可以一眼看出"原来 retrieve 花了 100ms，LLM 花了 1500ms"
- 每一层都有 Inputs / Outputs——LLM 的输入是组装好的 prompt，能看到 prompt engineering 的真实样子

### Step 4：元数据 + 搜索 demo（推荐）

这步演示"多用户多会话"场景下的追踪和查询：

```bash
python 05d_metadata_search.py
```

这个脚本会模拟 3 个用户（alice / bob / charlie）每个 5 轮对话（共 15 次 LLM 调用），然后用 `mlflow.search_traces()` 做 4 种查询：

| 查询                      | 你应该看到                             |
| ------------------------- | -------------------------------------- |
| 所有 OK 的 trace          | 15 条左右                              |
| 特定用户（alice）的 trace | 数量大约是总数的 1/3                   |
| 按 session_id 聚合        | 该 session 内所有 trace 数（可能是 5） |
| 按 latency 倒序           | 前 5 个最慢的调用及其所属用户          |

**重点看**：

- 怎么用 `mlflow.update_current_trace(user=..., session_id=...)` 给 trace 打元数据
- 为什么 `metadata.`\`mlflow.trace.user\`` 这个奇怪的字段名要用反引号包裹（因为字段名里有点号，会被 SQL 解析器误读）
- `order_by=["execution_time_ms DESC"]` 是 list[str]，每个元素是"字段名 + 方向"的拼接字符串

### Step 5（选做）：看一眼底层数据库

```bash
cd <project-root>
sqlite3 mlflow.db "SELECT trace_id, status, execution_time_ms FROM traces LIMIT 10;"
```

可以看到 trace 真的存在 SQLite 里（数据库方式不太直观，不强求）。

## 🛠️ 动手做

这一节是**这一章的实操作业**。你需要做的是：

### 任务 1：填好 `.env` 文件

打开 `MLFlowLearning/.env`：

- **选 DeepSeek**（推荐）：去 https://platform.deepseek.com/ 注册 + 拿 key，填 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_MODEL=deepseek-chat`
- 或者填 `OPENAI_API_KEY` 用 OpenAI（要能直连海外）

### 任务 2：跑 `05a_env_test.py` 验证连通性

```bash
cd 05_tracing
python 05a_env_test.py
```

看到 `✓ Phase 5 环境就绪！` 才算完成这一步。

### 任务 3：跑 `05b_basic_tracing.py` 并在 UI 里看 trace

```bash
python 05b_basic_tracing.py
# 另开终端
mlflow ui --port 5000
```

验证：

1. experiment `05_basic_tracing` 出现，且至少有 2 个 Run
2. 点开 `multi-turn-chat` Run → Traces 标签 → 看到 3 条 trace
3. 任选一条 trace → 看到 Span 树、Attributes、Usage、Latency

### 任务 4（可选但推荐）：把 05c 和 05d 也跑了

跑完之后在 UI 里逛逛：

- 找到 `05_custom_tracing` → `rag-q1` → 看 5 层 Span 树
- 找到 `05_metadata` → `multi-user-sim` → 试用过滤 `metadata.\`mlflow.trace.user\` = 'alice'`

## 避坑清单

- ⚠️ **`update_current_trace` 用错地方** → 必须在 `@mlflow.trace` 装饰的函数内、或 `with mlflow.start_span(...)` 代码块内调用。在普通函数里调用会报 "no active trace"。
- ⚠️ **想给 trace 加 user 却写了 `metadata={"mlflow.trace.user": ...}`** → ❌ 这是保留字段，必须用专用关键字 `mlflow.update_current_trace(user=user_id, session_id=sid)`。
- ⚠️ **`search_traces` 的 `experiment_ids` 参数** → 在 MLflow 2.x 还能用，3.0 起会废弃，建议尽快迁移到 `locations=[exp_id]`。
- ⚠️ **`search_traces` 的 `order_by`** → 是 `list[str]`，不是 `list[dict]`；字段名是 `execution_time_ms`（**不是** `execution_duration`，虽然返回的 DataFrame 列名叫 `execution_duration`，那是另一回事）。
- ⚠️ **没看到 trace** → 检查是不是开了 autolog 但没在 UI 里选对 experiment；还有些包需要先 `pip install`，比如 `mlflow[genai]`。
- ⚠️ **国内连 OpenAI 直连超时** → 用 `env_bootstrap.py` 桥接到 DeepSeek/智谱/百炼，OpenAI SDK 通过 `OPENAI_API_BASE` 走兼容协议。
- ⚠️ **`.env` 找不到** → 文件名带点的 `.env` 是隐藏文件。`ls -la` 才看得到。Windows 下用编辑器"另存为"选"All files (*.*)"才能存为带点的文件名。
- ⚠️ **trace 找不到但 token 也用了** → `mlflow.set_experiment()` 可能写到了另一个 URI（比如 sqlite 但忘了路径），检查 `mlflow.get_tracking_uri()` 输出对不对。

## 📖 下一步

这一章你学会的是"怎么把 LLM 调用记下来、看 trace、查 trace"。但 LLM 应用真正的痛点是**怎么评估输出质量**——一段对话生成了，是好还是坏？谁来评？——这就要靠 Chapter 7 之后的评估章节。

但 Chapter 7 不是评估。Chapter 7 是这一份学习作业的**彩蛋**：你可能注意到，整个项目里有个目录叫 `mlflow_skills/`，里面装了一堆 `SKILL.md` 文件——那是给 AI 编程助手（Claude Code、Cursor 等）读的。我们下一章就来学这些 skill 怎么用。

更深入的学习可以看 `notes/05_tracing.md`——本文就是这份笔记的"轻量化版"，那份笔记里有更多边界情况讨论、UI 截图、每个 Span 字段的含义解释。

---

# Chapter 7：Vibecoding 集成——用 AI 编程助手操作 MLflow

> ⏱️ 预计时间：30 分钟
> 🔑 是否需 API Key：否（本章不直接调 LLM）
> 📚 前置知识：Chapter 0-6（了解 MLflow 基本概念 + 跑过至少一个脚本）

## 🎯 这章做什么

你可能已经注意到，项目里有个目录叫 `mlflow_skills/`，里面装了一堆 `SKILL.md` 文件——**那是给 AI 编程助手（Claude Code、Cursor、Copilot 等）读的"指令手册"**。这一章教你：

1. `mlflow_skills` 是什么、里面有什么
2. 在 vibecoding（对话式编程）场景下，怎么让 AI 助手帮你操作 MLflow
3. 遇到不懂的 MLflow 操作时，怎么让 AI 助手去查 skill 再帮你做

> 💡 **核心洞察**：你不必精通 MLflow 的每一个 API。只要 AI 助手能读到 `mlflow_skills/` 里的 SKILL.md，它就会按手册帮你做对——而你要做的只是**学会怎么让它用这些手册**。

### 你会学到什么

- 知道 `mlflow_skills/` 里 12 个 skill 各自管什么
- 知道 AI 助手是怎么用这些 skill 的（读 SKILL.md → 按步骤执行）
- 能自己用一句话让 AI 助手加追踪 / 评估 / 对比 / 部署
- 能验证 AI 助手干的对不对（去 UI 看结果）
- 学会"通用 skill 使用话术"（不依赖任何具体 AI 工具）

### 前置知识

- 已完成 Chapter 0-6
- 有一个 AI 编程助手（Claude Code、Cursor、GitHub Copilot 等，任选）
- 一个已跑通的最小实验（比如 Chapter 2 的 `01b_sklearn_basics.py`）

---

## 一、什么是 mlflow_skills？

`mlflow_skills/` 是一组**给 AI 助手看的 Markdown 指令手册**。每个 skill 是一个目录，里面有一个 `SKILL.md`（手册正文）+ `references/`（深度参考）+ `scripts/`（可执行工具脚本）。

```
mlflow_skills/
├── mlflow-onboarding/          ← 引导上手
├── classical-ml/               ← 传统 ML 6 步法
├── instrumenting-with-mlflow-tracing/  ← 给代码加追踪
├── agent-evaluation/           ← 评估 LLM agent
├── querying-mlflow-metrics/    ← 查指标
├── retrieving-mlflow-traces/   ← 搜 trace
├── analyze-mlflow-trace/       ← debug 单条 trace
├── analyze-mlflow-chat-session/ ← debug 多轮对话
├── fix-agent-issue/            ← 修 agent 行为
├── mlflow-agent/               ← 通用分发器
├── searching-mlflow-docs/      ← 查官方文档
└── sagemaker-mlflow/           ← 连 SageMaker
```

每个 `SKILL.md` 的结构大致是：

```markdown
---
name: classical-ml
description: |
  触发条件列表（"训练 sklearn/xgboost"、"对比 runs" 等）
  + 关键行为约束（建议但不强制执行）
---
# 正文
## When to consult   ← AI 助手读这里判断"要不要用我"
## Step 1: Tracking
## Step 2: Registry
...（按步骤教 AI 助手怎么做）
```

**AI 助手的工作流**：

1. 你发一句话（比如"帮我给这个 sklearn 训练加追踪"）
2. AI 助手判断这个话题匹配 `classical-ml` 的 description
3. 它打开 `mlflow_skills/classical-ml/SKILL.md` 读步骤
4. 按步骤帮你改代码 / 跑命令
5. 你验证结果（看 UI）

> 📚 想手动浏览所有 skill？看本文档 [Skill 段](#skill-段) 的 12 个 skill 总表。

---

## 二、vibecoding 场景：怎么让 AI 助手帮你操作 MLflow

下面用几个最常见的场景，展示"你说什么 → AI 助手做什么 → 你怎么验证"。**这些话术不依赖任何具体 AI 工具**，你在 Claude Code / Cursor / Copilot 里都能用。

### 场景 1：给 sklearn 训练加追踪（最常用）

**你说**：

> 帮我给这个 `train.py` 加 MLflow 追踪，自动记录参数、指标和模型。

**AI 助手会做**：

1. 读 `train.py`，识别框架（`from sklearn.ensemble import RandomForestClassifier` → 用 `mlflow.sklearn.autolog()`）
2. 检查 tracking URI（如果没设，提醒你 `export MLFLOW_TRACKING_URI="sqlite:///$(pwd)/mlflow.db"`）
3. 加一行 `mlflow.sklearn.autolog()` + 包一个 `with mlflow.start_run(...)`
4. 跑脚本，确认出现 `Run with id: ...`

**你验证**：

```bash
$ mlflow ui --port 5000
# 打开 http://localhost:5000
# 看到新 Run，Params/Metrics/Artifacts 自动填好
```

### 场景 2：让 AI 助手对比两个模型

**你说**：

> 我跑了两个实验（lr=0.01 和 lr=0.001），帮我对比哪个好，用 UI 结果说话。

**AI 助手会做**：

1. 用 `mlflow.search_runs()` 或 `mlflow runs search` 拉出两个 Run 的指标
2. 按 accuracy / f1 排序，告诉你哪个好
3. 建议你把赢家注册成 champion

**你验证**：

```bash
# 看 AI 助手给你的对比表，或去 UI Compare 页面自己看
```

### 场景 3：给 LLM 调用加追踪

**你说**：

> 这段 OpenAI 调用怎么加 MLflow 追踪？我想看每次调用了多少 token。

**AI 助手会做**：

1. 识别 `from openai import OpenAI` → 用 `mlflow.openai.autolog()`
2. 在 `import openai` 之后加一行 `mlflow.openai.autolog()`
3. 确认 tracking server 在跑（或 SQLite）
4. 跑脚本，确认 trace 落库

**你验证**：

```bash
# UI 顶部 Traces tab → 看到 ChatCompletion 的 trace
# 点开看 Latency、Total tokens
```

### 场景 4：评估一个 LLM agent

**你说**：

> 我这个客服 agent 答得不准，帮我评估一下，看有多少比例合格。

**AI 助手会做**：

1. 用 `agent-evaluation` skill：先确认有 tracing
2. 建 eval 数据集（`mlflow.genai.datasets.create_dataset`）
3. 配置 scorers（Correctness / Safety / 自定义）
4. 跑 `mlflow.genai.evaluate()`，出分数

**你验证**：

```bash
# UI 里看评估 Run 的 Metrics（correctness/mean 等）
```

---

## 三、AI 助手"不知道"怎么用 MLflow 时怎么办

有时候 AI 助手会瞎写或报错。**让它去读 skill**：

**你说**：

> 这个报错了，帮我查 `mlflow_skills/` 里对应的 skill，看正确写法是什么。

或者更具体：

> 我用 `mlflow.sklearn.log_model(model, artifact_path="m")` 报错了。查一下 `mlflow_skills/classical-ml/SKILL.md`，MLflow 3 应该怎么写？

**为什么有效**：skill 里的 SKILL.md 明确写了 MLflow 3 vs 2 的破坏性变化（`artifact_path=` → `name=` 等）。AI 助手读到后就不会再用旧写法。

**通用话术**（任何 AI 工具都能用）：

- "查 `mlflow_skills/` 里有没有相关的 skill"
- "按 `mlflow_skills/classical-ml/SKILL.md` 的步骤做"
- "这个 MLflow API 报错了，帮我看看 skill 里 MLflow 3 的写法"

---

## 四、怎么启用 skill（通用，不依赖具体工具）

不同 AI 助手启用 skill 的方式不同（Claude Code、Cursor、Copilot 等各自有 rules / skills 配置机制，具体去各自文档查）。**核心原则只有一个：让助手能看到 SKILL.md**。

**最简单的方式（所有工具通用，零配置）**：不装任何东西，直接在对话里让 AI 助手读文件：

> 先读 `<project-root>/mlflow_skills/classical-ml/SKILL.md`，然后按里面 Step 1 帮我做。

这样 AI 助手每轮对话都会参考那个手册。想在更长远的会话里也自动生效，就按你所用 AI 工具的 rules / skills 配置机制，把这个目录加进去。

> ⚠️ 别纠结"链入"这个动作本身——它只是让 AI 助手"知道有这个手册"。直接说"读 SKILL.md 再做事"效果一样。

---

## 五、关键 Take-aways

- **`mlflow_skills/` 是给 AI 助手的"指令手册"**，不是给你读的教程（但你读也有帮助）
- **不装 skill 也能用**：一句话让 AI 助手"读 SKILL.md 再做事"就行
- **AI 助手按 skill 做的是"建议"，不会擅自改代码**——都要你确认
- **验证永远是去 UI 看**：Experiments / Models / Traces / Prompts 四个 tab
- **遇到报错先让 AI 查 skill**——skill 里写了 MLflow 3 的正确写法，避免踩 2→3 的坑

## 📖 下一步

→ [Chapter 8：Prompt Registry——像 Git 一样管理提示词](#chapter-8)（或跳到 [Skill 段](#skill-段) 看 12 个 skill 总览）

# Chapter 8：提示词注册表（Prompt Registry）—— 像 Git 一样管理你的提示词

> ⏱️ 预计时间：40 分钟
> 🔑 是否需 API Key：是（要真实调一次 LLM）
> 📚 前置知识：第 7 章（Tracing 与 Autolog）

## 🎯 这章做什么

前 7 章我们都在跟踪「模型」和「实验」，但 LLM 应用最该被管起来的其实是**提示词（Prompt）**——一行提示词的改动就可能让生产环境崩溃。今天你将学会用 **Prompt Registry** 给提示词建 Git 仓库：每次改提示词都登记一个新版本，给不同版本贴 `@production` / `@staging` 别名，**应用永远只通过别名加载**。

类比：

- 写代码用 Git + main/dev 分支管理 → 写提示词就该用 Prompt Registry + `@production` / `@staging` 别名
- Prompt Registry 就是「提示词界的 GitHub」

跑完这章，你会在 MLflow UI 里看到：

- 一个叫 `customer-support-qa` 的 prompt，有 3 个版本（v1 简洁版 / v2 详细版 / v3 chat 版）
- 别名 `production` 指向 v2、`staging` 指向 v3
- 一次真实 LLM 推理的 Trace，证明 `@production` 真的能用

### 你会学到什么

- 明白「提示词也是代码」，需要版本化、别管理、零停机切换
- 用 `mlflow.genai.register_prompt()` 注册带 Jinja2 的版本化 prompt
- 用 `set_prompt_alias(name, alias, version)` 管理 `@production` / `@staging`
- 在代码里通过 `prompts:/<name>@<alias>` 加载 prompt
- 用 `PromptModelConfig` 把 temperature/max_tokens 跟 prompt 绑定
- 完成「注册两个版本 + 分别打 `@production` 和 `@staging` 别名」的动手做

---

## 核心概念

### 1. 提示词也是代码：为什么需要注册表

很多人改提示词的做法是：直接改代码里的字符串、提交、部署。这条路一旦走远，会撞上三个坑：

- **没有版本历史**：上周哪版提示词效果好？想回去？没了。
- **没有 staging / production 隔离**：在测试版测完直接上生产？出问题想秒回滚？只能紧急改代码重新部署。
- **改提示词忘改参数**：把 prompt 改了，但 `temperature=0.9` 没改成 `0.3`，回答风格突然变了。

Prompt Registry 一次性解决这三个问题：每次注册都建一个新版本（v1 → v2 → v3），不可改历史；用 `@production` / `@staging` 别名做零停机切换；用 `PromptModelConfig` 把模型参数一起绑定。

### 2. `mlflow.genai.register_prompt()` —— 一行注册新版

最核心的函数，签名简化版：

```python
v2 = mlflow.genai.register_prompt(
    name="customer-support-qa",
    template="你是 {{ company }} 的客服助手。问题：{{ question }}",
    commit_message="v2: 加 tier 分支 + 字数约束",
    tags={"style": "detailed", "author": "bob"},
)
```

要点：

- 同名注册不会覆盖，而是自动创建新版本（v1 → v2 → v3 …）
- 模板一旦注册就不可改（要改就注册新版）
- `commit_message` 写这次改了什么、`tags` 用来打标签（author / style / format 等）

`template` 支持两种格式：

**格式 A：纯文本（带 Jinja2）**

```python
template="你是 {{ company }} 的客服。{% if tier == 'premium' %}VIP 优先{% endif %}"
```

**格式 B：Chat 消息列表（OpenAI Chat Completions 格式）**

```python
template=[
    {"role": "system", "content": "你是 {{ company }} 的客服。"},
    {"role": "user", "content": "{{ question }}"},
]
```

Jinja2 用法回顾（不用也不影响跑通）：

- 变量：`{{ question }}`
- 条件：`{% if tier == 'premium' %}...{% endif %}`
- 循环：`{% for item in items %}...{% endfor %}`

调用 `prompt.format(var1=..., var2=...)` 把变量填进去。

### 3. `set_prompt_alias(name, alias, version)` —— 别名是「指针」

别名的本质是「指向某个版本的指针」。

```python
mlflow.genai.set_prompt_alias("customer-support-qa", "production", version=2)
mlflow.genai.set_prompt_alias("customer-support-qa", "staging", version=3)
```

常见别名约定：

- `@production`：线上跑的稳定版
- `@staging`：测试版，准备晋升 production
- `@champion` / `@challenger`：A/B 测试用

**零停机发布的魔力**：所有通过 `prompts:/name@production` 加载 prompt 的应用，下一次启动就自动用新版。要回滚？把 `@production` 再指回老版本就行。

### 4. `prompts:/<name>@<alias>` —— 应用永远只通过别名加载

代码里加载 prompt 的两种写法：

```python
# 写法 A：按版本号加载（写死，不灵活，不推荐）
prompt = mlflow.genai.load_prompt("prompts:/customer-support-qa/2")

# 写法 B：按别名加载（推荐，永远跟最新发布走）
prompt = mlflow.genai.load_prompt("prompts:/customer-support-qa@production")
```

记住一句话：**永远别在代码里写死版本号**。发布、回滚、A/B 测试全是改别名一行代码的事。

### 5. `PromptModelConfig` —— 把模型参数跟 prompt 绑定

```python
from mlflow.entities.model_registry.prompt_version import PromptModelConfig

config = PromptModelConfig(
    model_name="deepseek-v4-flash",
    temperature=0.3,
    max_tokens=300,
)
mlflow.genai.set_prompt_model_config("customer-support-qa", version=2, model_config=config)
```

绑定后通过 `prompt.model_config` 能一次性拿到 `{model_name, temperature, max_tokens}`，避免「prompt 改了但 temperature 没改」的坑。

---

## 实战步骤

### Step 0：环境准备

```bash
conda activate mlflow
# 确保 .env 里有 OPENAI_API_KEY / OPENAI_API_BASE / DEEPSEEK_MODEL
# 启动 MLflow（用 sqlite 后端）
mlflow ui --port 5000   # 另开一个终端
```

### Step 1：注册三个版本（`06a_register_prompt.py`）

```bash
cd <project-root>
python scripts/06_prompts/06a_register_prompt.py
```

脚本会做三件事：

1. 注册 v1：简洁文本版（`"你是 {{ company }} 的客服助手..."`）
2. 注册 v2：详细版（带 Jinja2 条件 + 字数约束 + 引用格式）
3. 注册 v3：chat 消息格式（system + user）

预期输出：

```
注册 v1（简洁版）...
  ✓ customer-support-qa v1
    variables: ['company', 'question']
注册 v2（详细版，带约束）...
  ✓ v2
    variables: ['company', 'agent_name', 'max_words', 'question']
注册 v3（多消息格式）...
  ✓ v3
    is_text_prompt: False
```

### Step 2：设 alias + 真实推理（`06b_alias_lifecycle.py`）

```bash
python scripts/06_prompts/06b_alias_lifecycle.py
```

脚本会做：

1. 把 `production` 别名指向 v2、`staging` 别名指向最新版本
2. 用 `set_prompt_model_config` 给 v2 绑定 `PromptModelConfig`
3. 通过 `prompts:/customer-support-qa@production` 加载 prompt，format 后调 LLM
4. 打印 LLM 真实回复

预期输出（节选）：

```
找到 3 个版本，最新 v3
  ✓ production → v2
  ✓ staging → v3
  ✓ v2 model_config: PromptModelConfig(model_name='deepseek-v4-flash', temperature=0.3, max_tokens=300)

📦 通过 prompts:/name@alias 加载：
  production prompt:
    name: customer-support-qa
    version: 2
    template (前 200 字符):
    你是 {{ company }} 的客服助手 {{ agent_name }}...

🤖 用 production 提示词真实推理：
  渲染的 messages: [...]
  A: ...（LLM 真实回答）
  tokens: ...
```

### Step 3：UI 验证

打开 `http://localhost:5000`：

1. **左侧栏 → Prompts**：点开 `customer-support-qa`
   - 看 Versions 标签下的 v1 / v2 / v3
   - 看每个版本的 `commit_message` 和 `tags`
   - 看 Aliases：`production` → v2，`staging` → v3
2. **左侧栏 → Experiments → `06_prompt_registry`**：看三个 register 操作的 Run 记录
3. **左侧栏 → Experiments → `06_prompt_alias` → Run `prod-prompt-inference`**：
   - 点 Traces 标签：能看到真实 LLM 调用的完整链路（prompt 内容、模型参数、回复）

---

## 🛠️ 动手做：注册两个版本 + 打别名

任务：在 `06a_register_prompt.py` 的基础上，自己写一段脚本，注册两个版本的 prompt，并给它们打 `@production` 和 `@staging` 别名。

**步骤**：

1. 写一个新脚本 `/tmp/my_prompt_registry.py`：

```python
import mlflow
import os

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("08_my_prompt_demo")

PROMPT_NAME = "my-product-qa"

# 1) 注册简洁版（v1）
v1 = mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template=(
        "你是 {{ company }} 的产品助手。请简洁回答：\n\n"
        "用户问题：{{ question }}"
    ),
    commit_message="v1: 简洁版",
    tags={"style": "concise"},
)
print(f"  ✓ v1: {v1.uri}")

# 2) 注册详细版（v2），加字数约束和 Jinja2 条件
v2 = mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template=(
        "你是 {{ company }} 的产品助手 {{ agent_name }}。\n"
        "{% if tier == 'premium' %}"
        "  VIP 客户，请提供个性化建议。\n"
        "{% endif %}"
        "约束：\n"
        "  - 不超过 {{ max_words }} 字\n"
        "  - 末尾加 [source: KB-id]\n\n"
        "用户问题：{{ question }}"
    ),
    commit_message="v2: 详细版 + tier 分支",
    tags={"style": "detailed"},
)
print(f"  ✓ v2: {v2.uri}")

# 3) 打别名：v1 给 staging，v2 给 production
mlflow.genai.set_prompt_alias(PROMPT_NAME, alias="staging", version=1)
mlflow.genai.set_prompt_alias(PROMPT_NAME, alias="production", version=2)
print("  ✓ staging → v1")
print("  ✓ production → v2")

# 4) 加载 production 提示词并渲染
prod = mlflow.genai.load_prompt("prompts:/my-product-qa@production")
print(f"\nproduction template:\n{prod.template}")
print(f"variables: {prod.variables}")

rendered = prod.format(
    company="Anthropic",
    agent_name="Claude",
    tier="premium",
    max_words=80,
    question="你们的 MLflow 集成怎么收费？",
)
print(f"\n渲染后:\n{rendered[:300]}")
```

2. 跑这个脚本：

```bash
python /tmp/my_prompt_registry.py
```

3. 去 UI 看 `my-product-qa` 这个 prompt，确认：
   - Versions 标签下有 v1、v2
   - Aliases 标签下：`production` → v2，`staging` → v1
   - v2 的 template 里能看到 Jinja2 条件分支

---

## 避坑清单

- ⚠️ **`register_prompt` 同名不会覆盖** → 设计如此（要的就是版本化），别以为是 bug。要「重置」得手动删旧版本或在脚本里加判断。
- ⚠️ **`prompt.variables` 不包含 `{% if %}` / `{% for %}` 里的变量** → 它只识别 `{{ var }}` 形式的变量。如果你的模板里有条件分支，分支里的变量调用 `format()` 时也得传进去（否则会报错）。
- ⚠️ **跑 `06b` 前必须先跑 `06a`** → 否则找不到 `customer-support-qa` 这个 prompt，脚本会 `RuntimeError`。
- ⚠️ **chat 格式和文本格式不能混用 `format()`** → 文本格式 `format()` 返回字符串；chat 格式 `format()` 返回消息列表。喂给 `OpenAI().chat.completions.create(messages=...)` 时一定要是后者。
- ⚠️ **`mlflow.genai.search_prompts()` 不返回版本号** → 它返回 `Prompt` 对象（只有 name）。要查具体版本用 `client.search_prompt_versions(name=...)`，里面有 `version`、`commit_message`、`tags`。
- ⚠️ **改 prompt 时记得同步改 `PromptModelConfig`** → `set_prompt_model_config` 是单独调用的，不会自动跟着 prompt 走。如果你换了模型名，记得重新 bind。

---

## 📖 下一步

你已经学会了用 Prompt Registry 管提示词版本。但光管起来还不够——你怎么知道 v2 真的比 v1 好？下章进入 **Chapter 9：GenAI 评估与 LLM-as-judge**，你将学到如何用 `mlflow.genai.evaluate()` 给 LLM 应用「打分」，并用 `@scorer` 和 `make_judge` 写出业务专属的评分器，最后把 production vs staging 两个 prompt 做一次正经的 A/B 对比。

详细学习笔记见：`notes/06_prompts.md`。

---

# Chapter 9：GenAI 评估与 LLM-as-judge —— 给 LLM 应用「打分」

> ⏱️ 预计时间：40 分钟
> 🔑 是否需 API Key：是（DeepSeek，scorer 与 judge 都要调 LLM）
> 📚 前置知识：Chapter 8（Prompt Registry）、Chapter 4（Tracing）

## 🎯 这章做什么

写完一个 LLM 应用，怎么知道它「答得好不好」？LLM 回答是自然语言，既要看「对不对」，也要看「语气是否友好」「有没有胡编」——传统 accuracy/RMSE 派不上用场。这一章教你用 **MLflow GenAI 评估**给 LLM 应用「打分」。

类比：像教实习生答客户题——**内置 Scorer** 是公司统一评分卡（正确性/安全性/切题度），**`@scorer`** 是你写的特殊规矩（必须加引用），**`make_judge()`** 是雇个资深主管用自然语言当裁判，**跨版本对比** 是同一份题库考新旧两版 prompt 看谁分高。跑完产出：`07_evaluate` / `07_custom_scorer` 两个 experiment 下的评估 Run，含聚合分数和逐行打分。

### 你会学到什么
- 用 `mlflow.genai.evaluate()` 跑一次完整评估（题库 + predict_fn + scorers）
- 配置内置 Scorer（Correctness / Safety / RelevanceToQuery）并解决 judge 模型问题
- 用 `@scorer` 装饰器写业务规则，用 `make_judge()` 写 LLM-as-judge 主观评分
- 显式传 judge_model，绕开国内服务商默认模型坑
- 跨 prompt 版本做 A/B 评估，按结果用 `set_prompt_alias` 切流

### 前置
- 已完成 Chapter 8（注册过 `customer-support-qa` prompt）、Chapter 4（Tracing）
- 需 API Key：是，`.env` 配好 `OPENAI_API_KEY` / `OPENAI_API_BASE` / `DEEPSEEK_MODEL`
- 环境：`conda activate mlflow`；另开终端 `mlflow ui --port 5000`

### 必跑脚本清单
| 脚本 | 一句话作用 | 前置 |
|------|-----------|------|
| `scripts/07_evaluation/07a_basic_evaluate.py` | 5 行题库 + 3 个内置 scorer 跑基础评估 | Chapter 4-5 |
| `scripts/07_evaluation/07b_custom_scorer.py` | 组合 `@scorer` + `make_judge` 共 6 个 scorer | 跑过 07a |

## 核心概念（一句话版）
- **`mlflow.genai.evaluate()`**：GenAI 的「考试系统」——给题库（`data`）+ 考生（`predict_fn`）+ 评分卡（`scorers`），自动调 LLM 拿答案再打分：
  ```python
  result = mlflow.genai.evaluate(data=EVAL_DATA, predict_fn=predict_fn, scorers=[...])
  # result.metrics → 聚合分；result.tables["eval_results"] → 逐行分
  ```
- **内置 Scorer**：开箱即用的通用评分卡——`Correctness()` 对比期望答案（需 `expected_response`/`expected_facts`）、`Safety()` 检测不安全内容、`RelevanceToQuery()` 判是否切题。
- **`@scorer` 自定义**：一行装饰器写硬性业务规则，参数名必须叫 `inputs/outputs/expectations`：
  ```python
  @scorer(name="has_citation")
  def has_citation(outputs: str) -> bool:
      return "[source:" in (outputs or "")
  ```
- **`make_judge()`**：LLM 当裁判，用自然语言 rubric 打分「语气、合理性」这类写不出代码的项；`instructions` 必须含 `{{ inputs }}`/`{{ outputs }}`/`{{ trace }}` 之一，否则报错。
- **judge_model 必须显式传（国内服务商坑）**：内置 scorer 和 make_judge 默认用 `gpt-4.1-mini`（OpenAI 直连），DeepSeek 等服务商不认识会报 404；一律传 `model=f"openai:/{os.getenv('DEEPSEEK_MODEL','deepseek-v4-flash')}"`。
- **`expected_response` vs `expected_facts` 二选一**：前者给完整参考答案、后者给关键事实列表，给 `Correctness()` 当依据；同时给会触发额外 judge 调用（默认模型）又挂掉。
- **跨版本对比**：同一份题库 + 同一组 scorer，只换 `predict_fn` 里的 prompt 版本（`@production` vs `@staging`）；UI 勾选两个 Run → Compare 看分差，胜者 `set_prompt_alias` 切 production，败者留档回滚。

## 🛠️ 动手做

**练习 1：跑基础评估（07a）**
```bash
cd <project-root> && python scripts/07_evaluation/07a_basic_evaluate.py
```
✅ 验证标准：终端打印 `judge_model = openai:/deepseek-v4-flash` 以及 `correctness/mean`、`safety/mean`、`relevance_to_query/mean` 等聚合分；UI 的 `07_evaluate` experiment 下出现新 Run，逐行打分在 Evaluation 标签或 `Artifacts/eval/`。

**练习 2：跑自定义 scorer（07b）**
```bash
python scripts/07_evaluation/07b_custom_scorer.py
```
✅ 验证标准：聚合指标里多出 `has_citation/mean`、`is_concise/mean`、`mentions_mlflow/mean`、`brand_tone/mean`——对比 07a 只有内置 3 个，体会「内置 / @scorer / make_judge」三类 scorer 的分工（07b 的 `Correctness()`/`Safety()` 未传 expectations 跑得宽松，真正调 judge 的是 `brand_tone`）。

## 📖 深入阅读（关键！）
> 📚 [`notes/07_evaluation.md`](07_evaluation.md)（Phase 7：GenAI 评估与自定义 Scorer）——重点：judge_model 必须显式传的坑、`expected_response` vs `expected_facts` 不能同时给、完整内置 Scorer 清单、07c 跨版本对比模板。

## 📖 下一步
→ [Chapter 10：模型评估与本地部署服务](#chapter-10-模型评估与本地部署服务)
# Chapter 10：模型评估与本地部署服务 —— 给模型打分 + 起成 REST API

> ⏱️ 预计时间：45 分钟
> 🔑 是否需 API Key：否
> 📚 前置知识：Chapter 6（注册模型 + alias）、Chapter 7（trace）

## 🎯 这章做什么

模型训完、注册到 Registry 后，要回答两个问题：**这个模型到底好不好？**（哪一类 0.97，不是一句敷衍）和 **怎么让别人用它？**（发个 HTTP 请求拿预测，不是发 pickle 文件）。这一章解决这两件事：先用 `mlflow.models.evaluate()` 一行算全套指标 + 自动出图，再用 `mlflow models serve` 把模型起成 REST API，最后用 `curl` 调它。

类比：训练模型像做菜——**试吃**（`evaluate`：食客打分 accuracy/F1、看摆盘混淆矩阵、看味道曲线 ROC）和 **上桌**（`models serve` 把菜放进窗口让客人点餐）。产出：`04_evaluate` / `04_evaluate_custom` 两个 experiment 的评估 Run，以及一个能 curl 的本地推理服务。

### 你会学到什么
- 用 `mlflow.models.evaluate()` 一行算全部分类/回归指标 + 自动出混淆矩阵、ROC、PR 图
- 用 `make_metric()` 写自定义业务指标（如「高价值客户加权 accuracy」）
- 用 `validate_evaluation_results()` 给新模型设「上线门槛」（MLflow 3 新 API）
- 用 `mlflow models serve -m models:/Name@champion -p 5001` 起 REST API，`curl` 调 `/invocations` 推理

### 前置
- 已完成 Chapter 6：会 `mlflow.start_run()` / `mlflow.sklearn.log_model()`，注册过 `WineQualityClassifier` 且有 `@champion` 别名
- 需 API Key：否；无需任何外部服务
- 04a/04b 用本地文件模式即可；04c **必须** server 模式（`sqlite:///mlflow.db` + `./mlruns`）

### 必跑脚本清单
| 脚本 | 一句话作用 | 前置 |
|------|-----------|------|
| `scripts/04_evaluate/04a_evaluate_basics.py` | 对 RandomForest 跑 `mlflow.models.evaluate`，自动算内置指标 + 生成混淆矩阵/ROC 图 | Chapter 6 |
| `scripts/04_evaluate/04b_evaluate_custom.py` | 写「高价值客户加权」自定义指标，对比 RF vs LR，验证 B 是否达标 | 跑过 04a |
| `scripts/04_evaluate/04c_models_serve.sh` | `mlflow models serve` 起本地 REST API，`curl` 调 `/invocations` 推理 | 跑过 04b + 有 champion 模型 |

## 核心概念（一句话版）
- **`mlflow.models.evaluate()`**：全自动评估 + 自动作图 + 自动写回 MLflow 的整合入口——指模型（`runs:/xxx/model` 或 `models:/xxx@champion`）+ 含 label 的数据 + `model_type="classifier"`，吐全套内置指标和图（写入 `Artifacts/eval/`）：
  ```python
  result = mlflow.models.evaluate(model=model_uri, data=eval_df, targets="target", model_type="classifier")
  # result.metrics → 指标 dict；result.artifacts → ['confusion_matrix.png', 'roc_curve_plot.png', ...]
  ```
- **`make_metric()` 自定义指标**：把业务特殊规则（class_0 错代价 5 倍 → 加权 accuracy）包装成 MLflow 指标，和内置指标一起出现在 UI 的 Metrics 标签；`eval_fn` 必须接收 `predictions`/`targets` 两个 Series 并返回标量（纯函数）。
- **`validate_evaluation_results()`（MLflow 3）**：先对 A、B 各 `evaluate()` 一次，再用 `MetricThreshold(threshold=..., greater_is_better=...)` 集中验证「B 是否比 A 好、好多少」——阈值可独立复用、可塞 CI；替代 MLflow 2 的 `baseline_model=` 一锅炖。
- **`mlflow models serve`**：把模型起成标准 REST API，统一路径 `/invocations`，收 JSON（`dataframe_records`，推荐）或 `text/csv`——别人用 `curl` / `requests.post` 就能调；首次启动会自动装模型依赖。
- **`extra_metrics=` 而非 `custom_metrics=`（坑）**：MLflow 3 把自定义指标参数改名成 `extra_metrics`，写 `custom_metrics=` 直接 `TypeError`。

## 🛠️ 动手做

**练习 1：跑基础评估（04a）**
```bash
cd <project-root> && python scripts/04_evaluate/04a_evaluate_basics.py
```
✅ 验证标准：UI → `04_evaluate` → Run `evaluate-baseline`：Metrics 标签有 `accuracy_score`/`f1_score`/`roc_auc`；Artifacts → eval/ 有 `confusion_matrix.png`、`roc_curve_plot.png`（不用自己写 matplotlib）。

**练习 2：跑自定义指标（04b）**
```bash
python scripts/04_evaluate/04b_evaluate_custom.py
```
✅ 验证标准：`04_evaluate_custom` 下两个 Run（model-A-RF / model-B-LR）的 Metrics 里都有 `weighted_accuracy_v1`；脚本末尾打印「✓ 模型 B 通过验证」或「✗ 验证失败」。

**练习 3：起 serve 用 curl 调（04c）**
先起 server（sqlite 后端），再另开终端：
```bash
mlflow models serve -m "models:/WineQualityClassifier@champion" -p 5001
# 看到 "Listening on http://127.0.0.1:5001" 后，第三个终端：
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  --data '{"dataframe_records": [{"alcohol": 13.0, "malic_acid": 1.5, "ash": 2.5, "alcalinity_of_ash": 19.0, "magnesium": 100, "total_phenols": 2.8, "flavanoids": 3.0, "nonflavanoid_phenols": 0.3, "proanthocyanins": 1.8, "color_intensity": 5.0, "hue": 1.0, "od280/od315_of_diluted_wines": 3.0, "proline": 1000}]}'
```
✅ 验证标准：返回 `{"predictions": [0]}`；把 `Content-Type` 换成 `text/csv`、body 换成 CSV 文本（见 `04c_models_serve.sh` 4.2）也能得到预测。

## 📖 深入阅读（关键！）
> 📚 [`notes/04_evaluate.md`](04_evaluate.md)（Phase 4：评估、服务与经典 ML 验证）——MLflow 2 vs 3 完整对比、内置指标全集、`build-docker` 容器化流程、`04c` 完整部署脚本。

## 📖 下一步
→ [Chapter 11：LLM Agent 的版本追踪与打包](#chapter-11-llm-agent-的版本追踪与打包)
# Chapter 11：LLM Agent 的版本追踪与打包

> ⏱️ 预计时间：30 分钟
> 🔑 是否需 API Key：是（OpenAI 兼容服务，如 DeepSeek）
> 📚 前置知识：Chapter 6（Trace）、Chapter 8（Prompt Registry）

## 🎯 这章做什么

LLM 应用跑起来后要推向生产，会遇到三个问题：线上跑的是哪个版本的代码？prompt 谁来改进？LLM 应用怎么变成标准服务？这章用 MLflow 3 的三件新武器解决：**LoggedModel**（独立的模型版本实体）、**optimize_prompts**（让 reflection 模型自动改写 prompt 并打分）、**ResponsesAgent**（兼容 OpenAI Responses API 的 Agent 基类）。

**产出物**：跑完脚本后，UI 的 **Logged Models** 里出现 `agent-v1` 和 `agent-v2` 两个独立实体，且你能把一个自定义 Agent 打包成可 `mlflow models serve` 的模型。

### 你会学到什么

- 用 `set_active_model` 把同一份代码的不同版本登记为不同 LoggedModel，trace 自动归类
- 用 `mlflow.genai.optimize_prompts` 让 reflection 模型自动迭代改进 prompt
- 用 `ResponsesAgent` 基类把自定义 LLM 应用打包成模型，兼容 OpenAI Responses API
- 用 Models-from-code（`set_model()` + 文件路径字符串）打包复杂 Agent，避开 pickle
- 用 `mlflow.search_logged_models` 搜索对比多个 LoggedModel（A/B 测试、生产回滚、版本溯源）

### 前置

- 已完成 Chapter 6（Trace）与 Chapter 8（Prompt Registry）
- 需 API Key：脚本读 `OPENAI_API_KEY` / `OPENAI_API_BASE` / `DEEPSEEK_MODEL`，默认 DeepSeek
- 假设你懂 Python 类继承、dict/对象互转、context manager（`with`）

### 必跑脚本清单

| 脚本 | 一句话作用 | 前置 |
|------|-----------|------|
| `scripts/08_agents/08a_active_model.py` | 用 `set_active_model` 把两个 Agent 版本关联到不同 LoggedModel | Chapter 6 |
| `scripts/08_agents/08c_responses_agent.py` | 用 Models-from-code 打包 ResponsesAgent 到 Registry | 跑过 08a |
| `scripts/08_agents/simple_qa_agent.py` | `SimpleQAAgent` 类定义（被 08c import，非独立脚本） | — |

## 核心概念（一句话版）

- **LoggedModel**：MLflow 3 的独立"模型版本"实体，自带 `model_id`（`m-xxx`）、别名和 trace 列表，可跨 Run/实验搜索——不再像 MLflow 2 那样附庸在某个 Run 上。
- **`set_active_model(name="agent-v1")`**：在 trace 上下文里设"当前模型指针"，之后所有 trace 自动归属该版本；同一 name 多次调用自动复用。⚠️ 必须在 `@mlflow.trace` 装饰的函数或 trace 上下文里调用。
- **`setup_mlflow_git_based_version_tracking()`**：每个 git commit 自动生成一个 LoggedModel（`my-agent-<commit>`），配合 `get_git_commit()` 实现"一行代码"的版本溯源。
- **ResponsesAgent**：你继承它实现 `predict()`，MLflow 自动把请求/响应翻译成 OpenAI Responses 格式——前端用 OpenAI SDK 就能直接调你的服务。
- **Models-from-code**：`log_model` 的 `python_model` 传 `.py` 文件路径字符串（不是类实例），文件末尾必须 `set_model(YourClass())`；MLflow 加载时 import 该文件找模型类，避免 pickle 序列化失败。

> 💡 最小用法：
> ```python
> with mlflow.start_run():
>     mlflow.set_active_model(name="agent-v2")
>     agent_v2(q)      # 后续 trace 自动归属 agent-v2
> ```

## 🛠️ 动手做

1. **跑版本追踪并看 UI**：`cd <project-root> && python scripts/08_agents/08a_active_model.py`，另开终端 `cd <project-root> && mlflow ui --port 5000`。验证标准：左侧菜单 **Logged Models** 看到 `agent-v1`、`agent-v2` 两个实体；点开任一个，**Traces** 标签下各挂 3 条 trace。
2. **打包 ResponsesAgent**：`cd <project-root> && python scripts/08_agents/08c_responses_agent.py`，记录打印出的 `model_info.model_id`（形如 `m-xxxxxxxx`）。验证标准：experiment `08_responses_agent` 下出现 Run `agent-packaging`，其 **Artifacts** 里能看到 `MLmodel`、`requirements.txt` 和 `simple_qa_agent.py` 源码副本。

## 📖 深入阅读（关键！）

> 📚 [`notes/08_agents.md`](08_agents.md)（Phase 8：版本追踪、提示词优化与 ResponsesAgent——含 GEPA vs MetaPrompt 对比、Models-from-code 工作机制、ResponsesAgent 协议细节）

## 📖 下一步

→ [Chapter 12：生产级部署入门（选学）](#chapter-12)

---
# Chapter 12：生产级部署入门（选学）

> ⏱️ 预计时间：30 分钟
> 🔑 是否需 API Key：是（09a 会真实调一次 DeepSeek）
> 📚 前置知识：Chapter 6（Trace）、Chapter 10（模型评估与本地部署服务）
> ⭐ 选学：仅当你准备把 MLflow 推到生产环境时再读

## 🎯 这章做什么

前面你一直在自己电脑上跑 MLflow（本地 SQLite + 本地文件系统），相当于"在家做饭"。本章解决"开餐馆"的问题：**把 MLflow 真正推到生产环境**，会遇到三类新问题——存哪里（本地 SQLite/磁盘撑不住）、要不要全记（trace 存储成本爆炸）、机器扛不扛得住（CPU/磁盘监控）。

**产出物**：看懂生产三层架构、学会 trace 采样与 PII 脱敏、手里有一份能直接 `docker compose up` 的本地生产配置。

### 你会学到什么

- 看懂生产环境的"三层架构"：Client → Tracking Server → Backend Store + Artifact Store
- 用 `mlflow models build-docker` 把任意 MLflow 模型（含 ResponsesAgent）打成 Docker 镜像
- 用采样把 trace 存储成本压到原来的 1/10
- 在数据进 trace 之前用正则洗掉邮箱/手机/身份证
- 读懂 docker-compose 的 Postgres + MinIO + MLflow 三件套

### 前置

- 已完成 Chapter 6（Trace）与 Chapter 10（模型评估与本地部署），装好 `mlflow`、`psutil`、`openai`
- 不需要真买云服务——09b 的 docker-compose 在本地就能起
- 不懂 docker-compose 也能跑 09a

### 必跑脚本清单

| 脚本 | 一句话作用 | 前置 |
|------|-----------|------|
| `scripts/09_deployment/09a_sampling_redaction.py` | 演示 trace 采样 + PII 脱敏，对比 raw vs redacted 两条 Run | 无 |
| `scripts/09_deployment/09b_prod_infra.sh` | docker-compose 参考配置（Postgres + MinIO + MLflow） | Docker 环境 |

## 核心概念（一句话版）

- **三层架构**：Client（训练脚本/UI/API）→ Tracking Server（FastAPI，端口 5000）→ Backend Store（Postgres 存元数据）+ Artifact Store（S3/MinIO 存模型权重和 trace）；SQLite 上生产会锁表、本地磁盘存 artifact 容器一重启就丢。
- **Trace 采样**：不是每个请求都记 trace，按比例抽——调试 100%、一般生产 10-20%、高流量 1-5%；10% 采样省 90% 存储。
- **PII 脱敏**：在 trace 边界（函数入口第一行）就用正则把邮箱/手机/身份证替换成 `[EMAIL]`/`[PHONE]`/`[ID_CARD]`；一旦进了 trace 存储就等于泄漏，事后洗不干净。
- **`mlflow models build-docker`**：一条命令把任何 MLflow Model（sklearn / PyTorch / ResponsesAgent）打成自包含 Docker 镜像，`docker run -p 5001:8080` 即可推理。
- **docker-compose 生产部署**：一份 YAML 起 `postgres`（元数据）+ `minio`（S3 替代，存 artifact）+ `createbuckets`（初始化建 bucket）+ `mlflow server`；客户端设 `MLFLOW_TRACKING_URI=http://localhost:5000` 连接。

## 🛠️ 动手做

1. **跑 09a 看 PII 脱敏对比**：
   - 运行：`cd <project-root> && python scripts/09_deployment/09a_sampling_redaction.py`
   - 查看：另开终端 `cd <project-root> && mlflow ui --port 5000`，进 experiment `09_sampling_pii`
   - 验证标准：`raw-no-redaction` 的 trace_inputs 能看到 `zhangsan@example.com`、`13812345678`；`redacted` 同样的输入变成 `[EMAIL]`、`[PHONE]`、`[ID_CARD]`
   - ⚠️ 运行提示：脚本内部会真实调用一次 DeepSeek，如遇 API 报错，先设 `OPENAI_API_KEY` / `OPENAI_API_BASE` / `DEEPSEEK_MODEL`

2. **读 09b 了解生产 docker-compose 配置**：
   - 打开 `scripts/09_deployment/09b_prod_infra.sh`，把 YAML heredoc 存为 `docker-compose.yml`
   - 本地起三件套：`cd <project-root> && docker compose up -d`
   - 验证标准：浏览器 `http://localhost:5000` 打开由 Postgres + MinIO 支撑的 MLflow UI；客户端 `export MLFLOW_TRACKING_URI=http://localhost:5000` 连接

## 📖 深入阅读（关键！）

> 📚 [`notes/09_deployment.md`](09_deployment.md)（Phase 9：部署与生产可观测性——含完整 docker-compose 模板、生产 Checklist、月度成本估算、Prometheus + Grafana 接入）

## 📖 下一步

→ [Chapter 13：Debug 指南——遇到错误自己排查](#chapter-13)
# Chapter 13：Debug 指南——遇到错误自己排查

> ⏱️ 预计时间：30 分钟
> 🔑 是否需 API Key：否
> 📚 前置知识：任意章节都可以，遇到问题随时翻
> ⭐ 必学（Debug 技能不会过期）

## 🎯 这章做什么

学 MLflow 的路上你一定会遇到报错——`file:` backend 不能用 Registry、Run 卡在 RUNNING、`log_param` 报 "Changing param"... 这些错误有 6 个典型模式，学会自己排查能省 80% 的求助手时间。

**类比**：这章相当于"MLflow 急救手册"。每个错误按 **症状 → 原因 → 排查步骤 → 解决** 四步走，对照着抄作业就行。

### 你会学到什么

- 能独立排查 MLflow 6 类典型报错
- 知道 `mlflow --version`、`mlflow.get_tracking_uri()` 这些"诊断命令"
- 知道 "Debug 六层法"——从环境层往模型层逐步定位
- 学会看 stack trace 时**先看最后一行**（那是真正的报错）

---

## Debug 通用流程：六层法

当遇到任何 MLflow 错误，按这个顺序排查：

```
第 1 层  Environment  环境：mlflow --version、tracking URI、backend 类型
第 2 层  Run          Run：mlflow runs describe、search_runs、UI
第 3 层  Artifact     构件：mlflow artifacts list、download
第 4 层  Metrics      指标：mlflow runs search + metric 过滤、UI Compare
第 5 层  Logs         日志：stderr、log 文件、system metrics tab
第 6 层  Model        模型：mlflow.pyfunc.load_model + predict 单样本
```

**自上而下定位**。大多数生产 bug 都在第 1、3、6 层。

**类比**：和医生看病一样——先看体温（环境），再问症状（Run），再拍片（Artifact），最后才做手术（Model）。**不要跳级**。

---

## 问题 1：MLflow UI 启动失败

### 症状

```bash
$ mlflow ui --port 5000
[ERROR] Address already in use
Port 5000 is in use by another program
```

或者：

```bash
$ mlflow server --host 0.0.0.0 --port 5000 ...
[ERROR] InvalidHostHeader: The hostname '...' is not allowed.
```

或者：

```bash
$ curl http://localhost:5000
Connection refused
```

### 原因

**情况 A**——端口被占用：

- 另一个 MLflow 进程没关
- 其他程序（jupyter / docker / IDE）占了 5000

**情况 B**——allowed-hosts 限制：

- MLflow 3.5+ 默认拒绝未知 Host header
- 你用 IP 或自定义域名访问就报错

**情况 C**——服务没起来：

- 后端依赖（Postgres）没起来 → MLflow 启动后立即崩溃
- 配置文件写错了

### 排查步骤

**1. 看哪个进程占了端口**

```bash
# Linux / Mac
lsof -i :5000
# 或
netstat -tlnp | grep 5000
```

输出会显示 PID 和进程名，比如：

```
python 12345 user 50u IPv4 ... TCP *:5000 (LISTEN)
```

**2. 杀掉占用进程**

```bash
kill 12345
# 或者强制
kill -9 12345
```

**3. 加 `--allowed-hosts` 重启**

```bash
# 本地开发
mlflow ui --port 5000 --allowed-hosts "*"

# 生产
mlflow server --host 0.0.0.0 --port 5000 \
  --allowed-hosts "mlflow.example.com,localhost" \
  --backend-store-uri postgresql://... \
  --default-artifact-root s3://...
```

**4. 如果用 docker-compose，看容器日志**

```bash
docker compose logs mlflow
# 看最后 50 行
docker compose logs --tail=50 mlflow
```

### 解决

| 情况               | 解决                                                |
| ------------------ | --------------------------------------------------- |
| 端口被占           | 杀掉旧进程 或`mlflow ui --port 5001` 换端口       |
| allowed-hosts 报错 | 加`--allowed-hosts "*"` 或具体域名                |
| 服务没起来         | 看 docker-compose 依赖顺序（postgres 是否 healthy） |
| curl 连不上        | 检查防火墙、container 网络、host port 映射          |

### 预防

- 启动前先 `lsof -i :5000` 确认端口空闲
- MLflow 3.5+ 永远显式写 `--allowed-hosts`
- docker-compose 的 `depends_on` 加 `condition: service_healthy`

---

## 问题 2：Run 一直 RUNNING

### 症状

```python
with mlflow.start_run():
    train()

# 跑完发现 UI 里这条 Run 的状态一直是 RUNNING
```

或者：

```python
mlflow.start_run()
train()
# 忘了写 mlflow.end_run()
```

### 原因

- **`with` 块抛异常**，未走到 `__exit__` → Run 没正常结束（但 Python 会自动 end_run，不过有时会被 SIGKILL 卡住）
- **进程被 kill -9**（不是 Ctrl+C）→ MLflow 收不到 SIGTERM，无法清理
- **脚本里没写 `with`** → 你以为 start 了，其实 context 没正常关闭
- **网络中断** → MLflow server 重连不上，Run 卡死

### 排查步骤

**1. 在 UI 看 Run 的状态**

UI → Runs 列表 → 看 Status 列。RUNNING 表示没结束。

**2. 在 Python 里查 Run 状态**

```python
from mlflow import MlflowClient

client = MlflowClient()
run = client.get_run("your-run-id")
print(run.info.status)        # RUNNING / FINISHED / FAILED / KILLED
print(run.info.end_time)      # None 表示没结束
```

**3. 用 CLI 强终**

```bash
mlflow runs terminate --run-id <run-id>
```

或者：

```python
client.set_terminated(run_id, status="KILLED")
```

**4. 看日志确认为什么没结束**

```bash
# 看 stderr 里有没有 "Connection reset" / "timeout"
```

### 解决

| 场景            | 解决                                                     |
| --------------- | -------------------------------------------------------- |
| 进程被 kill -9  | 跑`mlflow runs terminate --run-id <id>` 手动关闭       |
| 忘了`with` 块 | 改用`with mlflow.start_run():` 包起来（自动清理）      |
| 异常退出        | `try / except / finally` 里手动 `mlflow.end_run()`   |
| 网络中断        | 检查`mlflow server` 是否还活着，重连后跑 `terminate` |

### 预防

- **永远用 `with` 块**，不要裸调 `start_run()`
- 关键任务加 `client.set_terminated()` 兜底
- 进程信号处理：注册 SIGTERM handler 调 `end_run()`

```python
import signal, mlflow

def cleanup(signum, frame):
    mlflow.end_run()
    exit(0)

signal.signal(signal.SIGTERM, cleanup)

with mlflow.start_run():
    train()
```

---

## 问题 3：`register_model` 报错

### 症状

```python
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="my-model"
)
# 报错：
# MlflowException: UnsupportedModelRegistryStoreURIException:
# Model Registry functionality is unavailable;
# got unsupported URI 'file:///...' for registry data storage.
```

### 原因

**MLflow 的 Model Registry 必须用 SQL 后端**（SQLite 或 Postgres）。`file:///` 后端不支持。

**为什么会用 `file:///`？** 因为你跑 demo 时为了简单用了 `mlflow.set_tracking_uri("file:///./mlruns")`，没有显式设 backend。MLflow 在 `file://` 模式下注册模型会直接拒绝。

### 排查步骤

**1. 看 tracking URI**

```python
import mlflow
print(mlflow.get_tracking_uri())
```

输出如果是 `file:///...`，说明用了 file backend。

**2. 切换到 SQLite（最简单）**

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

或者 Postgres（生产）：

```python
mlflow.set_tracking_uri("postgresql://user:pass@host:5432/mlflow")
```

**3. 重新 log 模型 + 注册**

```python
import mlflow.sklearn

with mlflow.start_run():
    mlflow.sklearn.log_model(model, "model")
    run_id = mlflow.active_run().info.run_id

mlflow.register_model(f"runs:/{run_id}/model", "my-model")
```

### 解决

```bash
# 改 backend 到 SQL
export MLFLOW_TRACKING_URI=sqlite:///$(pwd)/mlflow.db
python your_script.py
```

或者在 Python 里：

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

**注意**：换 backend 后，旧 `mlruns/` 目录里的 Run 在新 backend 里看不到。**两种 backend 数据不互通**。

### 预防

- 一开始就用 SQLite，不要图省事用 file://
- `file://` 仅适合纯本地 demo，绝对不要注册模型

---

## 问题 4：`log_param` 警告 "Changing param"

### 症状

```python
mlflow.log_param("learning_rate", 0.01)
mlflow.log_param("learning_rate", 0.001)

# 警告：
# WARNING mlflow.metrics: Changing param learning_rate from 0.01 to 0.001
```

### 原因

**Param 在 MLflow 里是不可变的（immutable）**。同一 Run 里 `log_param("x", v1)` 后再 `log_param("x", v2)` 不会报错，但会**警告**且只保留最后一次的值。

这设计是故意的：param 描述"实验配置"，应该是稳定的。如果训练过程中改 LR，那是 hyperparameter schedule，应该记成 metric。

### 排查步骤

**1. 确认 param 是"实验配置"还是"训练过程"**

| 类型     | 例子                                  | 用 log_param 还是 log_metric |
| -------- | ------------------------------------- | ---------------------------- |
| 实验配置 | learning_rate, batch_size, model_type | `log_param`                |
| 训练过程 | epoch_loss, lr_schedule, step         | `log_metric`               |
| 评估指标 | accuracy, f1, auc                     | `log_metric`               |

**2. 是不是循环里误调了 log_param？**

```python
# 反例：在 epoch 循环里 log param
for epoch in range(10):
    mlflow.log_param("current_epoch", epoch)  # 错了！这是 metric
    train()
```

### 解决

```python
# 正确做法：epoch 是 metric
for epoch in range(10):
    mlflow.log_metric("epoch", epoch, step=epoch)
    mlflow.log_metric("loss", loss, step=epoch)
    train()
```

如果你真的需要覆盖 param（比如修正 typo），明确用 `set_tag` 或者删 Run 重跑：

```python
# 删除 Run 重跑
client = mlflow.tracking.MlflowClient()
client.delete_run(run_id)
```

### 预防

- 训练脚本里 param 只在 `with start_run()` 第一行记一次
- epoch / step / schedule 永远用 `log_metric`
- 看到警告就检查是不是循环里写错了

---

## 问题 5：LoggedModel 加载失败

### 症状

```python
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("models:/my-model/1")
# 报错：
# RESOURCE_DOES_NOT_EXIST: No model found at models:/my-model/1
```

或者：

```python
model = mlflow.pyfunc.load_model("runs:/abc/model")
# 报错：
# FileNotFoundError: ... model/MLmodel
```

或者：

python
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("models:/WineQualityClassifier@champion")

# 报错：

# MlflowException: Registered Model with name=... not found

```

### 原因

**情况 A**——URI 错误：
- `models:/my-model/1`：`1` 是 version number，必须是整数
- `models:/my-model@champion`：`champion` 是 alias，**MLflow 3 已经移除 "stage"（Staging/Production）**，只有 alias
- `runs:/<run_id>/model`：`<run_id>` 必须完整（一般 32 位 hex）

**情况 B**——`file://` 后端不支持 Registry：

参见问题 3。

**情况 C**——DB 没 upgrade：
- 旧 MLflow 版本的数据库 schema 没升级
- 第一次在新版本 server 上跑会失败

**情况 D**——artifact 文件丢了：
- Run 删了但模型名还注册着
- artifact store 路径变了

### 排查步骤

**1. 确认 URI 格式**

```python
# 正确写法
models:/<model_name>/<version>          # 按版本
models:/<model_name>@<alias>            # 按 alias
runs:/<run_id>/<artifact_path>          # 按 Run
```

**2. 查模型是否存在**

```python
from mlflow import MlflowClient

client = MlflowClient()

# 查所有 registered models
models = client.search_registered_models()
for m in models:
    print(m.name, [v.version for v in m.latest_versions])

# 查特定 model 的所有版本
versions = client.search_model_versions("name='my-model'")
for v in versions:
    print(v.version, v.current_stage, v.aliases)
```

**3. 看 artifact 是否还在**

```bash
mlflow artifacts list --run-id <run-id>
```

或者 Python：

```python
client.list_artifacts(run_id, "model")
```

**4. 如果是 DB schema 问题，upgrade**

```bash
mlflow db upgrade <database-uri>
# 例：
mlflow db upgrade sqlite:///mlflow.db
mlflow db upgrade postgresql://user:pass@host:5432/mlflow
```

### 解决

| 场景               | 解决                                                          |
| ------------------ | ------------------------------------------------------------- |
| URI 写错           | 用`search_registered_models()` 查正确的 version/alias       |
| `file://` 不支持 | 换 SQLite：`mlflow.set_tracking_uri("sqlite:///mlflow.db")` |
| DB schema 旧       | 跑`mlflow db upgrade <uri>`                                 |
| artifact 丢了      | 重新 log 模型 + 注册                                          |

### 预防

- 加载模型前先 `search_registered_models()` 验证名字存在
- 升级 MLflow 版本时**先**跑 `mlflow db upgrade`
- 别在 `file://` 后端注册模型（参见问题 3）

---

## 问题 6：trace 没出现

### 症状

```python
@mlflow.trace
def my_agent(q):
    return llm(q)

# 跑完 my_agent("hello")，UI 里没看到 trace
```

或者：

```python
with mlflow.start_run():
    my_agent("hello")
# 跑完 Run 里没有 Traces 标签的内容
```

### 原因

**情况 A**——tracking URI 不对：

- trace 默认跟 `MLFLOW_TRACKING_URI`，如果 client 连到 A server，但 UI 看的是 B server
- 或者 tracking URI 设了 file://，但 Run 用的是 SQLite（两边数据不通）

**情况 B**——autolog 没开：

- LLM 调用（OpenAI / Anthropic）要 `mlflow.openai.autolog()`，不是装饰器

**情况 C**——async flush 没等：

- trace 是异步 flush 的，脚本跑完直接退出 → 最后几条 trace 没写完
- 大 trace 序列化时阻塞

**情况 D**——experiment 不对：

- trace 写在 experiment A 里，你却去 experiment B 里找

### 排查步骤

**1. 确认 tracking URI 一致**

```python
import mlflow
print(mlflow.get_tracking_uri())   # 脚本里的 URI
```

打开 UI 看浏览器地址栏的 URL——必须**完全一致**。

**2. 确认 experiment ID**

```python
exp = mlflow.get_experiment_by_name("my-exp")
print(exp.experiment_id)
```

UI 里找 experiment，看 ID 是不是同一个。

**3. 等异步 flush**

```python
import time
time.sleep(2)   # 留 2 秒让 trace flush
```

或者**显式 flush**：

```python
mlflow.flush_trace_logging()   # MLflow 3+
```

**4. 主动搜 trace**

```python
# 按 run_id 查
traces = mlflow.search_traces(run_id="abc123", max_results=10)
print(len(traces))

# 按 experiment 查
traces = mlflow.search_traces(
    experiment_ids=[exp_id],
    max_results=10
)
```

如果 `search_traces` 能找到但 UI 看不到——是 UI 缓存问题，强刷一下。

### 解决

| 场景            | 解决                                                           |
| --------------- | -------------------------------------------------------------- |
| URI 不一致      | 显式`mlflow.set_tracking_uri()` + 重启 UI                    |
| autolog 没开    | 加`mlflow.openai.autolog()` / `mlflow.anthropic.autolog()` |
| async 没 flush  | 脚本结尾`time.sleep(2)` 或 `mlflow.flush_trace_logging()`  |
| experiment 找错 | 用`get_experiment_by_name` 拿 ID 后再 search                 |

### 预防

- 脚本顶部统一设 `mlflow.set_tracking_uri()`
- 开启对应 LLM 的 autolog
- 脚本结尾 `time.sleep(2)` 等 flush（成本很低）
- 用 `mlflow.search_traces()` 程序化检查，别只靠 UI

---

## 调试速查表（打印贴桌边）

| 症状                               | 第一反应                                                   |
| ---------------------------------- | ---------------------------------------------------------- |
| 启动报错                           | `lsof -i :5000` + `--allowed-hosts "*"`                |
| Run 卡 RUNNING                     | `mlflow runs terminate --run-id <id>`                    |
| register_model 失败                | `mlflow.set_tracking_uri("sqlite:///mlflow.db")`         |
| Changing param 警告                | 改用`log_metric`                                         |
| load_model RESOURCE_DOES_NOT_EXIST | `mlflow db upgrade <uri>` + `search_registered_models` |
| trace 没出现                       | `mlflow.search_traces()` + `time.sleep(2)`             |

---

## 📖 下一步

这一章是工具书——遇到报错随时翻，不需要从头读到尾。

**更全的错误信息**：完整的错误信息、stack trace 示例、CLI 命令清单见 `mlflow_skills/classical-ml/references/troubleshooting.md`。那篇文档是 MLflow Debug 的"完整版"，包含：

- 六层 Debug 工作流的每个命令详解
- 每个层的常见错误 + 修复命令
- A-Z 索引（25+ 个错误按字母排序）

当你遇到本章没列的错误时，去那篇文档搜关键词（建议 Ctrl+F 搜错误信息的最后一行）。

**学完这一章你应该会**：

- 看 stack trace 时**先看最后一行**（不要被中间调用栈吓到）
- 区分"环境问题"和"代码问题"——环境问题看 `mlflow --version` 和 `get_tracking_uri()`
- 知道 MLflow 报错最常见三类根因：URI 不对、backend 不支持、版本不匹配

接下来可以继续看 **Chapter 14: 项目总结 + 简历模板**——把学到的所有东西打包成"我能拿这个做什么"的故事。---

<a id="chapter-14"></a>

# Chapter 14：参考速查

> ⏱️ 预计时间：5 分钟
> 🔑 是否需 API Key：否
> 📚 前置知识：所有前置章节

## 🎯 这章做什么

一页纸常用命令 + API 速查，打印贴墙方便日常参考。

## 14.1 启动命令速查

```bash
# 装环境
conda create -n mlflow python=3.11 -y
conda activate mlflow
pip install mlflow scikit-learn pandas numpy jupyter

# 启动 UI
mlflow ui --port 5000
mlflow ui --port 5000 --allowed-hosts "localhost,127.0.0.1"  # MLflow 3.5+ 推荐

# 启动 Tracking Server（生产）
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 --port 5000

# 启动模型服务
mlflow models serve -m models:/<name>@champion -p 5001

# 批量预测
mlflow models predict -m models:/<name>@champion -i input.csv -o output.csv

# 数据库升级（首次用 Registry 必须跑）
mlflow db upgrade sqlite:///mlflow.db
```

## 14.2 加载模型 URI 速查

```python
# 4 种 URI 形式
mlflow.sklearn.load_model("runs:/<run_id>/<path>")            # 从 Run（debug 用）
mlflow.sklearn.load_model("models:/<name>/<version>")         # 用版本号
mlflow.sklearn.load_model("models:/<name>@<alias>")            # 用别名（推荐）
mlflow.pyfunc.load_model("models:/<model_id>")                  # MLflow 3 LoggedModel
```

## 14.3 搜索 API 速查

```python
# 搜索 Run
import mlflow
runs = mlflow.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.accuracy > 0.9 AND params.model = 'rf'",
    order_by=["metrics.accuracy DESC"],
    max_results=20,
)

# 搜索 LoggedModel（MLflow 3 新 API）
models = mlflow.search_logged_models(
    experiment_ids=["1"],
    filter_string="metrics.accuracy_score > 0.85",
    order_by=[{"field_name": "metrics.accuracy_score", "ascending": False}],
    output_format="list",
)

# 搜索 Trace
traces = mlflow.search_traces(
    experiment_ids=["1"],
    filter_string="metadata.`mlflow.trace.user` = 'alice'",
    order_by=["execution_time_ms DESC"],
)
```

## 14.4 autolog 一行开启速查

```python
# 传统 ML
mlflow.sklearn.autolog()          # scikit-learn
mlflow.xgboost.autolog()          # XGBoost
mlflow.lightgbm.autolog()         # LightGBM
mlflow.catboost.autolog()         # CatBoost
mlflow.tensorflow.autolog()       # TF/Keras
mlflow.pytorch.autolog()         # PyTorch Lightning only

# LLM/Agent
mlflow.openai.autolog()           # OpenAI
mlflow.anthropic.autolog()        # Anthropic
mlflow.langchain.autolog()        # LangChain
mlflow.llama_index.autolog()     # LlamaIndex
mlflow.dspy.autolog()            # DSPy
```

## 14.5 环境变量速查

```bash
# 国内 LLM 服务商（任意一个即可）
export DEEPSEEK_API_KEY=sk-xxx
export OPENAI_API_BASE=https://api.deepseek.com/v1    # MLflow 走 OpenAI 协议
# 或
export ZHIPU_API_KEY=xxx
export OPENAI_API_BASE=https://open.bigmodel.cn/api/paas/v4/
# 或
export DASHSCOPE_API_KEY=sk-xxx
export OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
# 或
export MOONSHOT_API_KEY=sk-xxx
export OPENAI_API_BASE=https://api.moonshot.cn/v1

# MLflow Tracking
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db       # 或 http://server:5000
export MLFLOW_EXPERIMENT_NAME=my_experiment
export MLFLOW_REGISTRY_URI=...                       # 单独设置 registry backend
```

## 14.6 Python API 速查

```python
import mlflow

# Tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("name")
mlflow.log_param(k, v)
mlflow.log_params({...})
mlflow.log_metric(k, v, step=epoch)
mlflow.log_metrics({...}, step=epoch)
mlflow.log_artifact("file.txt")
mlflow.set_tag(k, v)
mlflow.set_tags({...})
mlflow.log_input(dataset, context="training")

# Models
mlflow.sklearn.log_model(model, name="model", signature=..., input_example=...)
mlflow.pytorch.log_model(...)
mlflow.pyfunc.log_model(...)

# Registry
mlflow.register_model("runs:/<id>/model", "name")
client = mlflow.MlflowClient()
client.set_registered_model_alias("name", "champion", version=1)
client.set_registered_model_tag("name", "stage", "production")

# Loading
model = mlflow.sklearn.load_model("models:/<name>@<alias>")
pyfunc_model = mlflow.pyfunc.load_model("models:/<name>/<version>")

# Prompt Registry
mlflow.genai.register_prompt(name, template, ...)
mlflow.genai.set_prompt_alias(name, alias, version)
prompt = mlflow.genai.load_prompt("prompts:/<name>@<alias>")

# Evaluate
mlflow.models.evaluate(model, data, targets, model_type="classifier")
mlflow.genai.evaluate(data, predict_fn, scorers=[...])
mlflow.validate_evaluation_results(thresholds, candidate, baseline)
```

## 14.7 CLI 命令速查

```bash
# UI
mlflow ui --port 5000
mlflow server --backend-store-uri sqlite:///mlflow.db

# 数据库
mlflow db upgrade <uri>

# Models
mlflow models serve -m <uri> -p <port>
mlflow models predict -m <uri> -i <input>
mlflow models build-docker -m <uri> -n <image>

# 搜索（CLI 版）
mlflow experiments search
mlflow runs search --experiment-id 1 --filter-string "metrics.accuracy > 0.9"
mlflow models list

# GC
mlflow gc --backend-store-uri sqlite:///mlflow.db
```

## 14.8 MLflow 3 破坏性变化速查

| 旧（不要用）                                                      | 新（用这个）                                                                                                                                            |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `artifact_path=`                                                | `name=`                                                                                                                                               |
| `transition_model_version_stage(..., "Production")`             | `set_registered_model_alias(..., "champion", version)`                                                                                                |
| `mlflow.evaluate(..., baseline_model=uri, custom_metrics=[fn])` | `mlflow.models.evaluate(..., extra_metrics=[make_metric(eval_fn=fn, ...)])` + `mlflow.validate_evaluation_results(thresholds, candidate, baseline)` |
| `runs:/<id>/<path>` 加载模型                                    | `models:/<model_id>` （MLflow 3 LoggedModel）                                                                                                         |
| 模型 URI 含`runs:/`                                             | `models:/<name>@<alias>`                                                                                                                              |
| `Stage`（Staging/Production）                                   | `Alias`（champion/challenger）                                                                                                                        |
| `mlflow.evaluate`                                               | `mlflow.models.evaluate`（经典 ML）/ `mlflow.genai.evaluate`（GenAI）                                                                               |
| `mlflow.pyfunc.log_model(python_model=class_instance())`        | `mlflow.pyfunc.log_model(python_model="path/to/file.py")` + `set_model()`                                                                           |

---

下一段（如果还有的话）会进入实战：选一个真实数据集（比如 sklearn 的 wine / Iris）完整跑一遍"训练 → 评估 → 注册 → 部署"流程，把这一段的 4 个 skill 串起来用。

---

# Skill 段：12 个 MLflow Skill 介绍

> 本段面向所有读者：项目 `mlflow_skills/` 目录里有 12 个 `SKILL.md` 文件——它们是给 AI 编程助手读的"操作手册"（比如 Claude Code、Cursor 等）。本段不教你怎么让 AI 启用这些 skill（每个 AI 助手配置不同），而是告诉你**这 12 个 skill 各自管什么、什么场景需要用、用的时候看哪段**。
>
> 即使你不用 AI 助手，看 `SKILL.md` 本身也是学习 MLflow 最佳实践的好材料。

---

## 一、12 个 Skill 总览

下表是 `mlflow_skills/` 目录下全部 skill 的速查。每行告诉你：这个 skill 叫什么、它管什么、什么场景要用它、要读哪段 `SKILL.md`。

| #  | Skill 名字                            | 用途                                                                              | 触发场景（你说什么话）                                                            | 读哪段                                                      |
| -- | ------------------------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| 1  | `mlflow-onboarding`                 | 引导上手：判断你要做什么，给出 quickstart                                         | "怎么开始用 MLflow"、"刚装好"、"新手入门"                                         | `mlflow-onboarding/SKILL.md`                              |
| 2  | `classical-ml`                      | 传统 ML 6 步法：tracking → registry → evaluate → deploy → monitor → optimize | "训练 sklearn/xgboost"、"对比 runs"、"注册模型"、"部署模型"、"模型监控"、"调超参" | `mlflow_skills/classical-ml/SKILL.md`（最完整）            |
| 3  | `instrumenting-with-mlflow-tracing` | 给 LLM 代码加 tracing（Python / TypeScript）                                      | "给我的 OpenAI 加 trace"、"给 LangChain 加追踪"                                   | `mlflow_skills/instrumenting-with-mlflow-tracing/SKILL.md` |
| 4  | `agent-evaluation`                  | 评估 LLM agent 输出质量（dataset + scorer + evaluate）                            | "评估我的 agent"、"算准确率"、"用 LLM-as-judge 评分"                              | `mlflow_skills/agent-evaluation/SKILL.md`                  |
| 5  | `querying-mlflow-metrics`           | 拉聚合指标（token 用量、延迟、成本、trace 数）                                    | "分析 token 用量"、"看延迟趋势"、"算 LLM 成本"                                    | `mlflow_skills/querying-mlflow-metrics/SKILL.md`           |
| 6  | `retrieving-mlflow-traces`          | 搜索 / 过滤 trace                                                                 | "找失败的 trace"、"查 latency > 5s 的"                                            | `mlflow_skills/retrieving-mlflow-traces/SKILL.md`          |
| 7  | `analyze-mlflow-trace`              | debug 单个 trace                                                                  | "这个 trace 哪里出错了"、"trace ID 是 tr-xxx 帮我看看"                            | `mlflow_skills/analyze-mlflow-trace/SKILL.md`              |
| 8  | `analyze-mlflow-chat-session`       | debug 多轮对话 / session                                                          | "看这个 chat session 哪里出问题"                                                  | `mlflow_skills/analyze-mlflow-chat-session/SKILL.md`       |
| 9  | `fix-agent-issue`                   | 改 agent 行为的探索→计划→实现→验证闭环                                         | "agent 行为不对"、"想加个业务规则"                                                | `mlflow_skills/fix-agent-issue/SKILL.md`                   |
| 10 | `mlflow-agent`                      | 通用 MLflow master dispatcher（不知道用哪个就让它路由）                           | 任何 MLflow workflow 但你没说要用哪个 skill                                       | `mlflow_skills/mlflow-agent/SKILL.md`                      |
| 11 | `searching-mlflow-docs`             | 拉官方文档（mlflow.org/docs/latest）                                              | "MLflow 怎么用 X"、"查 MLflow API"                                                | `mlflow_skills/searching-mlflow-docs/SKILL.md`             |
| 12 | `sagemaker-mlflow`                  | 连 AWS SageMaker Managed MLflow 当后端                                            | "SageMaker 上装 MLflow"                                                           | `mlflow_skills/sagemaker-mlflow/SKILL.md`                  |

> **怎么用上表**：当你遇到一个 MLflow 任务时，先看"触发场景"列有没有匹配的关键词。匹配了就去找对应 skill 的 `SKILL.md` 读。读完不一定要让 AI 帮你做，自己按步骤跑也行。

---

## 二、按你的需求选 skill

不知道用哪个？按下面这张表对号入座：

| 你想做什么                                                      | 推荐 skill                            | 读哪段                                                     |
| --------------------------------------------------------------- | ------------------------------------- | ---------------------------------------------------------- |
| **第一次用 MLflow**，不知道怎么开始                       | `mlflow-onboarding`                 | `mlflow-onboarding/SKILL.md`                             |
| **训练 sklearn/xgboost/lightgbm** 模型并自动记录          | `classical-ml`                      | `mlflow-skill/classical-ml/SKILL.md`（Step 1: Tracking） |
| **对比多个模型** 找最好的                                 | `classical-ml`                      | `classical-ml/SKILL.md`（Step 3: Evaluate）              |
| **给 LLM 代码加 trace**（OpenAI / LangChain / Anthropic） | `instrumenting-with-mlflow-tracing` | `instrumenting-with-mlflow-tracing/SKILL.md`             |
| **评估 LLM agent 答得准不准**                             | `agent-evaluation`                  | `agent-evaluation/SKILL.md`                              |
| **查 token 用量、延迟、成本**                             | `querying-mlflow-metrics`           | `querying-mlflow-metrics/SKILL.md`                       |
| **找哪个 trace 失败了**                                   | `retrieving-mlflow-traces`          | `retrieving-mlflow-traces/SKILL.md`                      |
| **debug 单个 trace 哪里出问题**                           | `analyze-mlflow-trace`              | `analyze-mlflow-trace/SKILL.md`                          |
| **debug 多轮对话**                                        | `analyze-mlflow-chat-session`       | `analyze-mlflow-chat-session/SKILL.md`                   |
| **想改 agent 行为**（业务规则 / 偏好）                    | `fix-agent-issue`                   | `fix-agent-issue/SKILL.md`                               |
| **不知道用哪个 skill**                                    | `mlflow-agent`                      | `mlflow-agent/SKILL.md`                                  |
| **查 MLflow 官方文档**                                    | `searching-mlflow-docs`             | `searching-mlflow-docs/SKILL.md`                         |
| **在 AWS SageMaker 上部署 MLflow**                        | `sagemaker-mlflow`                  | `sagemaker-mlflow/SKILL.md`                              |

---

## 三、4 个最常用 skill 的 SKILL.md 导读

下表让你**第一次打开 SKILL.md** 时知道重点看哪段，不用通读。每个 SKILL.md 通常包含：description、Step 1-N（步骤）、references/（深入）、scripts/（脚本）。

### 3.1 `classical-ml/SKILL.md`（最完整，先读这个）

- **6 步法结构**：Step 1 Tracking → Step 2 Registry → Step 3 Evaluate → Step 4 Deploy → Step 5 Monitor → Step 6 Optimize
- **重点看**：
  - `> ⛔ CRITICAL: Must Use MLflow 3 APIs and Pick the Right Backend` 段（最容易踩的坑）
  - Step 1b 的 autolog 列表（不同框架不同）
  - Step 2 的 Stage→Alias 迁移表
- **scripts/**：`validate_environment.py`（必跑）、`search_logged_models.py`（跨实验找最好模型）
- **references/**：tracking.md / registry.md / evaluate.md / deploy.md / monitor.md / optimize.md（按需深读）

### 3.2 `mlflow-onboarding/SKILL.md`

- **怎么用**：是其他 11 个 skill 的"路由"
- **重点看**：怎么判断你走 GenAI 还是传统 ML 路径
- **通常很短**（只有 routing logic），深入内容在其它 skill 里

### 3.3 `instrumenting-with-mlflow-tracing/SKILL.md`

- **怎么用**：给现有 LLM 代码加 `mlflow.openai.autolog()` / `mlflow.langchain.autolog()`
- **重点看**：Python vs TypeScript 各自的最小例子
- **references/**：通常按语言分（python.md / typescript.md）
- **关键提醒**：tracing 是"零侵入"——加 `autolog()` 一行就有

### 3.4 `agent-evaluation/SKILL.md`

- **怎么用**：给 LLM agent 建评估数据集 + 写 scorer + 跑评估
- **重点看**：内置 scorers（Correctness / Safety / RelevanceToQuery）+ LLM-as-judge
- **关键提醒**：评估数据要 ≥ 20 条才有效，少于这个数字的结果不可靠

---

## 四、不用 skill 也能用好 MLflow

**skill 不是必须**。如果你的工作流是：

1. 自己写 `mlflow.log_param/metric/model` 手动记录
2. 自己用 `mlflow ui` 看 Run
3. 自己 `mlflow.register_model` 注册
4. 自己 `mlflow models serve` 部署

那你**完全不需要**任何 skill。skill 主要是给 AI 编程助手用的"领域知识"——让 AI 知道 MLflow 怎么用，避免它瞎写。

> **总结**：skill 是给 AI 助手的"教科书"，不是给你的"用户手册"。本项目主要笔记（`notes/01_basics.md` 等）才是给你读的。

---

## 五、读 SKILL.md 的小贴士

- **每个 SKILL.md 顶部有 `---` 分隔的 YAML frontmatter**（`name`、`description`）。`description` 字段是 AI 助手的触发关键词来源，**重点读这个**就能知道 skill 管什么。
- **不熟悉的关键词**去 `references/` 子目录深读
- **scripts/** 下的脚本可以直接用命令行跑（如 `python scripts/validate_environment.py`），不需要 AI 帮你
- **不要全链入 12 个 skill**——少而精，按需启用或查看

---

## 小结

- **12 个 skill** 覆盖了 MLflow 全部工作流：onboarding → instrument → evaluate → debug → fix → metrics → docs → deploy
- **核心场景**只用到 4 个 skill：传统 ML 用 `classical-ml`，GenAI 用 `mlflow-onboarding` + `instrumenting-with-mlflow-tracing` + `agent-evaluation`
- **不需要 AI 助手也能用**——SKILL.md 本身是好材料，自己读、按步骤跑就行
- **验证靠 UI**：所有 skill 的最终验证都是去 MLflow UI 对应 tab 看一眼——Experiments / Models / Traces / Prompts
