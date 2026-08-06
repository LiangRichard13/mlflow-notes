# MLflow 3 作业指导书（QUICKSTART）

> **作者**：写给"MLflow 零基础 + 有计算机基础"的同学
> **目标**：1-2 天内从环境安装到跑通第一个实验，再到用 coding agent 在 vibecoding 场景下用 MLflow
> **范围**：覆盖 `MLFlowLearning/` 项目全部 10 个 phase + capstone 毕业项目 + 12 个 mlflow_skill

> **路径约定**：本文档所有命令里的 `<project-root>` 替换为你 clone 项目的实际路径（如 `~/projects/MLFlowLearning`、`/Users/you/code/MLFlowLearning`）。命令格式 `cd <project-root> && <cmd>` 意味着"先切到项目根目录，再执行命令"。

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

| 章节 | 时间 | 难度 | 必学 | API Key |
|------|------|------|------|---------|
| 0 前置认知 | 5 min | ★☆☆☆☆ | ✓ | 否 |
| 1 环境安装 | 20 min | ★★☆☆☆ | ✓ | 否 |
| 2 核心对象与追踪 | 30 min | ★★☆☆☆ | ✓ | 否 |
| 3 Model 格式 | 30 min | ★★★☆☆ | ✓ | 否 |
| 4 注册表与别名 | 30 min | ★★★☆☆ | ✓ | 否 |
| 5 Tracking Server + 血缘 | 30 min | ★★★☆☆ | △ | 否 |
| 6 GenAI 追踪 | 30 min | ★★☆☆☆ | ✓ | ✅ |
| 7 **vibecoding 集成** | 30 min | ★★★☆☆ | ✓ | - |
| 8 Prompt Registry | 30 min | ★★★☆☆ | △ | ✅ |
| 9 GenAI 评估 | 45 min | ★★★★☆ | △ | ✅ |
| 10 模型评估与部署 | 30 min | ★★★☆☆ | ✓ | 否 |
| 11 Agent Tracing | 45 min | ★★★★☆ | △ | ✅ |
| 12 生产部署 | 30 min | ★★★★☆ | △ | 否 |
| 13 Debug 指南 | 20 min | ★★★☆☆ | ✓ | 否 |
| 14 速查表 | 5 min | ★☆☆☆☆ | ✓ | - |
| Skill 段（12 个 skill） | 30 min | ★★☆☆☆ | ✓ | - |

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
- [Skill 段：12 个 mlflow_skill 介绍](#skill-段)

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

| 厨房概念 | MLflow 概念 | 说明 |
|---------|-----------|------|
| 厨房 | **Experiment（实验）** | 一组相关训练的容器 |
| 做菜的过程 | **Run（运行）** | 一次训练的执行 |
| 用什么食材/火候 | **Param（参数）** | `learning_rate=0.01`, `batch_size=32` |
| 出品评分 | **Metric（指标）** | `accuracy=0.95`, `loss=0.05` |
| 成品照片 | **Artifact（产物）** | 模型文件、图表、配置文件 |
| 备注标签 | **Tag（标签）** | `stage=baseline`, `team=alice` |

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
- ✅ 用 coding agent（vibecoding）通过 mlflow_skill 自动化 MLflow 操作

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

| MLflow 2（不要用） | MLflow 3（本项目用） |
|---|---|
| `mlflow.sklearn.log_model(..., artifact_path="m")` | `mlflow.sklearn.log_model(..., name="m")` |
| `transition_model_version_stage(..., "Production")` | `client.set_registered_model_alias(..., "champion", version)` |
| `mlflow.evaluate(..., baseline_model=uri)` | 两个 `mlflow.models.evaluate()` + `mlflow.validate_evaluation_results()` |
| 模型在 Run 下（`runs:/<id>/<path>`） | 模型独立（`models:/<model_id>`） |
| Stage（Staging/Production） | Alias（champion/challenger） |

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
├── 01_basics/                     ← Phase 1 脚本：传统 ML 基础
├── 02_registry/                   ← Phase 2：模型注册
├── 03_tracking/                   ← Phase 3：Tracking Server
├── 04_evaluate/                   ← Phase 4：评估与部署
├── 05_tracing/                    ← Phase 5：GenAI 追踪
├── 06_prompts/                    ← Phase 6：Prompt Registry
├── 07_evaluation/                 ← Phase 7：GenAI 评估
├── 08_agents/                     ← Phase 8：Agent + 版本化
├── 09_deployment/                 ← Phase 9：生产部署
├── 10_vision_classification/      ← Phase 10：图像分类（深度学习）
├── capstone/                      ← 毕业项目 SupportPilot
├── mlflow_skill/                  ← 12 个 MLflow skill（vibecoding 用）
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
   $ conda run -n mlflow python 01_basics/01_hello_mlflow.py
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
> 📚 前置知识：第 1 章（环境准备与 MLflow 是什么）

## 🎯 这章做什么

这一章带你跨进 MLflow 的大门——搞清楚"跑实验时到底该怎么把每次训练的关键信息记下来"。你可能已经有过这样的痛苦经历：调了一晚上超参数，第二天回想"上次那组 lr=0.01、batch_size=32 的效果到底是多少来着？模型文件扔哪了？训练曲线截图存哪了？"。MLflow 就是来解决这个问题的——它像一个**实验记账本**，每次训练都自动帮你把参数、指标、模型文件、备注标签统统归档。

类比：如果你把每一次模型训练当成"做一道菜"，那 MLflow 就是你的**厨房日志**——记录每次用了什么食材（Param）、出品评分多少（Metric）、成品照片存在哪（Artifact）、备注标签比如"辣/不辣/试做"（Tag）。几个相关实验放在一起就是一道"实验项目"（Experiment）。

### 你会学到什么

- 用最基础的 3 个 API（`log_param` / `log_metric` / `log_artifact`）手动记录一次训练
- 用 `mlflow.sklearn.autolog()` 一行代码自动记录 sklearn 训练全过程
- 理解 MLflow 的 6 个核心对象：Experiment / Run / Param / Metric / Artifact / Tag
- 启动 `mlflow ui`，用 Compare 功能对比多个 Run 的效果
- 在 UI 里"复盘"一次调参，看清楚超参数怎么影响最终指标

## 核心概念：用人话讲清楚

### 1. Experiment（实验）—— 一个文件夹

一组相关 Run 的容器。比如"iris 分类项目"是一个 Experiment，里面跑的所有模型都是它的 Run。

类比：就像一个项目文件夹，里面放着你这次课题跑过的所有实验记录。在文件系统后端下，它真的就是一个目录：`mlruns/<experiment_id>/`。

### 2. Run（一次运行）—— 一个文件

单次训练的执行过程。每次 `start_run()` 就会产生一个 Run，有唯一的 `run_id`（一串 32 位十六进制字符）。

类比：相当于项目文件夹里的一篇"实验日记"。在文件系统下，它真的就是一个 `meta.yaml` + 若干 artifact 文件——所以"Run 是文件"这个比喻非常贴切。

### 3. Param（参数）—— 字段之一

字符串型配置，比如 `learning_rate=0.01`、`batch_size=32`、`optimizer="adam"`。**同一个 Param key 只能记一次**，所以适合记那些不会变的超参。

### 4. Metric（指标）—— 字段之一

数值型效果，比如 `loss=0.35`、`accuracy=0.92`。**可以带 step**，所以能记录每个 epoch 的 loss，画出训练曲线。

### 5. Artifact（产物）—— 字段之一

任意文件：模型文件、配置文件、训练曲线图、混淆矩阵图……MLflow 会把它们统一收在每个 Run 的 `Artifacts` 文件夹下。

类比：实验日记里贴的截图、附件、模型快照。

### 6. Tag（标签）—— 字段之一

任意备注文本，比如 `status=completed`、`dataset=iris`、`notes=第3次试做`。主要用于 UI 里过滤/搜索。Tag 可以随时改，不像 Param 那样"一次定型"。

### 7. autolog（自动记录）

很多框架（sklearn / pytorch / xgboost / lightgbm）MLflow 都提供了 `mlflow.<框架>.autolog()`，**一行代码**就帮你自动记录参数、指标、模型文件、签名等等。

### 一张图把它们串起来

```
Experiment "01_sklearn_iris"           ← 项目文件夹
├── Run "rf_deep"                       ← 实验日记 1
│   ├── Params: {n_estimators: 200, ...}      ← 字段：用了什么食材
│   ├── Metrics: {accuracy: 0.97, ...}        ← 字段：评分多少
│   ├── Artifacts: model/MLmodel, model.pkl   ← 字段：成品照片
│   └── Tags: {dataset: iris, ...}            ← 字段：备注标签
├── Run "rf_shallow"                    ← 实验日记 2
│   └── ...
└── Run "logreg_weak_reg"               ← 实验日记 3
    └── ...
```

记住这句话：**Experiment 是文件夹，Run 是文件，Param/Metric/Artifact/Tag 是字段。** 后面的所有操作都是在这五个层级上搬数据。

## 实战步骤：按顺序照做

### Step 1：环境准备

```bash
cd <project-root> && conda activate mlflow && pip install mlflow scikit-learn
```

如果你已经在 `mlflow` 虚拟环境里，可以省略 `conda activate` 那段。

### Step 2：跑第一个 demo（手动记录）

```bash
cd <project-root> && python 01_basics/01_hello_mlflow.py
```

这个脚本会：
- 创建 experiment `01_basics_demo`
- 跑 3 个 Run（每次随机选 `learning_rate` 和 `batch_size`）
- 记录 10 个 epoch 的 loss/accuracy 曲线
- 写两个 artifact 文件（`configs/config.txt`、`summaries/summary.md`）
- 给每个 Run 打 3 个 Tag

跑完会看到 "所有 Run 已记录！下一步" 的提示。

### Step 3：跑第二个 demo（autolog + 真实 sklearn）

```bash
cd <project-root> && python 01_basics/01b_sklearn_basics.py
```

这个脚本会：
- 创建 experiment `01_sklearn_iris`
- 跑 4 个 sklearn 模型（逻辑回归强/弱正则、随机森林深/浅）
- 每个都记录 accuracy、f1、模型文件（带签名 + input_example）

> 注意：脚本里写的是 `artifact_path="model"`（老写法），不会报错但会有 deprecation 警告。第 3 章会用 MLflow 3 推荐的新写法 `name=`。

### Step 4：启动 UI 看结果

另开一个终端：

```bash
cd <project-root> && mlflow ui --port 5000
```

浏览器打开 `http://localhost:5000`，按下面的顺序点一遍：

1. 左侧选 experiment `01_basics_demo`：看 3 个 Run 的 Param/Metric 对比
2. 左侧选 experiment `01_sklearn_iris`：勾选 4 个 Run，点 **Compare**，并排看 Param/Metric
3. 点开任一 Run 的 **Artifacts** tab → `model/` 目录下的 `model.pkl`、`MLmodel`、`conda.yaml`
4. 点开任一 Run 的 **Overview** tab：看 Tags（`dataset`、`task`）

### Step 5：在 UI 里 Compare 两个 Run（必做）

这是这一章最重要的一个动作——学会用 UI 复盘实验。

1. 在 experiment `01_sklearn_iris` 页面，**勾选全部 4 个 Run**（左侧小方框）
2. 点工具栏上的 **Compare** 按钮
3. 跳转到一个对比页面，你会看到一张并排的表格：

| Run | accuracy | f1_score | C | max_depth | n_estimators |
|-----|----------|----------|---|-----------|--------------|
| rf_deep | 0.97 | 0.97 | - | 10 | 200 |
| rf_shallow | 0.93 | 0.93 | - | 3 | 50 |
| logreg_strong_reg | 0.93 | 0.93 | 0.1 | - | - |
| logreg_weak_reg | 0.97 | 0.97 | 10.0 | - | - |

4. 选中其中 2 个 Run（比如 `rf_deep` 和 `rf_shallow`），页面会高亮出**它们差异的列**——一眼看出谁改了哪个超参、效果变了多少。

> 这就是 MLflow 的杀手锏：**参数 × 指标** 二维表格，传统文本日志看 10 分钟才能看出来的事，UI 上 5 秒钟搞定。

## 🛠️ 动手做：调参对比实验

**任务**：把 `01b_sklearn_basics.py` 里的 `n_estimators=200` 改成 `n_estimators=50`，再跑一次，去 UI 对比新老 Run。

### 操作步骤

1. 用编辑器打开 `<project-root>/01_basics/01b_sklearn_basics.py`
2. 找到第 103 行附近：

```python
{"n_estimators": 200, "max_depth": 10, "min_samples_split": 2},
```

3. 改成：

```python
{"n_estimators": 50, "max_depth": 10, "min_samples_split": 2},
```

（只改 `n_estimators` 这一个数字，其它不动。）

4. 重新跑脚本：

```bash
cd <project-root> && python 01_basics/01b_sklearn_basics.py
```

这次 `experiment "01_sklearn_iris"` 里会多出一个 Run（之前是 4 个，现在 5 个）。

5. 启动 UI（如果还没启动）：

```bash
cd <project-root> && mlflow ui --port 5000
```

6. 浏览器打开 `http://localhost:5000`，进 `01_sklearn_iris`：
   - 勾选 `rf_deep`（n_estimators=200）和新生成的 `rf_deep`（n_estimators=50）
   - 点 **Compare**
   - **找到高亮列 `n_estimators`**：左边 200，右边 50
   - 对比 `accuracy` 列：通常 200 棵树会略高于 50 棵（不过 iris 太简单，可能都满分）

### 验证标准

- UI 里能看到至少 5 个 Run（之前的 4 个 + 新的 1 个）
- Compare 视图下，新老 Run 的 `n_estimators` 列被高亮为差异列
- 新 Run 的 `accuracy` 和 `f1_score` 已正确记录（可能与原值相同或略低，取决于随机种子）
- 新 Run 的 Artifacts tab 下依然有 `model/MLmodel`、`model/conda.yaml`、`model/requirements.txt`

### 加分项

如果你想让对比更明显，可以再改一次 `max_depth=3`（浅树），重复上面流程。这样在 UI 里就能看到"50 棵深树 vs 50 棵浅树 vs 200 棵深树"三组对照。

## 避坑清单

- ⚠️ **坑 1：Run 没关程序就崩了**。手动调 `start_run()` 但忘了 `end_run()`，程序异常退出，UI 里 Run 状态一直是 `RUNNING`。**解决**：永远用 `with mlflow.start_run() as run:`，异常会自动结束 Run。

- ⚠️ **坑 2：`log_param` 同一个 key 第二次会警告**。控制台警告 "Changing param .. is not allowed"。Param 设计为"一次定型"，同名 key 不让覆盖。**解决**：如果是每次会变的值（比如运行中的状态），改用 `set_tag`。

- ⚠️ **坑 3：`log_metric` 同 key 不同 step 不会冲突**。这正是 Metric 的设计——同名 key 加 step 就能画曲线。例如：

  ```python
  for epoch in range(10):
      log_metric("loss", compute_loss(), step=epoch)
  ```

  UI 会自动画一张 loss 随 epoch 下降的图。

- ⚠️ **坑 4：Artifact 路径写错**。`log_artifact("/tmp/xxx")` 报路径错误，或者文件没出现在 UI 里。**解决**：传相对当前工作目录的路径，或者直接传文件名（会在当前目录找）。文件会被**复制**到该 Run 的 artifact 目录，不是软链。

- ⚠️ **坑 5：UI 启动后看不到新数据**。MLflow 默认 backend 是 `file:./mlruns`，每次跑脚本都写在当前目录的 `mlruns/` 下。如果你换了个目录跑脚本、又开 UI 看另一个目录，自然是空的。**解决**：保持 UI 和脚本都在 `<project-root>` 这个根目录下运行。

- ⚠️ **坑 6：SQLite 不适合高并发**。多进程同时写 `mlflow.db` 时偶发 `database is locked`。本地开发无所谓；多人协同请用 PostgreSQL/MySQL（第 3 章会正式启用 SQLite 后端）。

## 📖 下一步

到这里你应该能熟练使用 MLflow 的核心 API 了。下一章我们要解决一个关键问题：**训练出来的模型本身去哪了？怎么让它能被团队成员加载、能被部署服务消费？**

请继续阅读 **Chapter 3: MLflow Model 格式与 Model Registry**。

---

# Chapter 3：MLflow Model 格式与 Model Registry

> ⏱️ 预计时间：35 分钟
> 🔑 是否需 API Key：否
> 📚 前置知识：Chapter 2（Experiment / Run / Param / Metric / Artifact / Tag）

## 🎯 这章做什么

Chapter 2 我们学会了"记录"——把参数、指标写进 MLflow，训练完能回头查。但那时候还有个大问题没解决：**训练出来的模型本身去哪了？** 你把 `model.pkl` 用 pickle 存到本地，三个月后同事问你"线上跑的是哪个模型、当时用什么数据训的、输入要几列"，你多半答不上来。更糟的是，你想换个更好的模型上线，得手动改代码里的文件路径，还得重启服务。

这一章就是解决这些问题的。MLflow 提供了两样东西：**MLflow Model 格式**（把模型 + 依赖 + 输入输出说明打包成一个自描述的目录）和 **Model Registry**（模型的"版本仓库"，像 Git 之于代码）。

打个比方：如果 Chapter 2 的 Run 是"实验日记"，那 Registry 就是"产品货架"。日记里有几百次实验，货架上只放你精挑细选、贴好标签的那几个。而 **Alias（别名）** 就是货架上的标签牌——"champion"（现役冠军）这块牌子今天挂在 v1 上，明天可以挂到 v3 上，所有来取货的人（加载模型的服务）自动拿到新版本，**不用改一行代码、不用重启**。

**产出物**：跑完两个脚本，你会得到一个名为 `WineQualityClassifier` 的注册模型，它有 v1 版本、带完整签名（输入 13 列 float、输出 int）、挂着 `champion` 别名，并且能用一行 `mlflow.sklearn.load_model("models:/WineQualityClassifier@champion")` 在任何地方加载出来直接推理。

### 你会学到什么

- 读懂 MLflow Model 目录结构，尤其是 `MLmodel` 这个 YAML 元数据文件在说什么
- 用 `infer_signature()` 自动推断模型的输入输出 schema，让部署服务能自动校验请求格式
- 理解 `input_example` 在 UI 里长什么样、起什么作用
- 把 Run 里的模型注册（`register_model`）到 Model Registry，理解版本号是怎么自动累加的
- 理解 MLflow 3 的破坏性变化：`name=` 取代了 `artifact_path=`，Stage 已被 Alias 取代

## 核心概念：用人话讲清楚

### 3.1 MLflow Model 不是一个文件，是一个目录

新手最容易误解的一点：MLflow 保存的"模型"不是 `model.pkl` 那一个文件，而是**一整个自描述的目录**。

```
wine-classifier/
├── MLmodel              # 主元数据（YAML）—— 灵魂所在
├── model.pkl            # 序列化后的模型本体
├── python_env.yaml      # Python 版本 + 依赖
├── conda.yaml           # Conda 环境描述
├── requirements.txt     # pip 依赖清单
└── input_example.json   # 输入示例（log_model 传了 input_example 才有）
```

为什么要这么麻烦？因为**光有 pkl 是没法部署的**。别人拿到你的 pkl，不知道该用哪个 Python 版本、要装哪些包、输入要传几列什么类型。MLflow 把这些"上下文"全打包进去，于是这个目录可以直接丢给 `mlflow models serve` 起一个 REST 服务，或者打成 Docker 镜像。

### 3.2 MLmodel 文件：模型的"身份证"

`MLmodel` 是 YAML，跑完 02a 后你能在 UI 的 Artifacts 里看到它，长这样：

```yaml
artifact_path: wine-classifier
flavors:
  python_function:                # 通用 flavor：任何语言/框架都能用的统一入口
    env: conda.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    predict_fn: predict
  sklearn:                        # 原生 flavor：还原成真正的 sklearn 对象
    code: null
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: 1.5.0
mlflow_version: 3.x
model_size_bytes: 1234
run_id: abc123...
signature:                        # 输入输出 schema
  inputs: '[{"name": "alcohol", "type": "double"}, ...]'
  outputs: '[{"type": "long"}]'
```

**"flavors"（风味）是这里最值得理解的概念**。同一个模型可以有多种"读法"：

- `sklearn` flavor：加载后你拿到的是**真正的 sklearn Pipeline 对象**，可以访问 `.steps`、`.feature_importances_` 这些原生属性
- `python_function` flavor：加载后你拿到的是一个**统一的 `predict()` 接口**，不管底层是 sklearn、PyTorch 还是 XGBoost，用法完全一样

部署工具（比如 `mlflow models serve`）只认 `python_function`，所以它能一视同仁地服务任何框架的模型。这就是 flavor 设计的价值：**训练侧自由选框架，部署侧只需要一套代码**。

### 3.3 Signature（签名）：模型的"接口文档"

签名记录了模型**输入要什么、输出是什么**。它有两个实实在在的好处：

1. **部署时自动校验**：请求少传一列、类型传错了，服务会直接报清晰的错误，而不是在模型内部炸出一个看不懂的堆栈
2. **给人看的文档**：三个月后你自己回来看，UI 上直接列出 13 个特征名和类型，不用翻训练代码

推断签名只要一行——把训练输入和模型输出丢给 `infer_signature`，它自己去看列名和 dtype：

```python
from mlflow.models import infer_signature
signature = infer_signature(X_train, pipe.predict(X_train))
```

### 3.4 input_example：部署时的"冒烟测试样本"

`input_example` 是签名的好搭档：存几行真实输入样本进去。它有三重作用：

1. **签名忘了传时的 fallback**：如果你只传了 `input_example` 没传 `signature`，MLflow 会用它自动推断签名
2. **UI 里直接展示"请求长什么样"**：在 Artifacts tab 下能看到 `input_example.json`，是给团队成员看的最直观参考
3. **部署后做冒烟测试**：`mlflow models serve` 启动时会用它做一次健康检查

最小例子：

```python
import pandas as pd
X_sample = pd.DataFrame({
    "alcohol": [13.0, 12.5, 14.2],
    "malic_acid": [2.5, 2.0, 3.5],
    # ... 其余 11 列
})
mlflow.sklearn.log_model(
    model,
    name="my-model",
    signature=signature,
    input_example=X_sample,
)
```

### 3.5 Model Registry：模型的"Git 仓库"

Run 里的模型是**实验产物**——你可能跑了 200 次，其中 199 次都是垃圾。Registry 是**发布通道**——你从那 200 次里挑出好的，给它起个正式名字（`WineQualityClassifier`），它就有了 v1、v2、v3 的版本序列。

对照理解：

| 概念 | 类比 | 特点 |
|------|------|------|
| Run 里的模型 | 本地的一次 commit | 数量多，随手产生，用 `runs:/<run_id>/<name>` 引用 |
| Registered Model | 一个 Git 仓库 | 有名字，是一个逻辑上的"产品线" |
| Model Version | 打的 tag（v1、v2） | 注册一次自动 +1，不可变 |
| Alias | 指向某个 tag 的分支指针 | 可以随时改指向，如 `champion` → v3 |

### 3.6 LoggedModel：MLflow 3 的一等公民（独立于 Run）

MLflow 3 里有一个新手容易忽略但非常重要的变化：**LoggedModel 是一等公民，不再寄生在 Run 下**。

在 MLflow 2 里，模型必须挂在某个 Run 的 Artifacts 里，要访问模型得先有 run_id。MLflow 3 引入了独立的 LoggedModel 概念：模型有自己的 ID（`model_id`），可以在 UI 的 `Models` 标签下直接列出，不依赖于 Run 是否还在。Run 和 Model 之间的关联变成可选的（`source_run_id` 字段）。

这个变化带来了三个好处：

1. **删除 Run 不会误删模型**：Run 可以清理，但 Registered Model 保留
2. **跨 Run 引用更直接**：`logged_model_id` 就是模型的全局唯一标识
3. **统一接口**：所有模型（Run 里的、Registry 里的、LoggedModel）都有相同的访问方式

### 3.7 ⭐ Alias 为什么取代了 Stage（新手最困惑的点）

**先说结论：MLflow 2 时代的 Stage（`None` / `Staging` / `Production` / `Archived`）在 MLflow 3 里已经废弃，取而代之的是 Alias。** 如果你在网上搜到 `transition_model_version_stage(...)` 的教程，那是旧写法，别学。

**Stage 的三个硬伤：**

1. **写死的四个值，改不了**。现实里团队的流程五花八门：有人要 `dev` / `qa` / `canary` / `prod` 四级，有人做 A/B 测试要同时上两个模型，有人还要区分"华东区在用"和"华南区在用"。Stage 只给你四个固定选项，全都塞不下。Alias 是**自定义字符串**，你想叫什么叫什么。

2. **一个 stage 只能挂一个版本，一个版本只能有一个 stage**。这个 1 对 1 的死限制让 A/B 测试特别难做——你没法说"v2 和 v3 同时是生产模型"。Alias 是**多对多**的：一个版本可以同时挂 `champion` 和 `stable`，你也可以再加 `challenger` 挂到 v3 上做灰度。

3. **语义模糊，容易误会**。"Production" 到底是"正在生产环境跑"还是"通过了测试可以上生产"？不同团队理解不一样，还得靠口头约定。Alias 强迫你自己命名，反而更明确。

**Alias 的核心好处——热切换（这是最实用的部分）：**

生产服务里你的加载代码写死一行：

```python
model = mlflow.sklearn.load_model("models:/WineQualityClassifier@champion")
```

新模型 v2 上线时，你**不改代码、不重启服务**，只要执行：

```python
client.set_registered_model_alias("WineQualityClassifier", "champion", version=2)
```

下一次加载就自动是 v2 了。要回滚？把别名指回 v1，一秒钟的事。这个切换是**原子操作**，不存在"改到一半"的中间状态。

常用的别名约定（社区惯例，非强制）：

| 别名 | 含义 |
|------|------|
| `champion` | 当前生产在用的冠军模型 |
| `challenger` | 正在评测、准备挑战冠军的候选 |
| `baseline` | 用于对比的基准模型 |
| `archived` | 已下线但保留，方便回溯 |

### 3.8 MLflow 3 破坏性变化速查

这一章踩到的两个新写法，都是 MLflow 3 强制要求的：

| MLflow 2 写法 | MLflow 3 写法 | 影响 |
|---------------|---------------|------|
| `log_model(model, artifact_path="xxx")` | `log_model(model, name="xxx")` | ⚠️ 旧写法触发 deprecation 警告，部分版本直接报错 |
| `transition_model_version_stage(name, v, "Production")` | `set_registered_model_alias(name, "champion", v)` | ⚠️ Stage 整套 API 废弃 |

本章脚本 `02a_log_model.py` 里已经用了 `name="wine-classifier"`，请仔细对照 Chapter 2 的 `artifact_path="model"`。

## 实战步骤：按顺序照做

### Step 1：环境准备（确认 SQLite 后端可用）

```bash
cd <project-root> && conda activate mlflow && pip install mlflow scikit-learn pandas
```

**关键前置**：Model Registry 必须有数据库后端。`02a_log_model.py` 第一行就是：

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
```

纯文件系统（`file:./mlruns`）**不支持** Registry。如果你跑 02b 时报 `RESOURCE_DOES_NOT_EXIST` 或 `registry` 相关的错，多半是 backend 没设对。

### Step 2：训练并记录带签名的模型

```bash
cd <project-root> && python 02_registry/02a_log_model.py
```

预期输出里有 `模型性能: accuracy=1.0000, f1=1.0000`（Wine 数据集很简单，满分正常）和一行 `模型 URI: runs:/<run_id>/wine-classifier`。

这个脚本做了 5 件事：

1. 训练一个 `StandardScaler + RandomForest` 的 Pipeline
2. 推断签名（输入 13 列 double，输出 long）
3. 截取 3 行作为 `input_example`
4. 用 `name="wine-classifier"`（**不是** `artifact_path`）记录模型
5. 打两个 Tag（`pipeline`、`dataset`）

### Step 3：注册模型 + 设别名

```bash
cd <project-root> && python 02_registry/02b_register_alias.py
```

预期看到 `✓ 已注册为 WineQualityClassifier v1` 和 `✓ 已设置 champion alias → v1`，最后打印出版本列表和别名映射。

这个脚本做了 5 件事：

1. 用 `search_runs` 找最近一次 Run（必须是 02a 跑过的）
2. `mlflow.register_model()` 把模型从 Run 提升到 Registry（自动成为 v1）
3. `set_registered_model_alias("WineQualityClassifier", "champion", version=1)` 设别名
4. `update_model_version()` 补充描述
5. 列出版本表和别名映射

### Step 4：去 UI 验证（别跳过）

```bash
cd <project-root> && mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

打开 `http://localhost:5000`，按下面的顺序点：

1. **左侧导航栏点 `Models`** → 看到 `WineQualityClassifier`
2. **点进去看 Version 1**，重点看三处：
   - **Aliases** 那一栏显示 `champion`
   - **Description** 是 02b 里 `update_model_version` 写进去的那段文字
   - **Source Run** 链接能跳回 02a 的那次 Run
3. **再从 `Experiments` → `02_model_registry` → 打开 Run → `Artifacts` tab**
4. **点开 `wine-classifier` 目录**，**亲眼看一遍 `MLmodel` 文件的内容**，这是理解模型格式最直接的方式
5. 同一个目录下应该还有 `input_example.json`，点开看里面是 3 行带列名的数据

### Step 5：自己下载 model artifact zip 看结构

这一章最重要的"动手做"练习——亲手把模型目录下载下来，用 `unzip` 解压，用文本编辑器看 `MLmodel` YAML。

详见下一节。

## 🛠️ 动手做：下载 model artifact zip 并解读 MLmodel YAML

**任务**：从 UI 下载 02a 产生的 model artifact zip，解压后看 `MLmodel` YAML 的真实内容。

### 操作步骤

1. 启动 UI（如果还没启动）：

```bash
cd <project-root> && mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

2. 浏览器打开 `http://localhost:5000`：
   - 左侧选 `Experiments` → `02_model_registry`
   - 点开 02a 那次 Run（`wine-rf-v1`）
   - 进 `Artifacts` tab
   - 找到 `wine-classifier/` 目录，右上角应该有下载按钮（或者点目录名前的图标）

3. UI 会下载一个 zip 文件，默认名类似 `wine-classifier.zip`。**找到它的位置**（一般在 `~/Downloads/`），用 `cd` 切到那个目录：

```bash
cd ~/Downloads && ls -lh wine-classifier.zip
```

4. **解压并查看结构**：

```bash
cd ~/Downloads && unzip -l wine-classifier.zip
```

你应该看到这些文件：

```
Archive:  wine-classifier.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
        0  ...                MLmodel
     1234  ...                conda.yaml
     2345  ...                model.pkl
      567  ...                python_env.yaml
      234  ...                requirements.txt
      890  ...                input_example.json
        ...
```

5. **解压到本地目录**：

```bash
cd ~/Downloads && unzip wine-classifier.zip -d wine-classifier-unpacked && ls wine-classifier-unpacked/
```

6. **看 MLmodel YAML 的真实内容**（这是这章最重要的一个命令）：

```bash
cd ~/Downloads && cat wine-classifier-unpacked/MLmodel
```

你应该看到一段 YAML，里面包含：

- `artifact_path: wine-classifier`
- `flavors.python_function` 段（loader_module、model_path）
- `flavors.sklearn` 段（sklearn_version、serialization_format）
- `signature` 段（inputs 13 列 double + outputs long）
- `run_id`（指向 02a 的那次 Run）
- `mlflow_version: 3.x`

7. **再看 input_example.json**：

```bash
cd ~/Downloads && cat wine-classifier-unpacked/input_example.json
```

里面是 3 行带列名的真实数据（`alcohol`、`malic_acid` 等 13 个特征），就是 UI 上 Artifacts tab 显示的那个文件的本体。

### 验证标准

- `wine-classifier.zip` 解压后能看到 `MLmodel`、`model.pkl`、`conda.yaml`、`python_env.yaml`、`requirements.txt`、`input_example.json` 这 6 个文件
- `MLmodel` 里 `flavors.sklearn` 段的 `sklearn_version` 是你当前安装的版本（如 `1.5.0`）
- `MLmodel` 里 `signature.inputs` 列出了 13 个特征名（`alcohol`、`malic_acid`、…）
- `input_example.json` 是合法的 JSON，可以用 `python -m json.tool` 验证：

```bash
cd ~/Downloads && python -m json.tool wine-classifier-unpacked/input_example.json > /dev/null && echo "✓ JSON 合法"
```

### 加分项

如果你想更深入，可以打开 `python_env.yaml` 看 MLflow 是怎么描述 Python 环境的；打开 `conda.yaml` 看它用了什么 conda 渠道。这两个文件是部署时的关键——`mlflow models build-docker` 会基于它们构建镜像。

## 避坑清单

- ⚠️ **坑 1：Registry 必须有数据库后端**。用 `file:./mlruns` 时调 `register_model` 会直接报错。本项目统一 `sqlite:///mlflow.db`，生产环境用 PostgreSQL/MySQL。另外注意：脚本是相对路径打开 sqlite，**必须在项目根目录运行**，否则会在别处生成一个空的 `mlflow.db`，然后你会困惑"为什么 UI 里什么都没有"。

- ⚠️ **坑 2：`artifact_path` 已改名为 `name`**。MLflow 3 的 `log_model(model, name="...")`，网上大量旧教程还在用 `artifact_path=`。第 2 章的 `01b_sklearn_basics.py` 还在用 `artifact_path="model"`，属于过渡写法，到本章 `02a_log_model.py` 已经切到 `name="wine-classifier"`。

- ⚠️ **坑 3：Stage 相关 API 全部废弃**。`transition_model_version_stage()`、`stage="Production"` 这类写法不要再用，一律换成 `set_registered_model_alias()`。UI 里也已经看不到 Stage 下拉框了。

- ⚠️ **坑 4：版本号只增不减，删了也不会复用**。删掉 v2 之后，下次注册是 v3 而不是补上 v2。所以版本号可以放心当唯一标识用。

- ⚠️ **坑 5：02b 用 `search_runs` 取"最近一次 Run"，有隐患**。如果你在跑完 02a 之后又在 `02_model_registry` 这个实验里跑了别的 Run，02b 就会注册错的那个。稳妥做法是显式指定 run_id，或者在过滤条件里加上 run_name：

  ```python
  runs = mlflow.search_runs(
      experiment_names=["02_model_registry"],
      filter_string="attributes.run_name = 'wine-rf-v1'",
      order_by=["start_time DESC"], max_results=1,
  )
  ```

- ⚠️ **坑 6：model URI 里的路径名必须和 `log_model` 的 `name` 完全一致**。02a 写的是 `name="wine-classifier"`，02b 就必须拼 `runs:/{run_id}/wine-classifier`。写错一个字符就是 `RESOURCE_DOES_NOT_EXIST`，而错误信息不会告诉你"你是不是拼错了名字"。

- ⚠️ **坑 7：别名区分大小写，且不能用作纯数字**。`Champion` 和 `champion` 是两个不同的别名；别名也不能起成 `1`、`2` 这种，会和版本号语法冲突。

- ⚠️ **坑 8：签名太严格也会咬人**。如果推断签名时用的是 DataFrame（有列名），那推理时也必须传 DataFrame，传 numpy 数组会因为缺列名而校验失败。保持训练和推理的数据形态一致。

- ⚠️ **坑 9：02b 文件头注释里写的是 `python 03_registry/...`，这是笔误**，正确目录是 `02_registry/`。

## 📖 下一步

到这里你应该理解了 MLflow Model 的自描述目录结构、Signature 的作用、以及 Registry + Alias 的热切换优势。下一章我们要解决"模型怎么上线"——怎么用 `models:/<name>@<alias>` 在 Python 脚本里加载模型、做推理、做 A/B 测试。

请继续阅读 **Chapter 4: 模型加载、推理与上线**。

---

## 📚 深入阅读

- **`notes/01_basics.md`**：Chapter 2 的进阶笔记，覆盖 autolog 细节、批量记录、SQLite 锁问题
- **`notes/02_registry.md`**：Chapter 3 的进阶笔记，覆盖三种 model URI 写法、`pyfunc` vs `sklearn` 加载的区别、A/B 测试最佳实践
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

| 脚本 | 作用 | 是否必跑 | 前置 |
|------|------|---------|------|
| `02a_log_model.py` | 训练 sklearn 模型 + 推断签名 + 记到 Run | 必跑（上章） | Phase 1 |
| `02b_register_alias.py` | 把 Run 提升为 Registered Model + 设 champion alias | 必跑 | 跑过 02a |
| `02c_load_predict.py` | 用 `models:/name@champion` 加载并推理 | 必跑 | 跑过 02b |

---

## 核心概念

### 1. Model Registry：模型的"Git 仓库"

Run 里的模型是**实验产物**——你可能跑了 200 次，其中 199 次都是垃圾。Registry 是**发布通道**——你从那 200 次里挑出好的，给它起个正式名字（`WineQualityClassifier`），它就有了 v1、v2、v3 的版本序列。

打个比方：

| 概念 | 类比 | 特点 |
|------|------|------|
| Run 里的模型 | 本地的一次 commit | 数量多，随手产生，用 `runs:/<run_id>/<name>` 引用 |
| Registered Model | 一个 Git 仓库 | 有名字，是一个逻辑上的"产品线" |
| Model Version | 打的 tag（v1、v2） | 注册一次自动 +1，**不可变** |
| Alias | 指向某个 tag 的分支指针 | 可以随时改指向，`champion` → v3 |

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

| Alias | 含义 | 跑在线上吗 |
|-------|------|----------|
| `champion` | 当前生产在用的冠军模型 | 是 |
| `challenger` | 正在评测、准备替换 champion 的候选 | 灰度流量或离线评估 |
| `baseline` | 用于对比的基准模型 | 否 |
| `archived` | 已下线但保留，方便回溯 | 否 |

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
cd <project-root> && python 02_registry/02a_log_model.py
```

预期输出里有 `模型性能: accuracy=1.0000, f1=1.0000`（Wine 数据集很简单，满分正常），还会打印一行 `模型 URI: runs:/<run_id>/wine-classifier`，**把这个 run_id 记下来**。

> ⚠️ **必须确认数据库后端**：02a 里有 `mlflow.set_tracking_uri("sqlite:///mlflow.db")` 这一行。如果你看到 `mlruns/` 目录被创建而不是 `mlflow.db`，说明没连上数据库，**Registry 一会儿会报错**。

### Step 2 — 注册 + 设别名

```bash
cd <project-root> && python 02_registry/02b_register_alias.py
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
cd <project-root> && python 02_registry/02c_load_predict.py
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
cd <project-root> && sed -i 's/n_estimators=200/n_estimators=50/' 02_registry/02a_log_model.py
cd <project-root> && python 02_registry/02a_log_model.py
cd <project-root> && python 02_registry/02b_register_alias.py
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
cd <project-root> && python 02_registry/02c_load_predict.py
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

- ⚠️ **坑 1：Registry 必须有数据库后端**。用 `file:./mlruns` 时调 `register_model` 会直接报错。本项目统一 `sqlite:///mlflow.db`。**脚本是相对路径打开 sqlite，必须在项目根目录运行**，否则会在别处生成一个空的 `mlflow.db`，然后你会困惑"为什么 UI 里什么都没有"。

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

- ⚠️ **坑 8：02b/02c 文件头注释里写的是 `python 03_registry/...`，这是笔误**，正确目录是 `02_registry/`。脚本本身能跑，但别照抄错路径。

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

| 概念 | 类比 | 存什么 | 选型 |
|------|------|--------|------|
| **Backend Store** | 餐厅的点菜单（数据库） | experiments、runs、metrics、params、tags、aliases | SQLite（小团队）、PostgreSQL（生产） |
| **Artifact Store** | 餐厅的仓库（文件系统） | 模型文件、图片、配置、特征文件 | 本地路径（学习）、S3/MinIO（生产） |

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

或者 `RESOURCE_DOES_NOT_EXIST` 类似的错误。本项目从第 4 章开始统一用 `sqlite:///mlflow.db`，所有脚本第一条都是 `mlflow.set_tracking_uri("sqlite:///mlflow.db")`。**这一行不是装饰，是功能要求。**

### 4. 数据集血缘：给训练数据发"身份证"

每次训练时，不光要记录参数和指标，还要告诉 MLflow "我用了这份数据"。`mlflow.data.from_pandas(df, source=..., name=..., targets=...)` 创建一个 `Dataset` 对象，里面带四样东西：

| 字段 | 含义 | 举例 |
|------|------|------|
| **source** | 数据来自哪个文件/URL/库 | `"data/wine.csv"`、`"sklearn.datasets.load_wine"` |
| **name** | 你给数据集起的名 | `"wine_dataset"` |
| **digest** | 数据集内容的哈希（指纹） | `"a3f5..."` |
| **schema** | 列名 + 类型 | `[{"name": "alcohol", "type": "double"}, ...]` |

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

| 维度 | MLflow 2 | MLflow 3 |
|------|----------|----------|
| 模型 URI | `runs:/<id>/<path>` | 加 `models:/<model_id>` 跨实验引用 |
| `log_model` 参数 | `artifact_path="..."` | `name="..."`（强制改名） |
| 阶段切换 | `transition_model_version_stage` | `set_registered_model_alias` |
| 搜索模型 | 只能用 `search_runs` | 新增 `search_logged_models` |
| UI 入口 | 模型藏在 Run 里 | 左侧栏独立 `Logged Models` |
| 服务器 | -- | 3.5+ 必须配 `--allowed-hosts` |

---

## 实战步骤

### Step 1：启动 Tracking Server（新开一个终端）

```bash
conda activate mlflow
cd <project-root> && bash 03_tracking/03a_start_server.sh
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
cd <project-root> && python 03_tracking/03b_dataset_lineage.py
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
cd <project-root> && python 03_tracking/03c_search_logged_models.py
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
cd <project-root> && python 03_tracking/03b_dataset_lineage.py
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
cd <project-root> && python 03_tracking/03b_dataset_lineage.py
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

这一章对应 `05_tracing/` 目录下的 4 个脚本：

| 脚本 | 一句话作用 | 是否必跑 | 前置 |
|------|-----------|---------|------|
| `env_bootstrap.py` | 自动把国内 LLM（DeepSeek 等）的 key 桥接成 OpenAI 协议 | 必跑（其他脚本都依赖它） | 无 |
| `05a_env_test.py` | 验证 MLflow 能联通 DeepSeek，发一次最简单的请求 | 必跑 | 跑过 env_bootstrap |
| `05b_basic_tracing.py` | `mlflow.openai.autolog()` 实战 + 多轮对话追踪 | 必跑 | 跑过 05a |
| `05c_custom_decorator.py` | `@mlflow.trace` 自定义 Span，搭一个 RAG 链看嵌套 Span 树 | 推荐 | 跑过 05b |
| `05d_metadata_search.py` | 给 trace 打 user/session + `search_traces` 查询实战 | 推荐 | 跑过 05b |

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

| SpanType | 含义 | 用在哪儿 |
|----------|------|----------|
| `LLM` | 大模型调用 | 包装调大模型的函数 |
| `RETRIEVER` | 检索 | 包装向量查询、文档检索 |
| `TOOL` | 工具调用 | 包装外部 API / 工具函数 |
| `CHAIN` | 编排链 | 包装整个流程（最常用的"外层"装饰） |
| `AGENT` | Agent 决策 | 包装 Agent 主循环 |
| 任意字符串 | 自定义 | 你自己随便起名，比如 `"PROMPT_TEMPLATE"` |

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

| 查询 | 你应该看到 |
|------|-----------|
| 所有 OK 的 trace | 15 条左右 |
| 特定用户（alice）的 trace | 数量大约是总数的 1/3 |
| 按 session_id 聚合 | 该 session 内所有 trace 数（可能是 5） |
| 按 latency 倒序 | 前 5 个最慢的调用及其所属用户 |

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

但 Chapter 7 不是评估。Chapter 7 是这一份学习作业的**彩蛋**：你可能注意到，整个项目里有个目录叫 `mlflow_skill/`，里面装了一堆 `SKILL.md` 文件——那是给 AI 编程助手（Claude Code、Cursor 等）读的。我们下一章就来学这些 skill 怎么用。

更深入的学习可以看 `notes/05_tracing.md`——本文就是这份笔记的"轻量化版"，那份笔记里有更多边界情况讨论、UI 截图、每个 Span 字段的含义解释。

---

# Chapter 7：Vibecoding 集成——用 AI 编程助手操作 MLflow

> ⏱️ 预计时间：30 分钟
> 🔑 是否需 API Key：否（本章不直接调 LLM）
> 📚 前置知识：Chapter 0-6（了解 MLflow 基本概念 + 跑过至少一个脚本）

## 🎯 这章做什么

你可能已经注意到，项目里有个目录叫 `mlflow_skill/`，里面装了一堆 `SKILL.md` 文件——**那是给 AI 编程助手（Claude Code、Cursor、Copilot 等）读的"指令手册"**。这一章教你：

1. `mlflow_skill` 是什么、里面有什么
2. 在 vibecoding（对话式编程）场景下，怎么让 AI 助手帮你操作 MLflow
3. 遇到不懂的 MLflow 操作时，怎么让 AI 助手去查 skill 再帮你做

> 💡 **核心洞察**：你不必精通 MLflow 的每一个 API。只要 AI 助手能读到 `mlflow_skill/` 里的 SKILL.md，它就会按手册帮你做对——而你要做的只是**学会怎么让它用这些手册**。

### 你会学到什么

- 知道 `mlflow_skill/` 里 12 个 skill 各自管什么
- 知道 AI 助手是怎么用这些 skill 的（读 SKILL.md → 按步骤执行）
- 能自己用一句话让 AI 助手加追踪 / 评估 / 对比 / 部署
- 能验证 AI 助手干的对不对（去 UI 看结果）
- 学会"通用 skill 使用话术"（不依赖任何具体 AI 工具）

### 前置知识

- 已完成 Chapter 0-6
- 有一个 AI 编程助手（Claude Code、Cursor、GitHub Copilot 等，任选）
- 一个已跑通的最小实验（比如 Chapter 2 的 `01b_sklearn_basics.py`）

---

## 一、什么是 mlflow_skill？

`mlflow_skill/` 是一组**给 AI 助手看的 Markdown 指令手册**。每个 skill 是一个目录，里面有一个 `SKILL.md`（手册正文）+ `references/`（深度参考）+ `scripts/`（可执行工具脚本）。

```
mlflow_skill/
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
3. 它打开 `mlflow_skill/classical-ml/SKILL.md` 读步骤
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
> 这个报错了，帮我查 `mlflow_skill/` 里对应的 skill，看正确写法是什么。

或者更具体：
> 我用 `mlflow.sklearn.log_model(model, artifact_path="m")` 报错了。查一下 `mlflow_skill/classical-ml/SKILL.md`，MLflow 3 应该怎么写？

**为什么有效**：skill 里的 SKILL.md 明确写了 MLflow 3 vs 2 的破坏性变化（`artifact_path=` → `name=` 等）。AI 助手读到后就不会再用旧写法。

**通用话术**（任何 AI 工具都能用）：
- "查 `mlflow_skill/` 里有没有相关的 skill"
- "按 `mlflow_skill/classical-ml/SKILL.md` 的步骤做"
- "这个 MLflow API 报错了，帮我看看 skill 里 MLflow 3 的写法"

---

## 四、怎么启用 skill（通用，不依赖具体工具）

不同 AI 助手启用 skill 的方式不同（Claude Code、Cursor、Copilot 等各自有 rules / skills 配置机制，具体去各自文档查）。**核心原则只有一个：让助手能看到 SKILL.md**。

**最简单的方式（所有工具通用，零配置）**：不装任何东西，直接在对话里让 AI 助手读文件：
> 先读 `<project-root>/mlflow_skill/classical-ml/SKILL.md`，然后按里面 Step 1 帮我做。

这样 AI 助手每轮对话都会参考那个手册。想在更长远的会话里也自动生效，就按你所用 AI 工具的 rules / skills 配置机制，把这个目录加进去。

> ⚠️ 别纠结"链入"这个动作本身——它只是让 AI 助手"知道有这个手册"。直接说"读 SKILL.md 再做事"效果一样。

---

## 五、关键 Take-aways

- **`mlflow_skill/` 是给 AI 助手的"指令手册"**，不是给你读的教程（但你读也有帮助）
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
python 06_prompts/06a_register_prompt.py
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
python 06_prompts/06b_alias_lifecycle.py
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

> ⏱️ 预计时间：50 分钟
> 🔑 是否需 API Key：是（内置 scorer 和 judge 都要调 LLM）
> 📚 前置知识：第 8 章（Prompt Registry），第 4 章（Tracing）

## 🎯 这章做什么

写完一个 LLM 应用，怎么知道它「答得好不好」？传统 ML 有 accuracy、RMSE 这些现成指标，但 LLM 的回答是自然语言——既要看「对不对」，也要看「语气是否友好」、「有没有胡编」。这一章教你用 **MLflow GenAI 评估**给 LLM 应用「打分」。

类比：想象你在教一个实习生回答客户问题。
- **内置 Scorer** 就像公司统一出的「评分卡」（正确性、安全性、切题度），谁都能用。
- **`@scorer` 自定义**就像你给这位实习生写的「特殊规矩」（必须加引用、不能超过 100 字）。
- **`make_judge()`** 就像你雇了个资深主管当裁判，用自然语言写评分标准（「语气要友好专业」），让 LLM 来评判 LLM。
- **跨版本对比**就像你改了实习生的培训手册，跑同一份题库，对比「旧版」和「新版」哪个更好。

跑完这章，你会在 MLflow UI 里看到：
- 至少 1 次评估运行（基础评估 + 自定义 scorer）
- 每次评估都有聚合分数和逐行打分
- 跨 prompt 版本的对比结果（production vs staging）

### 你会学到什么

- 用 `mlflow.genai.evaluate()` 跑一次完整评估（题库 + predict_fn + scorers）
- 配置内置 Scorer（Correctness / Safety / RelevanceToQuery）并解决 judge 模型问题
- 写 `@scorer` 装饰器写业务规则
- 用 `make_judge()` 写 LLM-as-judge 主观评分器
- 跨 prompt 版本做 A/B 评估并根据结果做切流决策
- 完成「5 行 eval 数据集 + 1 个自定义 scorer，评估 `@production` prompt」的动手做

---

## 核心概念

### 1. 评估数据的重要性：不能只靠「感觉」

改完 prompt 之后，你是不是会说「感觉好一点了」？这种主观判断在生产里完全不可靠——不同人感觉不同、同一感觉在不同天不同。

MLflow 的评估系统给你一个客观框架：
- **题库**：固定一批问题 + 期望答案（你要让 LLM 答对的题目）
- **考生**：你的 LLM 应用（接收题库的一行，返回答案）
- **评分标准**：一组 scorer，每个 scorer 从一个角度打分
- **聚合指标**：所有题目的平均分（谁都能看懂、能对比）

类比：与其靠老板「感觉这版不错」，不如用一份试卷 + 一张评分卡打分。

### 2. `mlflow.genai.evaluate()` —— GenAI 的「考试系统」

三件套：

```python
result = mlflow.genai.evaluate(
    data=EVAL_DATA,          # 题库：DataFrame，每行有 inputs 和可选 expectations
    predict_fn=predict_fn,   # 考生：你的 LLM 应用
    scorers=[...],           # 评分标准：内置 + 自定义 scorer 列表
)
```

它会自动跑题库、用 `predict_fn` 调 LLM 拿答案、再用每个 scorer 打分，最后产出：
- `result.metrics`：聚合指标 dict（如 `{"correctness/mean": 0.8, "safety/mean": 1.0}`）
- `result.tables["eval_results"]`：逐行结果（每题的输入、输出、每个 scorer 的打分）

### 3. 内置 Scorer：`Correctness` / `Safety` / `RelevanceToQuery`

MLflow 自带一组通用评分器，开箱即用：

| Scorer | 打分维度 | 需要 expectations？ |
|--------|---------|-------------------|
| `Correctness()` | 答案是否正确 | 需要 `expected_response` 或 `expected_facts` |
| `Safety()` | 是否包含不安全内容（暴力、歧视、隐私等） | 不需要 |
| `RelevanceToQuery()` | 回答是否切题 | 不需要 |

完整清单在 `notes/07_evaluation.md` 末尾（`Fluency`、`RetrievalGroundedness`、`ToolCallCorrectness` 等）。

### 4. `@scorer` 装饰器 —— 写业务规则

内置 Scorer 是通用评分卡，但你的业务有「特殊规矩」——「回答必须加引用」、「不能超过 100 字」、「必须提到 MLflow」。这些用 Python 一行搞定：

```python
from mlflow.genai.scorers import scorer

@scorer(name="has_citation")
def has_citation(outputs: str) -> bool:
    """硬性规则：回答必须包含至少一个 [source:xxx] 引用"""
    if not isinstance(outputs, str):
        return False
    return "[source:" in (outputs or "")
```

**参数约定**（必须按这个命名，不能改名）：
- `inputs: dict`：原始输入
- `outputs`：任意（predict_fn 返回值）
- `expectations: dict`：数据集期望列

**返回类型**：`bool` / `float` / `int` / `str`（MLflow 会自动聚合求 mean）

### 5. `make_judge()` —— LLM-as-judge（让 LLM 给 LLM 打分）

有些判断用代码写不出来——「语气是否友好专业」、「回答是否合理但有歧义」、「是否有同理心」。这时候让 LLM 当裁判最自然：

```python
brand_tone_judge = mlflow.genai.make_judge(
    name="brand_tone",
    instructions=(
        "评估 {{ outputs }} 的语气是否符合品牌要求：\n"
        "- 友好但不轻浮\n"
        "- 专业但不冷漠\n"
        "- 简洁但不敷衍\n"
        "打分范围 1-5：1=完全不像品牌，5=非常符合"
    ),
    model="openai:/deepseek-v4-flash",
)
```

**核心思路**：用一个 LLM（judge model）按你写的 rubric 给另一个 LLM 的输出打分。

`instructions` 必须包含至少一个变量（`{{ inputs }}` / `{{ outputs }}` / `{{ trace }}`），否则 MLflow 会报错。

### 6. judge_model 必须显式传（国内服务商坑）

内置 Scorer 和 `make_judge` 本质都是让 **另一个 LLM** 来评分——这个「裁判 LLM」就叫 **judge model**。

**坑**：MLflow 默认的 judge model 是 `gpt-4.1-mini`（OpenAI 直连）。如果你的 `OPENAI_API_BASE` 指向的是国内代理（DeepSeek、月之暗面等），**它们根本不认识 `gpt-4.1-mini`**，会报 404 或模型不存在错误。

**解决**：显式传 `model=` 参数：

```python
# URI 格式必须是 <provider>:/<model-name>
judge_model = f"openai:/{os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-flash')}"

scorers = [
    Correctness(model=judge_model),
    Safety(model=judge_model),
    RelevanceToQuery(model=judge_model),
]
```

`openai:/` 表示「用 OpenAI 协议」（兼容 OpenAI API 的服务都能用，包括 DeepSeek、月之暗面、智谱等国内服务商）。

### 7. `expected_response` vs `expected_facts` 不能同时给

`Correctness` 这个 scorer 想知道「正确答案长什么样」，有两种方式告诉它：

- `expected_response`：完整参考答案字符串（`"MLflow 是开源的 ML 生命周期管理平台"`）
- `expected_facts`：关键事实列表（`["MLflow 是开源的", "ML 生命周期管理"]`）

**重要**：**两者只能选一个，不能同时给**。如果同时给，MLflow 会触发额外的 judge 调用（用默认的 `gpt-4.1-mini`），国内服务商又会挂掉。

### 8. 跨 prompt 版本对比的思路

把「新旧 prompt」当成两个「考生 A、B」，跑同一份题库，比较谁分高：

```
production (v2) ─┐
                 ├─→ 同一份题库 + 同一组 scorer → 看 metrics 差异
staging   (v3) ─┘
```

决策流程：
- 分高的胜出 → 用 `set_prompt_alias` 把胜出版本设为新的 production
- 输的版本保留在 Registry 方便回滚
- 在 UI 里勾选两个 Run → Compare 看指标差异

---

## 实战步骤

### Step 0：环境准备

```bash
conda activate mlflow
# 确保 .env 里有 OPENAI_API_KEY / OPENAI_API_BASE / DEEPSEEK_MODEL
# 启动 MLflow
mlflow ui --port 5000   # 另开终端
```

### Step 1：跑基础评估（`07a_basic_evaluate.py`）

```bash
cd <project-root>
python 07_evaluation/07a_basic_evaluate.py
```

脚本做的事：
1. 构造一个 5 行的 `EVAL_DATA`（5 个问题 + 期望答案）
2. 定义 `predict_fn(question)` 用 OpenAI 客户端调 LLM
3. 显式构造 `judge_model = f"openai:/{DEEPSEEK_MODEL}"`
4. 用 3 个内置 Scorer（Correctness / Safety / RelevanceToQuery）跑评估
5. 打印聚合指标和逐行结果

预期输出（节选）：
```
judge_model = openai:/deepseek-v4-flash
🔍 用内置 scorers 评估...
评估完成:
  行数: 5

聚合指标:
  correctness/mean: 0.800
  safety/mean: 1.000
  relevance_to_query/mean: 1.000
```

### Step 2：UI 看逐行打分

浏览器打开 `http://localhost:5000`：

1. 选 experiment `07_evaluate`
2. 点开最新的 Run
3. 看 **Metrics** 标签里的聚合分数
4. 看 **Evaluation** 标签（如果有）或 **Artifacts/eval/** 里的明细 JSON——每行能看到：问题、LLM 答案、每个 scorer 的打分 + reasoning

### Step 3：跑自定义 Scorer（`07b_custom_scorer.py`）

```bash
python 07_evaluation/07b_custom_scorer.py
```

脚本组合了 6 个 scorer：
- `Correctness()`、`Safety()`（内置）
- `has_citation`、`is_concise`、`mentions_mlflow`（@scorer）
- `brand_tone`（make_judge）

预期输出（节选）：
```
聚合指标:
  correctness/mean: ...
  safety/mean: ...
  has_citation/mean: ...
  is_concise/mean: ...
  mentions_mlflow/mean: ...
  brand_tone/mean: ...
```

### Step 4：跑跨 prompt 版本对比（`07c_prompt_comparison.py`）

**前置**：确保你在第 8 章注册过 `customer-support-qa` 这个 prompt，并且有 `production` 和 `staging` 两个 alias。

```bash
python 07_evaluation/07c_prompt_comparison.py
```

脚本会做：
1. 用 `prompts:/customer-support-qa@production` 跑一遍评估（run_name=`production`）
2. 用 `prompts:/customer-support-qa@staging` 跑一遍评估（run_name=`staging`）
3. 终端打印对比表 + 决策建议
4. 在 UI 里勾选 `production` 和 `staging` 两个 Run → 点 Compare 看差异

预期输出（节选）：
```
📊 对比结果：
指标                            production      staging        赢家
----------------------------------------------------------------------
correctness/mean                0.800           0.900          staging
safety/mean                     1.000           1.000          production
response_length_ok/mean         1.000           0.800          production

🎯 决策建议：
  ✅ staging 比 production 高 12.5%，建议切到 staging
```

### Step 5：根据对比结果决策

切流代码（一行搞定）：

```python
# 假设 staging 的版本号是 3
mlflow.genai.set_prompt_alias(
    "customer-support-qa",
    alias="production",
    version=3,
)
```

切完所有通过 `prompts:/customer-support-qa@production` 加载 prompt 的应用，下次启动就用新版本。

---

## 🛠️ 动手做：构建 5 行 eval 数据集 + 自定义 scorer，评估 `@production` 提示词

任务：写一个新脚本，构造自己的 5 行评估数据集 + 1 个自定义 scorer，用 `prompts:/customer-support-qa@production` 这个 prompt 跑一次评估。

**步骤**：

1. 写一个新脚本 `/tmp/my_eval.py`：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("<project-root>/04_tracing")))
import env_bootstrap

import mlflow
import os
import re
import pandas as pd
from openai import OpenAI
from mlflow.genai.scorers import scorer, Correctness, Safety


# ============ 1. 自定义 scorer：必须提到 MLflow ============
@scorer(name="mentions_mlflow")
def mentions_mlflow(outputs) -> float:
    """回答提到 MLflow 的次数（截断到 5）"""
    if not isinstance(outputs, str):
        return 0.0
    count = len(re.findall(r"(?i)mlflow", outputs))
    return min(float(count), 5.0)


# ============ 2. 5 行评估数据集 ============
EVAL_DATA = pd.DataFrame([
    {
        "inputs": {"question": "MLflow Tracking 干什么的？"},
        "expectations": {"expected_response": "记录实验参数、指标、artifact"},
    },
    {
        "inputs": {"question": "Prompt Registry 怎么用？"},
        "expectations": {"expected_response": "register_prompt + set_prompt_alias"},
    },
    {
        "inputs": {"question": "MLflow Model Registry 和 Prompt Registry 的区别？"},
        "expectations": {"expected_response": "前者管模型，后者管提示词"},
    },
    {
        "inputs": {"question": "GenAI evaluate() 需要什么？"},
        "expectations": {"expected_response": "data + predict_fn + scorers"},
    },
    {
        "inputs": {"question": "make_judge 是干什么的？"},
        "expectations": {"expected_response": "LLM-as-judge，让 LLM 当裁判打分"},
    },
])


# ============ 3. predict_fn：用 production 提示词 ============
def predict_fn(question: str) -> str:
    # 加载 production prompt
    prompt_obj = mlflow.genai.load_prompt("prompts:/customer-support-qa@production")

    # 渲染（自动判断文本格式还是 chat 格式）
    variables = prompt_obj.variables or set()
    fmt_kwargs = {"question": question}
    for v in variables:
        if v not in fmt_kwargs:
            fmt_kwargs[v] = "(默认)"

    if prompt_obj.is_text_prompt:
        prompt_text = prompt_obj.format(**fmt_kwargs)
        messages = [{"role": "user", "content": prompt_text}]
    else:
        messages = prompt_obj.format(**fmt_kwargs)

    # 调 LLM
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
    )
    resp = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        messages=messages,
        max_tokens=200,
        temperature=0.3,
    )
    return resp.choices[0].message.content


# ============ 4. 跑评估 ============
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("09_my_eval_demo")
mlflow.openai.autolog()

judge = f"openai:/{os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-flash')}"
print(f"judge_model = {judge}")

result = mlflow.genai.evaluate(
    data=EVAL_DATA,
    predict_fn=predict_fn,
    scorers=[
        Correctness(model=judge),
        Safety(model=judge),
        mentions_mlflow,
    ],
)

print("\n聚合指标:")
for metric, value in result.metrics.items():
    print(f"  {metric}: {value:.3f}")
```

2. 跑这个脚本：

```bash
python /tmp/my_eval.py
```

3. 去 UI 看 `09_my_eval_demo` 这个 experiment：
   - 看 Metrics 标签下的 `correctness/mean`、`safety/mean`、`mentions_mlflow/mean`
   - 看逐行打分（Evaluation 标签或 Artifacts/eval/）

4. **加分项**：把 `predict_fn` 改成加载 `@staging` 别名，再跑一次，然后在 UI 里 Compare 两个 Run，看哪个 prompt 的分数更高。

---

## 避坑清单

### 坑 1：内置 Scorer 默认用 `gpt-4.1-mini`（国内服务商不支持）

**症状**：跑 `Correctness()` 时报 `Model gpt-4.1-mini not found` 或 `404`。

**解决**：显式传 `model=f"openai:/{你的模型名}"`，让 MLflow 通过你的 `base_url` 调 judge 模型。

```python
# ✅ 正确
Correctness(model=f"openai:/{os.getenv('DEEPSEEK_MODEL', 'deepseek-v4-flash')}")

# ❌ 错误（用默认值，国内服务商挂）
Correctness()
```

### 坑 2：`expected_response` 和 `expected_facts` 不能同时给

**症状**：同时给两个字段时，会触发额外的 judge 调用（用默认 `gpt-4.1-mini`），然后报模型不存在。

**解决**：二选一。

```python
# ✅ 二选一
{"expectations": {"expected_response": "..."}}
{"expectations": {"expected_facts": ["事实1", "事实2"]}}

# ❌ 两个都给
{"expectations": {"expected_response": "...", "expected_facts": [...]}}
```

### 坑 3：`model=` 参数不能传 None 或空字符串

**症状**：传 `model=None` 会报错或悄悄退回到默认值（然后挂）。

**解决**：用 `os.getenv(..., 'deepseek-v4-flash')` 确保有非空默认值。

### 坑 4：`predict_fn` 签名不对

**症状**：跑评估时第一行就报错 `predict_fn() takes 0 positional arguments`。

**解决**：`predict_fn` 必须接收至少一个参数，参数名要与 `inputs` 里的 key 对应。

```python
# ✅ 正确：参数名 "question" 与 inputs 的 key 对应
data = [{"inputs": {"question": "..."}}]
predict_fn = lambda question: answer(question)

# ✅ 也正确（用 "row" 包一层）
data = [{"inputs": {"row": {"question": "..."}}}]
predict_fn = lambda row: answer(row["question"])

# ❌ 错误：不接收参数
predict_fn = lambda: "..."
```

### 坑 5：`make_judge` 的 instructions 忘了写变量

**症状**：`make_judge` 报 `instructions must contain at least one variable`。

**解决**：在 instructions 里至少包含 `{{ inputs }}`、`{{ outputs }}`、`{{ trace }}` 其中的一个。

```python
# ✅ 正确
instructions="评估 {{ outputs }} 的语气是否友好专业"

# ❌ 错误（没变量）
instructions="评估回答的语气是否友好专业"
```

### 坑 6：`@scorer` 函数参数名写错

**症状**：MLflow 找不到输入数据，或者 scorer 永远返回固定值。

**解决**：参数名必须是 `inputs`、`outputs`、`expectations` 这三个（不能拼错、不能改名）。

```python
# ✅ 正确
@scorer
def my_scorer(outputs: str) -> bool:
    return len(outputs) > 0

# ❌ 错误
@scorer
def my_scorer(output: str) -> bool:  # "output" 不是约定名
    return len(output) > 0
```

### 坑 7：跨版本对比的题库要固定

**症状**：换题库对比没有意义（A 答对难的题、B 答对简单的题，无法判断哪个 prompt 更好）。

**解决**：跨版本对比必须用**同一份题库**、**同一组 scorer**——只换 `predict_fn` 里的 prompt 版本。

### 坑 8：评估数据太少（5 行不够）

**症状**：评估结果波动大，今天高分明天低分，根本看不出真实差异。

**解决**：至少 30-50 行评估数据，覆盖你关心的典型场景。如果数据实在凑不齐，先用 5-10 行跑通流程，后续慢慢补。

---

## 📖 下一步

你已经学会了：
- 用 `mlflow.genai.evaluate()` 给 LLM 应用打分
- 用内置 Scorer / `@scorer` / `make_judge` 三种评分器组合
- 跨 prompt 版本做 A/B 评估并决策

至此你掌握了 MLflow 的「评估 → 决策 → 切流」完整闭环。下一阶段（Chapter 10+）会进入**部署与服务化**：把评估胜出的 prompt + 模型打包成 REST API、用 Docker 部署、监控线上数据漂移。

详细学习笔记见：`notes/07_evaluation.md`。
# Chapter 10：模型评估与本地部署服务

> ⏱️ 预计时间：60 分钟
> 🔑 是否需 API Key：否
> 📚 前置知识：第 6 章（注册模型 + alias）、第 7 章（trace）

## 🎯 这章做什么

模型训完、注册到 Registry 之后，真正的工作才刚开始：你要回答两个问题——

1. **这个模型到底好不好？** 不是"accuracy 0.97"那种一句敷衍，而是"哪一类 0.97、哪些图能看出来"。
2. **怎么让别人用它？** 不是"我发你 pickle 文件"，而是"你给我发个 HTTP 请求，我把预测结果返回给你"。

这一章解决这两件事：先用 `mlflow.models.evaluate()` 一键算全套指标 + 自动出图，再用 `mlflow models serve` 把模型起成 REST API，最后用 `curl` 调它。

> 🍳 **类比**：训练模型就像做菜——厨师（你）做完一道菜，要做两件事：
> - **试吃**（评估）：找几个食客打分（accuracy、F1）、看摆盘（混淆矩阵）、看味道曲线（ROC）。MLflow 的 `evaluate` 就是帮你搞了个自动试吃团。
> - **上桌**（部署）：把菜放进窗口让客人点餐（`models serve` 起一个 HTTP 服务）。

### 你会学到什么

- 用 `mlflow.models.evaluate()` 一行代码算全套分类/回归指标 + 自动出混淆矩阵、ROC、PR 曲线
- 用 `make_metric()` 写自定义业务指标（例如"高价值客户加权 accuracy"）
- 用 `validate_evaluation_results()` 对比新模型 vs baseline（MLflow 3 新 API）
- 知道 `extra_metrics=` 而不是 `custom_metrics=`（MLflow 3 改名了）
- 用 `mlflow models serve -m models:/<name>@champion -p 5001` 把模型起成 REST API
- 用 `curl` 推 JSON / CSV 到 `/invocations` 做推理

### 对应脚本清单

| 脚本 | 一句话作用 | 是否必跑 | 前置 |
|------|-----------|---------|------|
| `04a_evaluate_basics.py` | 跑一次 `mlflow.models.evaluate`，自动算内置指标 + 生成混淆矩阵/ROC 图 | ✓ 必跑 | 第 6 章 |
| `04b_evaluate_custom.py` | 写一个"高价值客户加权"自定义指标，对比 RF 和 LR，验证 B 是否比 A 好 | ✓ 必跑 | 跑过 04a |
| `04c_models_serve.sh` | 用 `mlflow models serve` 起本地 REST API，用 `curl` 调 `/invocations` 推理 | ✓ 必跑 | 跑过 04b + 注册过模型 |

### 前置知识

- 已完成第 6 章，会用 `mlflow.start_run()`、`mlflow.sklearn.log_model()`
- 至少注册过一个模型到 Model Registry，并且有 `@champion` 别名
- MLflow Tracking Server 已在 `sqlite:///mlflow.db` + `./mlruns` 跑起来（04a/04b 用本地文件模式也行；04c **必须** server 模式）

### 跑完必看（UI）

1. 启动 UI：`mlflow ui --port 5000`
2. 选 experiment `04_evaluate`
3. 点开 Run `evaluate-baseline`
4. 看：
   - **Metrics 标签**：`accuracy_score`、`f1_score`、`roc_auc` 等内置指标一行行排好
   - **Artifacts → eval/**：自动生成的 `confusion_matrix.png`、`roc_curve_plot.png`、`precision_recall_curve_plot.png`
   - **Artifacts → model/**：模型本身 + `signature` + `requirements.txt`

---

## 核心概念

### 1. `mlflow.models.evaluate()` 是干嘛的

想想你以前怎么"评估一个 sklearn 模型"：

```python
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average="macro"))
# 还得 import matplotlib, 手写混淆矩阵代码...
# 想跑回归？再 import 一堆 metrics...
```

`mlflow.models.evaluate()` 是 **"全自动评估 + 自动作图 + 自动写回 MLflow"** 的整合入口。你只需要：

- 指一个模型（`runs:/xxx/model` 或 `models:/MyModel@champion`）
- 给一份含 label 列的数据
- 说清楚是分类还是回归（`model_type="classifier"`）

它就帮你：

- 算所有内置指标（accuracy / F1 / precision / recall / ROC-AUC / log_loss...）
- 对分类自动生成：混淆矩阵图、ROC 曲线、PR 曲线、校准曲线
- 对回归自动生成：残差图、预测 vs 真实散点图
- 把所有指标和图都写回当前 Run

> 💡 **这意味着**：在 MLflow UI 里你点开一个 Run，**所有评估结果在同一个页面**，不用跳到别处找图。

### 2. 自定义指标 `make_metric`

内置指标对简单问题够用，但业务上经常要更"狡猾"的指标，例如：

- "class_0 是高价值客户，识别错了代价 5 倍" → **加权 accuracy**
- "假阳性罚 10 元，假阴性罚 100 元" → **业务损失函数**
- "top-5 推荐命中率" → **业务命中率**

`make_metric` 就是让你把这种"我的业务特殊规则"塞进去，包装成一个 MLflow 指标。它会和内置指标一起出现在 UI 的 Metrics 标签里。

### 3. `validate_evaluation_results`：MLflow 3 的"模型升级门槛"

业务里上线的真实流程：

1. 旧模型 A 在跑
2. 你训了新模型 B
3. 关键是：**B 必须比 A 好多少才允许替换？**

MLflow 2 时代这套规则藏在 `mlflow.evaluate(baseline_model=...)` 一个超长参数里，很难复用、很难调试。
MLflow 3 拆成了两步：

- 先分别对 A、B 各 `evaluate()` 一次（拿到两个 `EvaluationResult` 对象）
- 再用 `validate_evaluation_results(candidate=B, baseline=A, thresholds={...})` 验证 B 是否达标

好处：**阈值规则可以独立写、独立复用、独立测试**，还能塞进 CI。

### 4. `mlflow models serve`：把模型变成 HTTP 服务

训练完的模型本质是个文件，**别人没法直接用**（除非你把 sklearn + pickle 文件传给他）。`mlflow models serve` 把模型起成一个标准 REST API，路径统一是 `/invocations`，接受 JSON 或 CSV——这样前端、后端、别的服务都能用 `curl` 调。

它内部会装好这个模型需要的 Python 环境（用 model signature + requirements 推断），你不用管 conda。

---

## 实战步骤

### Step 1：跑 `04a_evaluate_basics.py`

```bash
cd <project-root>
python 04_evaluate/04a_evaluate_basics.py
```

你会看到：训练完一个 RandomForest，对它跑一遍 `mlflow.models.evaluate`，打印一坨指标和图的名字。

打开 MLflow UI → experiment `04_evaluate` → Run `evaluate-baseline`：

- **Metrics 标签**：看 `accuracy_score`、`f1_score`、`roc_auc` 等
- **Artifacts → eval/**：能看到 `confusion_matrix.png`、`roc_curve_plot.png` 等自动生成的图

> ✨ 这一步完成时，你应该理解：**模型评估这件事，UI 里点开 Run 就能看到所有结果——不用自己 plot。**

### Step 2：跑 `04b_evaluate_custom.py`

```bash
python 04_evaluate/04b_evaluate_custom.py
```

这一脚本做了 6 件事：

1. 训练 RF（A）和 LR（B）两个模型
2. 写一个"高价值客户（class_0）加权 5 倍"的自定义 `weighted_accuracy_v1`
3. 对 A、B 各跑一次 `evaluate()`，塞入 `extra_metrics=[custom_metric]`
4. 用 `validate_evaluation_results` 比 B 是否比 A 好
5. 打印 MLflow 2 vs 3 的 API 差异
6. 打印"通过/不通过"结果

**重点观察**：

- 两个 Run 的 Metrics 标签里都多了一行 `weighted_accuracy_v1`
- 脚本末尾会打印"✓ 通过"或"✗ 不通过"，看看 B 是不是真的比 A 好（业务场景下 LR 在小数据上可能不如 RF）

### Step 3：跑 `04c_models_serve.sh`

这个是 shell 脚本，需要**多个终端**：

**终端 A**（启动 MLflow server）：

```bash
bash 04_evaluate/04c_models_serve.sh
# 复制里面"终端 A"那段命令
mlflow server \
  --backend-store-uri sqlite:///$(pwd)/mlflow.db \
  --default-artifact-root $(pwd)/mlruns \
  --host 0.0.0.0 \
  --port 5000
```

> ⚠️ **必须用 server 模式**：纯本地文件模式不支持 Model Registry，会报 `No such registered model`。

**终端 B**（确保有 champion 模型，先跑过第 6 章的注册脚本）：

```bash
python 03_registry/02a_log_model.py
python 03_registry/02b_register_alias.py
```

**终端 C**（起模型服务）：

```bash
mlflow models serve \
  -m "models:/WineQualityClassifier@champion" \
  -p 5001
# 第一次启动会 pip install,等几十秒到一分钟
# 看到 "Listening on http://127.0.0.1:5001" 就 ok
```

**终端 D**（curl 推理）：

```bash
# JSON 格式（推荐）
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  --data '{
    "dataframe_records": [
      {"alcohol": 13.0, "malic_acid": 1.5, "ash": 2.5, "alcalinity_of_ash": 19.0,
       "magnesium": 100, "total_phenols": 2.8, "flavanoids": 3.0,
       "nonflavanoid_phenols": 0.3, "proanthocyanins": 1.8, "color_intensity": 5.0,
       "hue": 1.0, "od280/od315_of_diluted_wines": 3.0, "proline": 1000}
    ]
  }'
```

返回：

```json
{"predictions": [0]}
```

> ✨ 这一步完成时，你已经在用生产级的方式（HTTP + JSON）调用模型了——和 `requests.post(url, json=...)` 完全一样。

### Step 4：（可选）容器化部署

```bash
# 把模型打成 Docker 镜像
mlflow models build-docker -m "models:/WineQualityClassifier@champion" -n wine-classifier

# 跑容器（容器内 8080 端口对应外部 5001）
docker run -p 5001:8080 wine-classifier
```

> 这个就是你写论文/做 demo 时给评审看"我真的部署了一个 ML 服务"的证据。

---

## 代码模式（可复用模板）

### 模式 1：内置评估模板

```python
import mlflow

with mlflow.start_run(run_name="evaluate-baseline") as run:
    # 先把模型 log 进去（evaluate 要 model_uri）
    mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(3))
    model_uri = f"runs:/{run.info.run_id}/model"

    # 一行评估
    result = mlflow.models.evaluate(
        model=model_uri,
        data=eval_df,                       # 必须含 label 列
        targets="target",                   # label 列名
        model_type="classifier",            # 或 "regressor"
        evaluators=["default"],
    )

    # 拿指标
    print(result.metrics)                  # dict: {accuracy_score: 0.97, ...}
    # 拿可视化列表
    print(result.artifacts)                # ['confusion_matrix.png', 'roc_curve_plot.png', ...]
```

**自动产出**（写到 Run 的 `Artifacts/eval/`）：

- `confusion_matrix.png`：分类器最常看的图
- `roc_curve_plot.png`：ROC 曲线 + AUC
- `precision_recall_curve_plot.png`：不平衡数据更该看的图
- `calibration_curve_plot.png`：概率校准
- `per_class_metrics/`：每个类单独的指标 JSON

**内置指标全集**（分类）：`accuracy_score`、`precision_score`、`recall_score`、`f1_score`、`log_loss`、`roc_auc`、`precision_recall_auc`

**内置指标全集**（回归）：`mean_absolute_error`、`mean_squared_error`、`root_mean_squared_error`、`r2_score`、`mean_absolute_percentage_error`

> ⚠️ **MLflow 3 的参数名是 `extra_metrics`，不是 `custom_metrics`**——这是新手最容易翻车的地方，下面避坑清单有专门说明。

### 模式 2：自定义指标模板

```python
from mlflow.metrics import make_metric
import numpy as np

def my_metric_fn(predictions, targets):
    """
    predictions 和 targets 都是 pandas Series,index 一一对齐。
    返回值必须是 float (或可转 float 的标量)。
    """
    preds = np.asarray(predictions)
    targs = np.asarray(targets)
    # 例：按 class_weights 加权的 accuracy
    CLASS_WEIGHTS = {0: 5.0, 1: 1.0, 2: 1.0}
    total = 0.0
    correct = 0.0
    for p, t in zip(preds, targs):
        w = CLASS_WEIGHTS.get(int(t), 1.0)
        total += w
        if p == t:
            correct += w
    return float(correct / total)

custom_metric = make_metric(
    eval_fn=my_metric_fn,
    greater_is_better=True,         # 越大越好；如果是 loss 这种，填 False
    name="weighted_accuracy_v1",    # 在 UI 显示的名字
)

# 塞进 evaluate
result = mlflow.models.evaluate(
    model=model_uri, data=eval_df, targets="target",
    model_type="classifier",
    extra_metrics=[custom_metric],   # ← 注意：MLflow 3 用 extra_metrics
)
# 现在 result.metrics 里会多一项 "weighted_accuracy_v1"
```

**`eval_fn` 签名必须遵守的规矩**：

- 入参：`predictions`（模型预测值 Series）、`targets`（真实标签 Series）
- 返回：标量（float / int / numpy scalar）
- 不要在这里面 print 或写文件——**纯函数**（MLflow 在某些场景会并行调用 `eval_fn`，有副作用会乱序或丢）

### 模式 3：`validate_evaluation_results` 模板（MLflow 3 新写法）

```python
from mlflow.models import MetricThreshold

# 假设 result_a 是旧模型、result_b 是新模型的 EvaluationResult
result_a = mlflow.models.evaluate(model=old_uri,  data=eval_df, targets="target", model_type="classifier", evaluators=["default"])
result_b = mlflow.models.evaluate(model=new_uri,  data=eval_df, targets="target", model_type="classifier", evaluators=["default"])

# 定义 candidate (新模型 B) 必须达到的门槛
thresholds = {
    "accuracy_score": MetricThreshold(
        threshold=0.90,                # 绝对值下限：B 至少要 0.90
        greater_is_better=True,
        # 可选：相对 baseline 的提升要求
        # min_absolute_change=0.02,   # B 比 A 至少高 0.02
        # min_relative_change=0.05,   # 或至少高 5%
    ),
    "f1_score": MetricThreshold(
        threshold=0.85,
        greater_is_better=True,
    ),
}

# 验证
try:
    mlflow.validate_evaluation_results(
        validation_thresholds=thresholds,
        candidate_result=result_b,     # 新模型
        baseline_result=result_a,      # 旧模型（可省略,只验绝对值）
    )
    print("✓ 通过，新模型可以替换")
except Exception as e:
    print(f"✗ 不通过：{e}")
    # MLflow 会抛 MlflowException；你可以决定是否让上线流程中断
```

**MLflow 2 vs 3 对比**：

```python
# MLflow 2（你可能在旧文档里看到）
result = mlflow.evaluate(
    model=new_uri,
    data=eval_df,
    targets="target",
    model_type="classifier",
    baseline_model=old_uri,           # ← 旧写法,所有阈值挤在一个地方
    metric_thresholds=thresholds,
)

# MLflow 3（推荐）
result_a = mlflow.evaluate(model=old_uri, ...)
result_b = mlflow.evaluate(model=new_uri, ...)
mlflow.validate_evaluation_results(   # ← 新写法：拆出来,更清晰可测
    validation_thresholds=thresholds,
    candidate_result=result_b,
    baseline_result=result_a,
)
```

> 💡 **为什么 MLflow 3 要拆开？** 因为 `validate_evaluation_results` 拿到的 `EvaluationResult` 对象本身就是可序列化、可缓存、可存数据库的——你可以在 CI 里把它的 JSON 存下来，下次复用同样的 baseline 做对比。

### 模式 4：`mlflow models serve` 部署模板

```bash
# 终端 A：启 MLflow server（models serve 必须有 server）
mlflow server \
  --backend-store-uri sqlite:///$(pwd)/mlflow.db \
  --default-artifact-root $(pwd)/mlruns \
  --host 0.0.0.0 --port 5000

# 终端 B：部署 champion 模型
mlflow models serve \
  -m "models:/WineQualityClassifier@champion" \
  -p 5001
# 第一次启动会 pip install 模型依赖（conda env，要等几十秒）

# 终端 C：curl 推理（JSON 格式，推荐）
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  --data '{
    "dataframe_records": [
      {"alcohol": 13.0, "malic_acid": 1.5, "ash": 2.5, "alcalinity_of_ash": 19.0,
       "magnesium": 100, "total_phenols": 2.8, "flavanoids": 3.0,
       "nonflavanoid_phenols": 0.3, "proanthocyanins": 1.8, "color_intensity": 5.0,
       "hue": 1.0, "od280/od315_of_diluted_wines": 3.0, "proline": 1000}
    ]
  }'
```

**支持的请求格式**（格式由模型签名自动决定）：

| Content-Type | Body 格式 |
|---|---|
| `application/json` | `{"dataframe_records": [...]}` —— 每条是一个 dict（推荐） |
| `application/json` | `{"dataframe_split": {"columns": [...], "data": [[...]]}}` |
| `text/csv` | 纯 CSV 文本，第一行是列名 |

返回是 JSON：`{"predictions": [...]}` 或带 `{"predictions": [...], "probabilities": [[...]]}`。

---

## 🛠️ 动手做

> **目标**：把"评估 + 部署"两个能力连起来跑一遍：先评估新模型比 baseline 好，再把它 serve 起来，curl 一次得到预测。

**步骤**：

1. 先跑 `04a_evaluate_basics.py`，看到 UI 的混淆矩阵和 ROC 图。
2. 再跑 `04b_evaluate_custom.py`，观察 `weighted_accuracy_v1` 是否被两个 Run 都记录了，并看 `validate_evaluation_results` 是"通过"还是"不通过"。
3. 跑 `04c_models_serve.sh`：
   - 终端 A 启动 `mlflow server`
   - 终端 C 启动 `mlflow models serve -m models:/WineQualityClassifier@champion -p 5001`
   - 终端 D 用 `curl` 推一个 JSON 推理请求，得到 `{"predictions": [0]}`
4. 试着把 curl 的 `Content-Type` 换成 `text/csv`，把 body 换成 CSV 文本，看看是否一样能返回预测。
5. 观察 `mlflow models serve` 第一次启动时打印的 "Installing dependencies..." 日志——它用 `MLmodel` 文件里的 `requirements.txt` 自动装环境。

**预期结果**：

- UI 的 `04_evaluate` experiment 下，Run `evaluate-baseline` 有 `confusion_matrix.png`、`roc_curve_plot.png` 等 artifacts
- UI 的 `04_evaluate_custom` experiment 下，两个 Run 都有 `weighted_accuracy_v1` 这条自定义 metric
- `curl` 返回 `{"predictions": [...]}`，且数值合理

---

## 避坑清单

- ⚠️ **坑 1：把 `custom_metrics` 当参数名（最常见的 API 改名）**

  ```python
  # ❌ 报错：TypeError: got unexpected keyword argument 'custom_metrics'
  result = mlflow.models.evaluate(
      model=..., data=..., targets=..., model_type=...,
      custom_metrics=[custom_metric],
  )

  # ✓ MLflow 3 改名叫 extra_metrics
  result = mlflow.models.evaluate(
      model=..., data=..., targets=..., model_type=...,
      extra_metrics=[custom_metric],
  )
  ```

  **为啥改了**：MLflow 3 把内置 evaluator（`"default"`、`"shap"` 等）和自定义 metric 统一到一个 `extra_*` 命名空间，未来再加 evaluator / metric 不会再撞名。

- ⚠️ **坑 2：`models serve` 用 file store 不行**

  ```bash
  # ❌ mlflow models serve 需要 server 模式
  # ❌ 纯 --backend-store-uri ./mlruns 这种 fs 模式不支持 Model Registry
  # ✓ 必须 sqlite / postgres / mysql
  mlflow server \
    --backend-store-uri sqlite:///$(pwd)/mlflow.db \
    --default-artifact-root $(pwd)/mlruns \
    --host 0.0.0.0 --port 5000
  ```

  错误现象：`No such registered model: WineQualityClassifier`——明明 log 了，但找不到。

  **为啥**：Model Registry 是 server 的功能，纯文件模式不支持 stage/alias/registered model。

- ⚠️ **坑 3：predict 时 JSON 格式写错**

  ```bash
  # ❌ 报错：DataFrame column not found
  curl -X POST http://127.0.0.1:5001/invocations \
    -H "Content-Type: application/json" \
    --data '{"data": [{"alcohol": 13.0, ...}]}'   # 错！应该是 dataframe_records

  # ✓ 必须用 MLflow 约定的两个 key 之一
  --data '{"dataframe_records": [...]}'
  --data '{"dataframe_split": {"columns": [...], "data": [[...]]}}'
  ```

  **为啥**：MLflow 在服务侧会判断 key 名再决定怎么转 pandas。

- ⚠️ **坑 4：`min_absolute_change` 写了负数**

  ```python
  # ❌ 报错或行为反掉
  MetricThreshold(threshold=0.9, min_absolute_change=-0.02)

  # ✓ min_absolute_change 必须是 ≥ 0
  MetricThreshold(threshold=0.9, min_absolute_change=0.02, greater_is_better=True)
  ```

  **为啥**：`min_absolute_change` 的符号语义由 `greater_is_better` 自动决定，写负数会产生"我允许新模型比旧模型差"的诡异效果。

- ⚠️ **坑 5：MLflow 3.5+ 必须配 `--allowed-hosts`**

  ```bash
  # ❌ 浏览器打 mlflow ui 报 "Invalid Host header"
  mlflow ui --port 5000

  # ✓ MLflow 3.5 默认拒绝非 localhost 的 Host header（防 DNS rebinding）
  mlflow server --host 127.0.0.1 --port 5000 --allowed-hosts "*"
  ```

- ⚠️ **坑 6：`make_metric` 的 `eval_fn` 不是纯函数**

  ```python
  # ❌ 报错或结果不稳定
  def my_metric(predictions, targets):
      print(len(predictions))                     # 不允许
      open("/tmp/log.txt", "a").write("hi\n")    # 不允许
      return float((predictions == targets).mean())

  # ✓ 纯函数：只读入参,返回标量
  def my_metric(predictions, targets):
      return float((predictions == targets).mean())
  ```

  **为啥**：MLflow 在某些场景会并行调用 `eval_fn`，副作用会乱序或丢。

---

## 小结：5 个 take-aways

1. **`mlflow.models.evaluate` 是你的"一站式评估员"**：给模型 + 数据 + 类型，它吐一整套指标和图——不用再手写 `matplotlib`。
2. **MLflow 3 的自定义指标参数叫 `extra_metrics`**：`custom_metrics` 是 MLflow 2 的命名，新代码不要用。
3. **自定义指标用 `make_metric(eval_fn=..., greater_is_better=..., name=...)`**：`eval_fn` 必须是纯函数，接收 `predictions`/`targets` Series，return float。
4. **模型对比用 `validate_evaluation_results`（MLflow 3 新写法）**：拆成"对每个模型 evaluate 一次" + "集中验证 candidate vs baseline"，不再用 MLflow 2 的 `baseline_model=` 一锅炖。
5. **`mlflow models serve` 把模型变 REST API**：必须 sqlite/postgres 等数据库后端，配 `models:/Name@alias` 起服务，`/invocations` 收 `dataframe_records` JSON 或 `text/csv`——和 `curl`/`requests.post` 完全一样。

---

## 📖 下一步

下一章（**Chapter 11: LLM Agent 的版本追踪与打包**）会把这一章的"模型版本化"思路扩展到 LLM Agent：怎么让一个 LLM 应用变成可追踪、可对比、可部署的"模型实体"——包括 `LoggedModel`、`set_active_model`、`optimize_prompts`、`ResponsesAgent`、`Models-from-code` 这些 MLflow 3 的新武器。

更深入的内容请参考：

- `notes/04_evaluate.md` —— 评估 + serve 阶段的详细笔记，包括 MLflow 2 vs 3 的完整对比、所有内置指标的清单、`build-docker` 的容器化流程。
- MLflow 官方文档：[Model Evaluation](https://mlflow.org/docs/latest/models.html) 和 [Model Serving](https://mlflow.org/docs/latest/deployment.html)。

---

# Chapter 11：LLM Agent 的版本追踪与打包

> ⏱️ 预计时间：60 分钟
> 🔑 是否需 API Key：**是**（OpenAI 兼容服务，例如 DeepSeek）
> 📚 前置知识：第 7 章（trace）、第 9 章（Prompt Registry）

## 🎯 这章做什么

你已经把 LLM 应用跑起来了（第 7 章），也知道怎么注册 prompt（第 9 章）。现在要把它推向生产，会遇到三个最棘手的问题：

1. **你怎么知道线上跑的是哪个版本的代码？** 同一份代码改了几行 prompt，到底生效的是哪一版？
2. **谁负责把"蹩脚"的 prompt 改得更好？** 你自己手动改，还是让程序自动迭代？
3. **你的 LLM 应用怎么变成一个标准服务？** 怎么让前端用 OpenAI SDK 直接调你的 Agent？

这一章解决的就是这三个问题。我们会学到 MLflow 3 的三件新武器：

- **LoggedModel**：独立的"模型版本"实体，替代了"Run 的附庸"地位
- **`optimize_prompts`**：让 reflection 模型自动改写你的 prompt 并打分
- **ResponsesAgent**：一个跟 OpenAI Responses API 兼容的 Agent 基类，可以打包成模型服务

> 🍵 **类比**：想象你开了家奶茶店。LoggedModel 就像"配方卡片"——"v1 经典版"、"v2 加椰果版"——每改一次配方就登记一张卡片，店里哪天卖出去的奶茶都能溯源到用了哪张卡片。`optimize_prompts` 就像请了个"试喝员 + 配方师"组合，每天试喝新品、给打分、自动帮你改进配方。ResponsesAgent 就像"标准操作手册"——按手册做的奶茶放哪家分店都是同一个味道。

**产出物**：跑完三个脚本后，你在 MLflow UI 里能看到 `agent-v1` 和 `agent-v2` 两个独立的 LoggedModel、能看到 `optimize-demo` prompt 的多版本演进历史、能用 `mlflow.pyfunc.load_model` 加载一个 ResponsesAgent 并推理。

### 你会学到什么

- **能用 `set_active_model` 把同一份代码的不同版本登记为不同的 LoggedModel**，让 trace 自动归类到对应版本
- **能用 `mlflow.genai.optimize_prompts` 让 reflection 模型自动迭代改进 prompt**，每版自动注册到 Prompt Registry
- **能用 `ResponsesAgent` 基类把自定义 LLM 应用打包成 MLflow 模型**，并兼容 OpenAI Responses API
- **能用 Models-from-code 方式（`set_model()` + 文件路径字符串）打包复杂 Agent**，避免 pickle 序列化失败
- **能搜索和对比多个 LoggedModel**（A/B 测试、生产回滚、版本溯源）

### 对应脚本清单

| 脚本 | 一句话作用 | 是否必跑 | 前置 |
|------|-----------|---------|------|
| `08a_active_model.py` | 用 `set_active_model` 把两个 Agent 版本关联到不同的 LoggedModel | ✓ 必跑 | 第 7 章（tracing） |
| `08b_prompt_optimize.py` | 注册初始 prompt 并尝试用 MetaPromptOptimizer 自动改写 | ✓ 必跑 | 第 9 章（Prompt Registry） |
| `08c_responses_agent.py` | 用 Models-from-code 方式打包 ResponsesAgent 到 Registry | ✓ 必跑 | 跑过 08a |
| `simple_qa_agent.py` | `SimpleQAAgent` 的定义（被 08c import，不是独立脚本） | — | — |

### 前置知识

- **已完成第 7 章（tracing）**：知道 `@mlflow.trace`、`mlflow.openai.autolog()` 是什么
- **已完成第 9 章（Prompts）**：知道 `register_prompt` 和 `prompts:/name/version` URI
- **API key**：需要 OpenAI 兼容服务的 API key（脚本读 `OPENAI_API_KEY`、`OPENAI_API_BASE`、`DEEPSEEK_MODEL` 三个环境变量，默认是 DeepSeek）
- **假设你懂**：Python 类继承、dict/对象互转、context manager（`with`）
- **不假设你懂**：MLflow 3 的 LoggedModel 概念、prompt 优化算法细节、OpenAI Responses API 协议

### 跑完必看（UI）

1. 启动 UI：`mlflow ui --port 5000`
2. 左侧菜单选 **Logged Models**：
   - 看到 `agent-v1` 和 `agent-v2` 两个实体
   - 点开任意一个，看 **Traces** 标签：所有用此版本的 trace 都归在这里
   - 顶部 **Compare** 按钮：勾选两个 LoggedModel，对比 trace 数量、延迟
3. 左侧菜单选 **Prompts → optimize-demo**：
   - 看 v1 → v2 → v3… 的版本演进
   - 每版带 commit message 和 template 全文
4. 左侧菜单选 **Experiments → 08_responses_agent**：
   - 点开 Run `agent-packaging`
   - **Artifacts** 标签：能看到 `MLmodel` 文件、`requirements.txt`、agent 的 Python 源码

---

## 核心概念

### 1. LoggedModel —— MLflow 3 的"一等公民"

在 MLflow 2 里，模型只是 Run 的一个 artifact（产出物），附庸在某个 Run 上。这意味着如果你想"看某个模型的所有历史 trace"，会很别扭——得先找 Run，再找 Model artifact，再找关联的 trace。

MLflow 3 把模型拎出来变成**独立实体**。LoggedModel 有自己的 `model_id`（`m-xxx`）、自己的 `aliases`（`@champion`、`@challenger`）、自己的 trace 列表。它可以跨 Run、跨实验存在。

> 📚 **类比**：MLflow 2 像图书馆里"每本书只能在某个书架上"；MLflow 3 像"每本书有自己的 ISBN，可以放到任何书架、被任何人引用"。

**关键 API**：

```python
# 登记/激活一个 LoggedModel（同一 name 多次调用会自动复用）
mlflow.set_active_model(name="agent-v1")

# 之后所有 trace 自动关联到 agent-v1
agent_v1("问题")

# 搜索所有 LoggedModel
mlflow.search_logged_models(experiment_ids=[exp_id])
```

### 2. `set_active_model` —— 隐式的"当前模型指针"

它做的事情很简单：在当前 run 上下文里设一个"当前 LoggedModel"的指针。之后这个 run 里发生的所有 trace 自动打上"我属于这个 LoggedModel"的标签。

**什么时候用**：每个 git commit → 一个 LoggedModel；A/B 测试时把流量分到不同 LoggedModel。

> ⚠️ **必须在 `@mlflow.trace` 装饰的函数里调用，或在 trace 上下文里调用**。否则指针设了但 trace 没认领。

### 3. MetaPromptOptimizer / GepaPromptOptimizer —— 自动改 prompt

你给一个初始 prompt、一批带正确答案的训练数据、一个评分函数。optimizer 会：

1. 让 LLM 用当前 prompt 跑数据 → 得到每个 case 的输出和分数
2. 把"分数低 + 输出"的样本丢给 reflection 模型 → 让它分析"prompt 哪里不好"
3. 根据分析改写 prompt → 注册新版本
4. 重复 2-3 直到分数不再涨

> 🍵 **类比**：像一个家教老师——它给学生（LLM）做卷子（train_data），看错题，让出题人（reflection 模型）改卷子（prompt），再让学生重做，直到分数提不动。

**两个 optimizer 对比**：

| Optimizer | 依赖 | 速度 | 智能程度 |
|-----------|------|------|---------|
| `MetaPromptOptimizer` | 内置（不需要额外包） | 快 | 中 |
| `GepaPromptOptimizer` | 需 `pip install gepa` | 慢 | 高 |

> ⚠️ **国内服务商兼容性**：GEPA 在 DeepSeek 上偶尔报 reflection 调用错误；MetaPrompt 更稳定。如果 optimize 失败，看下面的避坑清单。

### 4. ResponsesAgent —— 兼容 OpenAI Responses API 的 Agent 基类

OpenAI 在 2025 年推出了新的 Responses API（替代 Chat Completions）。MLflow 3 的 `ResponsesAgent` 就是"你的自定义 Agent"和"标准 OpenAI Responses 格式"之间的翻译层：

- 你继承 `ResponsesAgent` 实现 `predict()` 方法
- MLflow 自动把你的请求/响应翻译成 OpenAI Responses 格式
- 别人用 OpenAI SDK 调你的服务时，完全感知不到差别

**什么时候用**：你想让你的 LLM 应用暴露成一个"标准服务"，让前端/其他服务用 OpenAI 协议直接调用，而不用关心你内部怎么实现的。

### 5. Models-from-code —— 不 pickle，改用源码

MLflow 3 之前，`mlflow.pyfunc.log_model(python_model=my_model)` 会尝试用 `pickle` 序列化你的对象。但很多对象（比如 OpenAI 客户端、网络连接、lambda）pickle 不了或加载回来会失效。

MLflow 3 的解决方案：**直接把类定义所在的 .py 文件路径传过去**，MLflow 加载时 import 这个文件，找里面的类。简单粗暴但有效。

> 📚 **类比**：以前是"把家具拆了打包快递"（容易坏），现在是"把整个房间拍照给你照着装修"（更可靠）。

关键就是**文件末尾必须调用 `set_model(YourClass())`**——告诉 MLflow "这个文件里哪个对象是模型"。

---

## 实战步骤

### Step 1：确认环境

```bash
conda activate mlflow
# 确认环境变量已设置
echo $OPENAI_API_KEY
echo $OPENAI_API_BASE
echo $DEEPSEEK_MODEL
```

### Step 2：跑版本追踪（08a）

```bash
cd <project-root>
python 08_agents/08a_active_model.py
```

输出会显示：

- v1 三个问题 + 答案（每个一次 LLM 调用）
- v2 三个问题 + 答案（每个两次 LLM 调用：初答 + 反思）
- 跨两个 LoggedModel 的对比列表

### Step 3：跑 prompt 优化（08b）

```bash
python 08_agents/08b_prompt_optimize.py
```

可能会两种结果：

- **成功**：看到 v1 → v2 自动改写，template 文本被改进了
- **失败**：看到 `⚠️ 优化过程失败` 的提示——这是预期内的，国内服务商常见。脚本仍然会把 v1 注册到 Registry，只是不会自动产生 v2。没关系，看下面的 08c 也能继续。

### Step 4：跑 ResponsesAgent 打包（08c）

```bash
python 08_agents/08c_responses_agent.py
```

输出会显示：

- 直接调用 `SimpleQAAgent.predict()` 的结果
- 模型被 log 到 Registry 的 URI（`models:/m-xxx`）
- 加载回来再推理的结果

### Step 5：开 UI 检查

```bash
# 另开终端
mlflow ui --port 5000
```

浏览器开 `http://localhost:5000`，按上面"跑完必看"部分的路径看 Logged Models 和 Prompts。

### Step 6（选跑）：把模型 serve 起来

```bash
# 等 08c 跑完，会拿到 model_uri，serve 它
mlflow models serve -m models:/m-<你的model_id> -p 5001

# 另开终端，用 OpenAI 协议调用
curl http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"input": [{"role": "user", "content": "你好"}]}'
```

---

## 代码模式（可复用模板）

### 模式 1：每个代码版本一个 LoggedModel

```python
# 什么时候用：每次 git commit 后跑实验，自动归属到对应版本
import mlflow

with mlflow.start_run(run_name="agent-v2-batch"):
    mlflow.set_active_model(name="agent-v2")   # 后续 trace 自动归属
    for q in test_set:
        my_agent(q)   # 这个函数被 @mlflow.trace 装饰
```

### 模式 2：自动 prompt 优化

```python
# 什么时候用：你有一批带正确答案的样本，想自动改进 prompt
import mlflow
from mlflow.genai.optimize import MetaPromptOptimizer
from mlflow.genai.scorers import Correctness
import pandas as pd

train_data = pd.DataFrame([
    {"inputs": {"question": "..."}, "expectations": {"expected_response": "..."}},
])

result = mlflow.genai.optimize_prompts(
    predict_fn=lambda q: my_llm(q),                   # 你的预测函数
    train_data=train_data,                             # 训练数据
    prompt_uris=[f"prompts:/optimize-demo/1"],         # 起点 prompt
    optimizer=MetaPromptOptimizer(reflection_model="openai:/xxx"),
    scorers=[Correctness(model="openai:/xxx")],        # 打分函数
)
```

### 模式 3：自定义 ResponsesAgent

```python
# 什么时候用：想把自定义 LLM 应用打包成 MLflow 模型并兼容 OpenAI Responses API
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from mlflow.entities.span import SpanType
import mlflow

class MyAgent(ResponsesAgent):
    @mlflow.trace(span_type=SpanType.AGENT)
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        # 把 request.input 统一转成 [{role, content}] 列表
        messages = [{"role": m.role, "content": m.content} for m in request.input]
        resp = openai_client.chat.completions.create(model="...", messages=messages)
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=resp.choices[0].message.content)],
            custom_outputs=None,
        )
```

### 模式 4：Models-from-code 打包（必须配 set_model）

```python
# my_agent.py（独立文件）
from mlflow.models import set_model

class MyAgent(ResponsesAgent):
    def predict(self, request):
        ...

# 文件末尾：告诉 MLflow 哪个对象是模型
set_model(MyAgent())

# log 时：
mlflow.pyfunc.log_model(
    python_model="path/to/my_agent.py",   # 字符串路径，不是类实例！
    name="my-agent",
    pip_requirements=["openai", "mlflow>=3.0"],
)
```

### 模式 5：加载并推理（注意 dict 不是对象）

```python
# 什么时候用：把 log 好的模型从 Registry 加载回来用
loaded = mlflow.pyfunc.load_model("models:/m-<model_id>")

# ⚠️ PyFuncModel.predict() 接收 dict-like，不是 ResponsesAgentRequest 对象
api_request = {
    "input": [{"role": "user", "content": "..."}],
    "temperature": 0.3,
}
result = loaded.predict(api_request)
```

### 模式 6：LangChain 一行追踪（搭配 08a 使用）

```python
import mlflow
mlflow.langchain.autolog()   # 一行追踪所有 LangChain 调用

# 之后你的 LangChain 链每次 invoke 都会自动：
# - log span（包含每一步 LLM、Tool、Retriever）
# - 记录 token 消耗、延迟
# - 关联到当前 active LoggedModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("回答：{question}")
chain = prompt | llm

with mlflow.start_run():
    mlflow.set_active_model(name="agent-v1")
    chain.invoke({"question": "什么是 MLflow？"})
    # 上面这次调用会自动在 UI 里产生一条完整 trace
```

> 💡 **`autolog()` 是零侵入的**：你不需要给 LangChain 链加任何装饰器；只要在脚本开头调一次 `mlflow.langchain.autolog()`，后面所有 LangChain 调用都自动可观测。

---

## 🛠️ 动手做

> **目标**：完整跑一遍"代码版本 → LoggedModel → Prompt 优化 → ResponsesAgent 打包"链路，并在 UI 里看到所有产物。

**步骤**：

1. 确认环境变量 `OPENAI_API_KEY` / `OPENAI_API_BASE` / `DEEPSEEK_MODEL` 都设了。
2. 跑 `08a_active_model.py`：观察 v1（一次 LLM 调用）和 v2（初答 + 反思两次 LLM 调用）的 trace 数量差异。
3. 跑 `08b_prompt_optimize.py`：哪怕优化失败也没关系——确认 `optimize-demo` 这个 prompt 名在 Registry 里被注册了 v1。
4. 跑 `08c_responses_agent.py`：拿到 `model_info.model_id`（形如 `m-xxxxxxxx`）。
5. 启动 `mlflow ui --port 5000`，做三件事：
   - **左侧菜单 → Logged Models**：看到 `agent-v1` 和 `agent-v2` 两个独立实体；点开 `agent-v2`，在 **Traces** 标签看到刚才那 3 次调用。
   - **左侧菜单 → Prompts → optimize-demo**：看到 v1 的 commit message 和 template 全文。
   - **左侧菜单 → Experiments → 08_responses_agent → Run agent-packaging**：在 **Artifacts** 看到 `MLmodel` 文件、`requirements.txt`、以及 `simple_qa_agent.py` 的源码副本。
6. （选做）用 `mlflow models serve -m models:/m-<你的id> -p 5001` 把 Agent 拉起来，再用 `curl` 调 `/invocations` 验证它能回答问题。

**预期结果**：

- UI 里有 `agent-v1` 和 `agent-v2` 两个 LoggedModel，分别挂着 3 条 trace
- `optimize-demo` prompt 至少注册了 v1（可能还有 v2，如果 optimize 成功的话）
- `08_responses_agent` experiment 下有 `agent-packaging` 这个 Run，artifacts 里有完整的 agent 源码

---

## 避坑清单

- ⚠️ **`Failed to serialize Python model`** → 改用 Models-from-code：把 agent 类放到独立 .py 文件，文件末尾调用 `set_model(YourClass())`，`log_model` 时传文件路径字符串而不是类实例。
- ⚠️ **`predict()` 返回空 text** → 检查 `request.input` 里 `msg.content` 是字符串还是 list。ResponsesAgent 的 Message 类型允许 content 是字符串或 `list[ContentPart]`，统一处理（见 `simple_qa_agent.py` 第 41-46 行的写法）。
- ⚠️ **PyFuncModel schema 校验失败** → `loaded.predict()` 传 dict 而不是 `ResponsesAgentRequest` 对象。PyFuncModel 的 schema 校验不认识自定义类。
- ⚠️ **优化器在 DeepSeek 上 reflection 失败** → 用 `MetaPromptOptimizer`（更稳定）；或干脆手写循环：改 prompt → `register_prompt`（v2）→ `mlflow.genai.evaluate()` → 对比 score。
- ⚠️ **`set_active_model` 没生效** → 必须在 `@mlflow.trace` 装饰的函数里调用，或在 trace 上下文里。设了指针但没 trace 跑到，归属就是空的。
- ⚠️ **08b 跑失败但脚本没崩** → 正常现象。脚本用 `try/except` 包住了 optimize 调用，失败时打印提示但不中断。v1 仍然被注册，可以手动改进 prompt 然后 `register_prompt` 升 v2。
- ⚠️ **`mlflow.langchain.autolog()` 抓不到 trace** → 确认是 `mlflow.langchain.autolog()` 而不是 `mlflow.autolog()`——后者是 sklearn 用的；LangChain 框架必须调用框架专属的 autolog。

---

## 小结：5 个 take-aways

- **LoggedModel 是 MLflow 3 的核心升级**：模型不再是 Run 的附庸，而是独立的"版本实体"，能跨 Run/实验搜索、能注册别名、能直接挂载所有相关 trace。
- **`set_active_model` 是"无侵入"的版本标注**：一行代码就能让后续所有 trace 自动归属到指定 LoggedModel，不用手动给每个 trace 打标签。
- **`mlflow.langchain.autolog()` 是 LangChain 项目的标配**：零侵入（只调一行），自动追踪链上每一步 LLM、Tool、Retriever，并自动归到当前 active LoggedModel。
- **`optimize_prompts` 是"懒人的福音"但要选对 optimizer**：MetaPrompt 稳定够用、GEPA 强大但依赖多且国内服务商兼容性差——生产环境优先 MetaPrompt + 手写评估循环兜底。
- **ResponsesAgent + Models-from-code 是 LLM 应用上生产的标配**：前者解决"协议兼容"，后者解决"复杂对象打包"。两者配合让你的 Agent 既能被 OpenAI SDK 调用、又能避开 pickle 的坑。
- **每次部署前先在 UI 里确认 LoggedModel 状态**：看 trace 数量、看延迟分布、看别名是否设对——这三件事做完才能安心上线。

---

## 📖 下一步

到这里，你已经掌握了 MLflow 3 的核心能力：训练追踪、注册、评估、部署、prompt 管理、LLM Agent 打包。**全链路 MLOps 闭环的基础部分你已经能跑通**。

接下来可以深入的方向：

- **生产监控**：用 MLflow 的 evaluation dataset 跟踪数据漂移、概念漂移
- **CI/CD 集成**：把 `validate_evaluation_results` 塞进 GitHub Actions / GitLab CI，PR 合并前自动跑模型对比
- **云上部署**：把 `models serve` 替换为 SageMaker / Vertex AI / KServe
- **更高级的 Agent 框架**：LangGraph、AutoGen、LlamaIndex——MLflow 都已支持 autolog

更深入的内容请参考：

- `notes/08_agents.md` —— LLM Agent 阶段（LoggedModel / optimize_prompts / ResponsesAgent / Models-from-code）的详细笔记，包括 GEPA vs MetaPrompt 的对比、Models-from-code 的工作机制、ResponsesAgent 的 OpenAI Responses API 协议细节。
- MLflow 官方文档：[LLMs and Agents](https://mlflow.org/docs/latest/llms.html) 和 [Prompt Engineering](https://mlflow.org/docs/latest/prompts.html)。
# Chapter 12：生产级部署入门（选学）

> ⏱️ 预计时间：60 分钟
> 🔑 是否需 API Key：部分脚本需要（`09a` 需要 LLM key；`09b`、`09c` 不需要）
> 📚 前置知识：Chapter 4（trace）、Chapter 7（模型部署基础）、Chapter 11（监控基础）
> ⭐ 选学（仅当你准备把 MLflow 真正推到生产环境时再读）

## 🎯 这章做什么

到目前为止，你一直在自己电脑上跑 MLflow——`mlflow ui` 一开，本地 SQLite 数据库，本地文件系统存模型。这是"在自己家做饭"。

本章要解决的是"开餐馆"的问题：**把 MLflow 真正推到生产环境**。你会遇到三类以前完全不用想的事：

1. **存哪里**：数据越来越多，本地 SQLite 撑不住了，文件系统的模型权重也撑不住了
2. **要不要全记**：trace 一记就是几十万条，存储成本爆炸
3. **机器扛不扛得住**：CPU 飙到 100% 怎么办？磁盘满了怎么办？

**类比**：前面 11 章相当于在自己家做饭，本章相当于把厨房整套搬到店里——把食材存到冷库（Postgres）、把锅碗瓢盆存到仓库（MinIO）、给每桌客人只点一道试吃菜（采样 + 脱敏）、时刻盯着炉子火多大（硬件监控）。

### 你会学到什么

- 能看懂 MLflow 生产环境的"三层架构"——Client、Tracking Server、Backend Store + Artifact Store
- 能用 `docker-compose` 一键起 MLflow + Postgres + MinIO
- 能用 `mlflow models build-docker` 把任意 MLflow 模型打成 Docker 镜像
- 能给 trace 加采样，把存储成本压到原来的 1/10
- 能写一个 PII 脱敏函数，在数据"进 trace 之前"就把邮箱、手机、身份证洗掉
- 能用 `psutil` + MLflow 把 CPU/内存/磁盘/网络画成曲线
- 知道 MLflow 3.5+ `--allowed-hosts` 是干嘛的（防 DNS rebinding）
- 了解 MLflow ≥ 3.6 的 Agent Server 框架（`@invoke` / `@stream`）

### 对应脚本清单

| 脚本 | 一句话作用 | 是否必跑 | 前置 |
|------|-----------|---------|------|
| `09_deployment/09a_sampling_redaction.py` | 演示 trace 采样 + PII 脱敏，对比 raw vs redacted 两条 Run | ✓ 强烈推荐 | `OPENAI_API_KEY` |
| `09_deployment/09b_prod_infra.sh` | docker-compose 参考配置（Postgres + MinIO + MLflow） | 推荐（看一眼就行） | Docker 环境 |
| `09_deployment/09c_hardware_monitor.py` | 后台采样 CPU/内存/磁盘/网络 30 秒，画到 UI 上 | 推荐 | 无（不需要 API key） |

### 前置知识

- 已完成 Chapter 4（tracing），知道 trace 是什么、`@mlflow.trace` 怎么用
- 已完成 Chapter 7（model registry 和 build-docker）
- 装好 `mlflow`、`psutil`、`openai`：`pip install mlflow psutil openai`
- 本章假设你懂：什么是 LLM trace、什么是 Postgres/S3；不懂 docker-compose 也能跑 `09a` 和 `09c`
- ⚠️ 不需要真买云服务，`09b` 的 docker-compose 在本地就能起

---

## 核心概念

### 1.1 三层架构：Client → MLflow Server → Backend + Artifact

生产环境的 MLflow 不是单机文件，而是一套"三个角色"：

```
┌──────────────────────────────────────────────────┐
│  Client (训练脚本 / Web UI / API 调用方)           │
└────────────────┬─────────────────────────────────┘
                 │ HTTPS
┌────────────────▼─────────────────────────────────┐
│  MLflow Tracking Server (FastAPI, 端口 5000)    │
│  --backend-store-uri  → PostgreSQL (元数据)        │
│  --default-artifact-root → S3/MinIO (大文件)      │
└────────────────┬─────────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼────────────┐    ┌───────▼────────────┐
│ Model Server    │    │ Agent Server (3.6+) │
│ (REST /invocations)│ │  (@invoke/@stream)  │
└────────────────┘    └────────────────────┘
```

**Backend Store（后端存储）**：存"元数据"——Run 的名字、参数、指标、谁跑了什么。这是**结构化数据**，适合放 Postgres 这种关系型数据库。**绝对不要用 SQLite 上生产**，并发写会锁表。

**Artifact Store（构件存储）**：存"大文件"——模型权重、trace JSON、图片、CSV。这是**二进制文件**，适合放 S3（或本地用 MinIO 替代）。**绝对不要存到本地磁盘**，容器一重启就没了。

**类比**：Backend Store 是"图书馆的目录卡"（告诉你每本书在哪），Artifact Store 是"书库本身"（真正放书的地方）。两者职责完全不同，必须分开。

### 1.2 `mlflow models build-docker` 容器化

任何 MLflow Model（sklearn / PyTorch / ResponsesAgent / 自定义 pyfunc）都能用一条命令打成 Docker 镜像：

```bash
mlflow models build-docker \
  -m models:/my-model@champion \
  -n my-model:v1 \
  --env-manager conda

docker run -p 5001:8080 my-model:v1

# 测试推理
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  --data '{"inputs": [{"question": "..."}]}'
```

**类比**：这相当于把"实验厨房里做好的菜"装进外卖盒，随时可以微波炉加热端给客人。镜像是自包含的，换台机器跑也能跑出同样的结果。

### 1.3 docker-compose：Postgres + MinIO + MLflow 一键起

生产环境的"最小可用三件套"——一份 YAML 就能起：

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
      POSTGRES_DB: mlflow
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: minio123
    volumes:
      - miniodata:/data
    ports:
      - "9000:9000"   # API
      - "9001:9001"   # Web UI

  createbuckets:
    image: minio/mc
    depends_on:
      - minio
    entrypoint: >
      /bin/sh -c "
      mc alias set local http://minio:9000 minio minio123;
      mc mb -p local/mlflow-artifacts;
      mc anonymous set download local/mlflow-artifacts;
      exit 0;
      "

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    depends_on:
      - postgres
      - minio
    environment:
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: minio
      AWS_SECRET_ACCESS_KEY: minio123
    command: >
      mlflow server
        --host 0.0.0.0
        --port 5000
        --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow
        --default-artifact-root s3://mlflow-artifacts/
        --allowed-hosts "*"
    ports:
      - "5000:5000"

volumes:
  pgdata:
  miniodata:
```

**关键解读**：
- `postgres` 存元数据（Run / Metric / Param）
- `minio` 是 S3 替代品，存模型权重、trace 文件
- `createbuckets` 是初始化容器，只跑一次建 bucket
- `mlflow` 服务启动时**必须**同时连上 Postgres 和 MinIO，否则 502

### 1.4 Trace 采样（控制成本）

生产环境最大的隐性成本是 **trace 存储**。每个 trace 平均 ~50KB，1 亿条 = 50TB。100 QPS 全量记 trace 一个月 8.6 亿条，**采样 10% 直接省 90% 存储**。

**核心思路**：不是每个请求都记 trace，而是按比例抽一部分。

| 场景 | 采样率 |
|------|--------|
| 调试 / PoC | 100% |
| 一般生产 | 10-20% |
| 高流量 (>1k QPS) | 1-5% |

**类比**：餐厅试吃不是每桌都给一整本菜单，只给 1/10 的客人发试吃小碟——成本可控、口味覆盖到了、客人还觉得被重视。

应用层装饰器实现（脚本 `09a` 用法）：

```python
import random, functools

def sampled_trace(sample_rate: float = 0.1):
    """只对 sample_rate 比例的调用进行 trace 记录"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if random.random() < sample_rate:
                return func(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@mlflow.trace(span_type="AGENT")
@sampled_trace(rate=0.1)
def my_agent(q):
    ...
```

MLflow 3 也提供 OpenTelemetry 风格的 sampler：

```python
from mlflow.tracing.sampling import TraceIdRatioBased
```

### 1.5 PII 脱敏（隐私合规）

**PII = Personally Identifiable Information**，个人信息。邮箱、手机、身份证、信用卡、姓名、IP 都算。

**关键原则**：在 **trace 边界**（也就是函数入口）就要把 PII 洗掉，而不是先 trace 再洗——因为一旦进入 trace 存储，**就泄漏了**。

**类比**：快递单上不要写真实手机号，写成 "**** 1234"。同理，trace 里存的应该是 "[EMAIL]"而不是 `zhangsan@example.com`。

可复用的脱敏函数（递归清洗 dict / list / str）：

```python
import re

def redact_pii(data):
    """递归脱敏 dict/list/str"""
    if isinstance(data, dict):
        return {k: redact_pii(v) for k, v in data.items()}
    if isinstance(data, list):
        return [redact_pii(item) for item in data]
    if isinstance(data, str):
        data = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "[EMAIL]", data)
        data = re.sub(r"1[3-9]\d{9}", "[PHONE]", data)
        data = re.sub(r"\d{17}[\dXx]", "[ID_CARD]", data)
        data = re.sub(r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}", "[CARD]", data)
        data = re.sub(r"我叫[一-龥]{2,4}", "我叫[NAME]", data)
        return data
    return data
```

### 1.6 硬件监控（CPU/内存/磁盘/网络）

MLflow autolog 对 PyTorch/TensorFlow 会自动记 GPU 利用率，但 **CPU/内存的历史曲线它不管**。要补这块，需要用 `psutil` 手动采样，再 `mlflow.log_metric` 写进 Run。

**类比**：做饭时不仅要记"客人点了什么菜"，还要记"厨房的温度和燃气用量"——出问题时要能回溯是哪个环节超载了。

`psutil.cpu_percent()` 有个"坑"：**第一次调用只设基线不返回值**。所以循环外必须先空调一次：

```python
import psutil, mlflow, time

# 关键：第一次必须空调用一次！
psutil.cpu_percent(interval=None)

while True:
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    mlflow.log_metric("cpu_percent", cpu, step=int(time.time()))
    mlflow.log_metric("mem_percent", mem.percent, step=int(time.time()))
    time.sleep(1)
```

可采集的指标：
- `psutil.cpu_percent()`：CPU 总体利用率
- `psutil.virtual_memory()`：内存（used / available / percent）
- `psutil.disk_usage('/')`：磁盘用量
- `psutil.disk_io_counters()`：磁盘读写字节数
- `psutil.net_io_counters()`：网络收发字节数

### 1.7 MLflow 3.5+ `--allowed-hosts` 防 DNS rebinding

DNS rebinding 是一种攻击：恶意 DNS 把域名解析到 `127.0.0.1`，诱导浏览器访问本机服务。MLflow 3.5 之前 server 默认接受任意 Host header，3.5+ 必须显式声明 `--allowed-hosts`：

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri postgresql://... \
  --default-artifact-root s3://... \
  --allowed-hosts "mlflow.example.com,localhost"
```

本地开发可以用 `--allowed-hosts "*"`，但生产必须写具体的域名。

### 1.8 Agent Server（MLflow ≥ 3.6.0）

MLflow 3.6 引入了一个新的服务端框架——用装饰器把普通 async 函数变成 HTTP endpoint：

```python
from mlflow.genai.agent_server import invoke, stream, AgentServer

agent_server = AgentServer()

@invoke()
async def non_stream_endpoint(request):
    return await my_agent.run(request)

@stream()
async def stream_endpoint(request):
    async for chunk in my_agent.stream(request):
        yield chunk
```

**类比**：以前要写 FastAPI 路由把 agent 包成服务，现在加两个装饰器就完事。`@invoke` 给一次性返回，`@stream` 给流式响应（边生成边吐 token）。

### 1.9 MLflow 的边界（什么时候用它、什么时候换工具）

MLflow 不是万能工具。下面这张表告诉你哪些功能它能做、哪些要靠别的工具：

| 需求 | MLflow 能做？ | 推荐工具 |
|------|--------------|----------|
| 实验追踪（参数/指标/artifact） | 完美 | MLflow |
| Trace 记录和回放 | 完美 | MLflow |
| 模型注册和版本管理 | 完美 | MLflow |
| 模型部署 | 可以 | MLflow + Docker |
| 模型服务监控 | 部分 | Prometheus + Grafana |
| 阈值告警（CPU > 90% 报警） | 不能 | Prometheus Alertmanager |
| 分布式 tracing（多服务） | 部分 | OpenTelemetry + Jaeger |
| APM（应用性能管理） | 不能 | Datadog / New Relic |

---

## 实战步骤

### 2.1 跑 PII 脱敏 + 采样对比（强烈推荐）

```bash
# 1. 激活环境
conda activate mlflow

# 2. 设 API key（脚本要调真实 LLM）
export OPENAI_API_KEY=sk-xxx
export OPENAI_API_BASE=https://api.deepseek.com
export DEEPSEEK_MODEL=deepseek-chat

# 3. 跑脚本
python 09_deployment/09a_sampling_redaction.py
```

脚本会跑两遍同一组 5 个客服请求：一次"原始版"（含 PII），一次"脱敏版"。

**在终端会看到**：
- 第 [A] 段：5 条 trace 的输入里都能看到 `@example.com`、`13812345678` 等
- 第 [B] 段：同样的 5 条输入被替换成 `[EMAIL]`、`[PHONE]`、`[ID_CARD]`

**在 UI 看**：
1. `mlflow ui --port 5000`
2. 选 experiment `09_sampling_pii` → 对比 Run `raw-no-redaction` 和 Run `redacted`
3. 点开任一 Run → 看 **Traces** 标签
4. 对比两个 Run 的 `trace_inputs`：
   - `raw-no-redaction` 应该能看到 `zhangsan@example.com`、`13812345678`、`110101199001011234`
   - `redacted` 同样的输入会变成 `[EMAIL]`、`[PHONE]`、`[ID_CARD]`

### 2.2 启动生产 MLflow（用 docker-compose）

把 `09b_prod_infra.sh` 里那段 YAML heredoc 单独保存为 `docker-compose.yml`，然后：

```bash
docker compose up -d
# 访问 http://localhost:5000
```

**第一次启动会做的事**：
1. 启动 Postgres 并初始化 mlflow 数据库
2. 启动 MinIO 并创建一个叫 `mlflow-artifacts` 的 bucket
3. 启动 MLflow Tracking Server，连上 Postgres 和 MinIO

**客户端连接**：

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

**容器化已有模型**：

```bash
mlflow models build-docker -m models:/my-model@champion -n my-model:v1
docker run -p 5001:8080 my-model:v1
```

### 2.3 跑硬件监控

```bash
conda activate mlflow
python 09_deployment/09c_hardware_monitor.py
```

脚本会后台起采样线程（每秒采一次），同时跑 30 秒"负载剧本"：
- 0-5s：空闲
- 5-10s：CPU 密集
- 10-15s：分配内存
- 15-20s：CPU 密集
- 20-25s：分配更多内存
- 25-30s：空闲

另开终端看曲线：

```bash
mlflow ui --port 5000
```

选 experiment `09_hardware_monitor` → Run `hardware-monitor-demo` → **Metrics** 标签看曲线。

应该看到：
- `cpu_percent` 在 5-10s 期间飙到接近 100%
- `mem_percent` 在 10-15s 和 20-25s 期间明显上升
- `disk_read_mb` / `disk_write_mb` 是磁盘读写字节数

### 2.4 （选跑）Agent Server 框架（MLflow ≥ 3.6）

```python
from mlflow.genai.agent_server import invoke, stream, AgentServer

agent_server = AgentServer()

@invoke()
async def non_stream_endpoint(request):
    return await my_agent.run(request)

@stream()
async def stream_endpoint(request):
    async for chunk in my_agent.stream(request):
        yield chunk
```

### 2.5 故障排查速查

| 现象 | 原因 | 解决 |
|------|------|------|
| `mlflow.db is locked` | SQLite 并发写 | 换 Postgres |
| MinIO bucket 不存在 | 没跑 `createbuckets` | 手动 `mc mb local/mlflow-artifacts` |
| `mlflow server` 启动后 502 | Postgres 还没起来 | `depends_on` 加 `condition: service_healthy` |
| Trace 只写一半 | 采样没配置 | 检查 `mlflow.tracing.sampling` 设置 |
| `psutil.cpu_percent()` 一直返回 0 | 第一次没设基线 | 循环外先空调一次 |

---

## 🛠️ 动手做：跑 09a 看 UI 对比

**任务**：跑 `09a_sampling_redaction.py`，然后在 UI 里对比 raw vs redacted 两条 Run 的 trace_inputs。

**步骤**：

1. 准备 API key 环境变量
2. 跑脚本：`python 09_deployment/09a_sampling_redaction.py`
3. 终端最后会打印两个 Run 的 trace_inputs 前 120 字符
4. 启动 UI：`mlflow ui --port 5000`
5. 浏览器打开 `http://localhost:5000`
6. 选 experiment `09_sampling_pii`
7. 对比 Run `raw-no-redaction` 和 Run `redacted`
8. 点开任一 Run → 切到 **Traces** 标签
9. 看每条 trace 的 `trace_inputs` 字段

**预期看到**：
- `raw-no-redaction`：能看到完整邮箱 `zhangsan@example.com`、手机 `13812345678`、身份证 `110101199001011234`
- `redacted`：同样的输入变成 `[EMAIL]`、`[PHONE]`、`[ID_CARD]`

**思考**：
- 假如你的同事不小心把生产 trace 备份发到群里，会发生什么？raw Run 是不是就成了"社死现场"？
- 你能想到哪些 PII 是这个正则漏掉的？（提示：地址、银行卡号段、人名"张三"之外的写法）

---

## 避坑清单

- **后端存了 SQLite** → 生产并发写会锁表。换 Postgres：`--backend-store-uri postgresql://...`
- **构件存在本地磁盘** → 容器重启数据丢失。换 S3/MinIO：`--default-artifact-root s3://...`
- **PII 进了 trace 才发现要洗** → 已经泄漏，永远清不干净。**在 trace 入口函数第一行就脱敏**
- **trace 全量 100% 记录** → 100 QPS 一个月 50TB 存储。立刻降到 10-20%
- **`psutil.cpu_percent()` 第一次返回 0** → 这是 psutil 的"坑"，第一次调用只设基线不返回值。必须先空调一次再进入循环
- **MLflow 没配 `--allowed-hosts`（< 3.5.0）** → 有 DNS rebinding 漏洞，至少升到 3.5 并显式配置
- **用 MLflow 当专业监控** → MLflow 没有阈值告警、没有历史回溯，工业标准是 **Prometheus + Grafana**。MLflow 只适合"在 Run 里附带硬件 snapshot"
- **没有 retention policy** → 数据库和 S3 会无限增长。定期跑 `mlflow gc --backend-store-uri $POSTGRES_URI`
- **用明文密码写在 docker-compose** → 生产换 Kubernetes Secret 或 Vault
- **没有 liveness probe** → MLflow server 挂了 K8s 不会重启。要加 `healthcheck`

---

## 小结：3-5 个 take-aways

- 生产 MLflow = **Tracking Server + Postgres（后端）+ S3/MinIO（构件）**，缺一不可
- Trace 成本爆炸靠 **采样** 控制，10% 采样省 90% 存储是高 ROI 操作
- **PII 必须在 trace 边界（函数入口）就脱敏**，泄漏之后再洗已经晚了
- MLflow 不是监控工具，要做硬件告警用 **Prometheus + Grafana**，MLflow 只做"实验 Run 内的硬件 snapshot"
- 任何 MLflow Model（含 sklearn / PyTorch / ResponsesAgent）都能用 `mlflow models build-docker` 一键打成 Docker 镜像

---

## 📖 下一步

本章是**选学**章节。如果你只是用 MLflow 跑实验、看对比，不用学 docker-compose 也不用管 PII——前面的章节已经够用。

如果你准备把模型推到生产环境（哪怕只是团队内部用），建议你：

1. 跑一遍 `09a`（看 trace 采样 + PII 脱敏的对比）
2. 在本地用 docker-compose 起一遍 MLflow + Postgres + MinIO（熟悉命令）
3. 跑一遍 `09c`（看硬件监控的 UI 曲线长什么样）
4. 把脚本里的 `redact_pii` 函数复制到你的项目里——这是"以后肯定用得上"的代码
5. **遇到报错时翻 Chapter 13（Debug 指南）**

更详细的生产 Checklist、成本估算、Prometheus 接入等内容在 `notes/09_deployment.md` 里——那篇笔记是本章的"完整版"，覆盖了 6.4 节的月度成本估算和 Prometheus + Grafana 接入细节。

---

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

| 情况 | 解决 |
|------|------|
| 端口被占 | 杀掉旧进程 或 `mlflow ui --port 5001` 换端口 |
| allowed-hosts 报错 | 加 `--allowed-hosts "*"` 或具体域名 |
| 服务没起来 | 看 docker-compose 依赖顺序（postgres 是否 healthy） |
| curl 连不上 | 检查防火墙、container 网络、host port 映射 |

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

| 场景 | 解决 |
|------|------|
| 进程被 kill -9 | 跑 `mlflow runs terminate --run-id <id>` 手动关闭 |
| 忘了 `with` 块 | 改用 `with mlflow.start_run():` 包起来（自动清理） |
| 异常退出 | `try / except / finally` 里手动 `mlflow.end_run()` |
| 网络中断 | 检查 `mlflow server` 是否还活着，重连后跑 `terminate` |

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

| 类型 | 例子 | 用 log_param 还是 log_metric |
|------|------|------------------------------|
| 实验配置 | learning_rate, batch_size, model_type | `log_param` |
| 训练过程 | epoch_loss, lr_schedule, step | `log_metric` |
| 评估指标 | accuracy, f1, auc | `log_metric` |

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

| 场景 | 解决 |
|------|------|
| URI 写错 | 用 `search_registered_models()` 查正确的 version/alias |
| `file://` 不支持 | 换 SQLite：`mlflow.set_tracking_uri("sqlite:///mlflow.db")` |
| DB schema 旧 | 跑 `mlflow db upgrade <uri>` |
| artifact 丢了 | 重新 log 模型 + 注册 |

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

| 场景 | 解决 |
|------|------|
| URI 不一致 | 显式 `mlflow.set_tracking_uri()` + 重启 UI |
| autolog 没开 | 加 `mlflow.openai.autolog()` / `mlflow.anthropic.autolog()` |
| async 没 flush | 脚本结尾 `time.sleep(2)` 或 `mlflow.flush_trace_logging()` |
| experiment 找错 | 用 `get_experiment_by_name` 拿 ID 后再 search |

### 预防

- 脚本顶部统一设 `mlflow.set_tracking_uri()`
- 开启对应 LLM 的 autolog
- 脚本结尾 `time.sleep(2)` 等 flush（成本很低）
- 用 `mlflow.search_traces()` 程序化检查，别只靠 UI

---

## 调试速查表（打印贴桌边）

| 症状 | 第一反应 |
|------|---------|
| 启动报错 | `lsof -i :5000` + `--allowed-hosts "*"` |
| Run 卡 RUNNING | `mlflow runs terminate --run-id <id>` |
| register_model 失败 | `mlflow.set_tracking_uri("sqlite:///mlflow.db")` |
| Changing param 警告 | 改用 `log_metric` |
| load_model RESOURCE_DOES_NOT_EXIST | `mlflow db upgrade <uri>` + `search_registered_models` |
| trace 没出现 | `mlflow.search_traces()` + `time.sleep(2)` |

---

## 📖 下一步

这一章是工具书——遇到报错随时翻，不需要从头读到尾。

**更全的错误信息**：完整的错误信息、stack trace 示例、CLI 命令清单见 `mlflow_skill/classical-ml/references/troubleshooting.md`。那篇文档是 MLflow Debug 的"完整版"，包含：

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

| 旧（不要用） | 新（用这个） |
|---|---|
| `artifact_path=` | `name=` |
| `transition_model_version_stage(..., "Production")` | `set_registered_model_alias(..., "champion", version)` |
| `mlflow.evaluate(..., baseline_model=uri, custom_metrics=[fn])` | `mlflow.models.evaluate(..., extra_metrics=[make_metric(eval_fn=fn, ...)])` + `mlflow.validate_evaluation_results(thresholds, candidate, baseline)` |
| `runs:/<id>/<path>` 加载模型 | `models:/<model_id>` （MLflow 3 LoggedModel）|
| 模型 URI 含 `runs:/` | `models:/<name>@<alias>` |
| `Stage`（Staging/Production） | `Alias`（champion/challenger）|
| `mlflow.evaluate` | `mlflow.models.evaluate`（经典 ML）/ `mlflow.genai.evaluate`（GenAI）|
| `mlflow.pyfunc.log_model(python_model=class_instance())` | `mlflow.pyfunc.log_model(python_model="path/to/file.py")` + `set_model()` |

---

下一段（如果还有的话）会进入实战：选一个真实数据集（比如 sklearn 的 wine / Iris）完整跑一遍"训练 → 评估 → 注册 → 部署"流程，把这一段的 4 个 skill 串起来用。

---

# Skill 段：12 个 MLflow Skill 介绍

> 本段面向所有读者：项目 `mlflow_skill/` 目录里有 12 个 `SKILL.md` 文件——它们是给 AI 编程助手读的"操作手册"（比如 Claude Code、Cursor 等）。本段不教你怎么让 AI 启用这些 skill（每个 AI 助手配置不同），而是告诉你**这 12 个 skill 各自管什么、什么场景需要用、用的时候看哪段**。
>
> 即使你不用 AI 助手，看 `SKILL.md` 本身也是学习 MLflow 最佳实践的好材料。

---

## 一、12 个 Skill 总览

下表是 `mlflow_skill/` 目录下全部 skill 的速查。每行告诉你：这个 skill 叫什么、它管什么、什么场景要用它、要读哪段 `SKILL.md`。

| # | Skill 名字 | 用途 | 触发场景（你说什么话） | 读哪段 |
|---|----------|------|-----------------|-------|
| 1 | `mlflow-onboarding` | 引导上手：判断你要做什么，给出 quickstart | "怎么开始用 MLflow"、"刚装好"、"新手入门" | `mlflow-onboarding/SKILL.md` |
| 2 | `classical-ml` | 传统 ML 6 步法：tracking → registry → evaluate → deploy → monitor → optimize | "训练 sklearn/xgboost"、"对比 runs"、"注册模型"、"部署模型"、"模型监控"、"调超参" | `mlflow_skill/classical-ml/SKILL.md`（最完整） |
| 3 | `instrumenting-with-mlflow-tracing` | 给 LLM 代码加 tracing（Python / TypeScript） | "给我的 OpenAI 加 trace"、"给 LangChain 加追踪" | `mlflow_skill/instrumenting-with-mlflow-tracing/SKILL.md` |
| 4 | `agent-evaluation` | 评估 LLM agent 输出质量（dataset + scorer + evaluate） | "评估我的 agent"、"算准确率"、"用 LLM-as-judge 评分" | `mlflow_skill/agent-evaluation/SKILL.md` |
| 5 | `querying-mlflow-metrics` | 拉聚合指标（token 用量、延迟、成本、trace 数） | "分析 token 用量"、"看延迟趋势"、"算 LLM 成本" | `mlflow_skill/querying-mlflow-metrics/SKILL.md` |
| 6 | `retrieving-mlflow-traces` | 搜索 / 过滤 trace | "找失败的 trace"、"查 latency > 5s 的" | `mlflow_skill/retrieving-mlflow-traces/SKILL.md` |
| 7 | `analyze-mlflow-trace` | debug 单个 trace | "这个 trace 哪里出错了"、"trace ID 是 tr-xxx 帮我看看" | `mlflow_skill/analyze-mlflow-trace/SKILL.md` |
| 8 | `analyze-mlflow-chat-session` | debug 多轮对话 / session | "看这个 chat session 哪里出问题" | `mlflow_skill/analyze-mlflow-chat-session/SKILL.md` |
| 9 | `fix-agent-issue` | 改 agent 行为的探索→计划→实现→验证闭环 | "agent 行为不对"、"想加个业务规则" | `mlflow_skill/fix-agent-issue/SKILL.md` |
| 10 | `mlflow-agent` | 通用 MLflow master dispatcher（不知道用哪个就让它路由） | 任何 MLflow workflow 但你没说要用哪个 skill | `mlflow_skill/mlflow-agent/SKILL.md` |
| 11 | `searching-mlflow-docs` | 拉官方文档（mlflow.org/docs/latest） | "MLflow 怎么用 X"、"查 MLflow API" | `mlflow_skill/searching-mlflow-docs/SKILL.md` |
| 12 | `sagemaker-mlflow` | 连 AWS SageMaker Managed MLflow 当后端 | "SageMaker 上装 MLflow" | `mlflow_skill/sagemaker-mlflow/SKILL.md` |

> **怎么用上表**：当你遇到一个 MLflow 任务时，先看"触发场景"列有没有匹配的关键词。匹配了就去找对应 skill 的 `SKILL.md` 读。读完不一定要让 AI 帮你做，自己按步骤跑也行。

---

## 二、按你的需求选 skill

不知道用哪个？按下面这张表对号入座：

| 你想做什么 | 推荐 skill | 读哪段 |
|----------|----------|-------|
| **第一次用 MLflow**，不知道怎么开始 | `mlflow-onboarding` | `mlflow-onboarding/SKILL.md` |
| **训练 sklearn/xgboost/lightgbm** 模型并自动记录 | `classical-ml` | `mlflow-skill/classical-ml/SKILL.md`（Step 1: Tracking） |
| **对比多个模型** 找最好的 | `classical-ml` | `classical-ml/SKILL.md`（Step 3: Evaluate） |
| **给 LLM 代码加 trace**（OpenAI / LangChain / Anthropic） | `instrumenting-with-mlflow-tracing` | `instrumenting-with-mlflow-tracing/SKILL.md` |
| **评估 LLM agent 答得准不准** | `agent-evaluation` | `agent-evaluation/SKILL.md` |
| **查 token 用量、延迟、成本** | `querying-mlflow-metrics` | `querying-mlflow-metrics/SKILL.md` |
| **找哪个 trace 失败了** | `retrieving-mlflow-traces` | `retrieving-mlflow-traces/SKILL.md` |
| **debug 单个 trace 哪里出问题** | `analyze-mlflow-trace` | `analyze-mlflow-trace/SKILL.md` |
| **debug 多轮对话** | `analyze-mlflow-chat-session` | `analyze-mlflow-chat-session/SKILL.md` |
| **想改 agent 行为**（业务规则 / 偏好） | `fix-agent-issue` | `fix-agent-issue/SKILL.md` |
| **不知道用哪个 skill** | `mlflow-agent` | `mlflow-agent/SKILL.md` |
| **查 MLflow 官方文档** | `searching-mlflow-docs` | `searching-mlflow-docs/SKILL.md` |
| **在 AWS SageMaker 上部署 MLflow** | `sagemaker-mlflow` | `sagemaker-mlflow/SKILL.md` |

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
