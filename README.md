# MLflow 3 学习项目

> 从零开始学 **MLflow 3**（2025-2026 主流版），同时覆盖传统 ML 和 LLM/Agent 工作流
> 基于官方中文文档站系统性调研：`mlflow.org.cn/docs/latest/ml/` + `mlflow.org.cn/docs/latest/genai/`

---

## 🎯 新人第一站

**如果这是你第一次接触这个项目**，按这个顺序来：

1. 📌 **[`notes/00_quickstart.md`](notes/00_quickstart.md)** — **作业指导书（必读！）**
   面向 MLflow 零基础用户的完整引导：14 章节 + 12 个 skill 介绍，从环境安装到 vibecoding 集成一气呵成。

   | 章节 | 内容 | 必学 |
   |------|------|------|
   | Ch 0-1 | 前置认知 + 环境安装 | ✓ |
   | Ch 2-6 | 传统 ML 核心 + GenAI 追踪 | ✓ |
   | Ch 7 | **Vibecoding 集成**（用 AI 助手操作 MLflow）| ✓ |
   | Ch 8-12 | Prompt / 评估 / 部署 / Agent / 生产（选学）| △ |
   | Ch 13-14 | Debug 指南 + 速查表 | ✓ |
   | Skill 段 | 12 个 mlflow_skills 介绍 | ✓ |

2. 📌 **[`mlflow_skills/README.md`](mlflow_skills/README.md)** — **MLflow Skills 官方合集**
   12 个给 AI 编程助手用的 skill（tracing / evaluation / debugging / classical-ml），配合 vibecoding 使用。

3. 📌 **[`notes/appendix/roadmap.md`](notes/appendix/roadmap.md)** — 9 阶段完整学习路线图

---

## 目录结构

```
MLFlowLearning/
├── README.md                     本文件
├── environment.yml               conda 一键建环境（推荐）
├── requirements.txt              Phase 1-9 + capstone 主依赖（pip 装）
├── requirements-phase10.txt      深度学习图像分类额外依赖
├── .env.example                  API Key 模板（DeepSeek）
├── notes/                        学习笔记
│   ├── 00_quickstart.md          ← 作业指导书（新人第一站）
│   ├── 01_basics.md              ↔ scripts/01_basics/        （Phase 1）
│   ├── 02_registry.md            ↔ scripts/02_registry/      （Phase 2）
│   ├── 03_tracking.md            ↔ scripts/03_tracking/      （Phase 3）
│   ├── 04_evaluate.md            ↔ scripts/04_evaluate/      （Phase 4）
│   ├── 05_tracing.md             ↔ scripts/05_tracing/       （Phase 5）
│   ├── 06_prompts.md             ↔ scripts/06_prompts/       （Phase 6）
│   ├── 07_evaluation.md          ↔ scripts/07_evaluation/    （Phase 7）
│   ├── 08_agents.md              ↔ scripts/08_agents/        （Phase 8）
│   ├── 09_deployment.md          ↔ scripts/09_deployment/    （Phase 9）
│   ├── 10_vision_classification.md  ↔ scripts/10_vision_classification/  （Phase 10：图像分类）
│   ├── capstone.md               ↔ scripts/capstone/         （毕业项目）
│   └── appendix/                 不直接对应脚本的参考文档
│       ├── roadmap.md              9 阶段路线图
│       ├── api_cheatsheet.md       MLflow 3 API 速查
│       └── mlflow3_breaking_changes.md  MLflow 3 vs 2 关键变化
├── mlflow_skills/                 12 个 MLflow skill（给 AI 编程助手用）
│   ├── README.md                 skill 总览 + 安装说明
│   ├── instrumenting-with-mlflow-tracing/   给代码加追踪
│   ├── agent-evaluation/         评估 LLM agent
│   ├── classical-ml/             传统 ML 6 步法
│   └── ...（共 12 个）
├── scripts/01_basics/                Phase 1 脚本：入门与追踪基础
├── scripts/02_registry/              Phase 2 脚本：模型格式与注册表
├── scripts/03_tracking/              Phase 3 脚本：追踪服务器与数据集血缘
├── scripts/04_evaluate/              Phase 4 脚本：评估、服务与经典 ML 验证
├── scripts/05_tracing/               Phase 5 脚本：GenAI 追踪（含 env_bootstrap.py）
├── scripts/06_prompts/               Phase 6 脚本：Prompt Registry
├── scripts/07_evaluation/            Phase 7 脚本：GenAI 评估
├── scripts/08_agents/                Phase 8 脚本：版本追踪 + ResponsesAgent
├── scripts/09_deployment/            Phase 9 脚本：部署与生产可观测性
├── scripts/10_vision_classification/  Phase 10 脚本：图像分类深度学习对比（9 个 timm CNN）
└── scripts/capstone/                 毕业项目 SupportPilot
```

**约定**：每个 phase 一个目录，一个脚本目录对应一个同名笔记（`notes/<NN>_<name>.md`）。
不直接对应脚本的参考文档放 `notes/appendix/`。

**运行约定**：脚本可以从**任何目录**直接运行（项目根、IDE 当前文件、phase 子目录都行）。
每个 phase 脚本顶部都 `import _paths`，自动把 tracking URI 锚定到项目根的 `mlflow.db`——
无论你在哪儿跑，UI 里看到的都是同一个数据库。db 文件 = `<project-root>/mlflow.db`，
artifact 目录 = `<project-root>/mlruns/`。

---

## 10 阶段速览

| # | 主题 | 关键能力 | 脚本目录 | 笔记 | API Key |
|---|------|----------|---------|------|---------|
| 1 | 入门与追踪基础 | Experiment/Run/Autolog | `scripts/01_basics/` | [01_basics](notes/01_basics.md) | ❌ |
| 2 | 模型格式与注册表 | MLmodel + 签名 + Registry + Aliases | `scripts/02_registry/` | [02_registry](notes/02_registry.md) | ❌ |
| 3 | 追踪服务器与数据集血缘 | sqlite backend + `mlflow.data` + `search_logged_models` | `scripts/03_tracking/` | [03_tracking](notes/03_tracking.md) | ❌ |
| 4 | 评估、服务与 ML 验证 | `mlflow.models.evaluate` + `validate_evaluation_results` + `models serve` | `scripts/04_evaluate/` | [04_evaluate](notes/04_evaluate.md) | ❌ |
| 5 | **GenAI 追踪**（重点） | OpenAI/Anthropic autolog + `@mlflow.trace` + 元数据 | `scripts/05_tracing/` | [05_tracing](notes/05_tracing.md) | ✅ |
| 6 | Prompt Registry + Flavors | 注册/版本化 + alias + LangChain autolog | `scripts/06_prompts/` | [06_prompts](notes/06_prompts.md) | ✅ |
| 7 | GenAI 评估 + 自定义 Scorer | `mlflow.genai.evaluate` + `@scorer` + `make_judge` | `scripts/07_evaluation/` | [07_evaluation](notes/07_evaluation.md) | ✅ |
| 8 | 版本追踪 + 提示词优化 + ResponsesAgent | `set_active_model` + GepaOptimizer + ResponsesAgent | `scripts/08_agents/` | [08_agents](notes/08_agents.md) | ✅ |
| 9 | 部署 + 生产可观测性 | trace 采样 + PII 脱敏 + 生产部署参考 | `scripts/09_deployment/` | [09_deployment](notes/09_deployment.md) | ✅ |
| 10 | **图像分类深度学习对比** | timm 9 个 CNN + CIFAR-10 + 失败案例分析 | `scripts/10_vision_classification/` | [10_vision_classification](notes/10_vision_classification.md) | ❌（但需 torch/timm） |
| 🎓 | **SupportPilot** 毕业项目 | sklearn gate + LangChain RAG + Prompt Registry + GenAI eval | `scripts/capstone/` | [capstone](notes/capstone.md) | ✅ |

**设计理念**：先用 scikit-learn（无需 API key）跑通 MLflow 核心概念 → 再用同一套概念套到 LLM/Agent → 毕业项目串联全能力。

---

## mlflow_skills：12 个 skill 总览

`mlflow_skills/` 是 [MLflow Skills](https://github.com/mlflow/skills) 官方合集，给 AI 编程助手（Claude Code / Cursor / Codex / Gemini CLI / OpenCode 等）用的"指令手册"。配合 **vibecoding**（对话式编程），你不需要精通 MLflow API，让 AI 助手读这些手册帮你操作即可。

### 四大类 skill

| 类别 | Skill | 用途 |
|------|-------|------|
| **可观测性 / 调试** | `instrumenting-with-mlflow-tracing` | 给 Python/TypeScript 代码加 MLflow 追踪 |
| | `analyze-mlflow-trace` | debug 单条 trace（查 span、找根因）|
| | `analyze-mlflow-chat-session` | debug 多轮对话 session |
| | `retrieving-mlflow-traces` | 搜索/过滤 trace（按状态/用户/session/时间）|
| **评估 / 指标** | `agent-evaluation` | 端到端评估 LLM agent（数据集 + scorer + 执行）|
| | `querying-mlflow-metrics` | 拉聚合指标（token 用量/延迟/成本）|
| **新手引导** | `mlflow-onboarding` | 判断你的用例（GenAI vs 传统 ML）并引导 |
| | `searching-mlflow-docs` | 高效搜官方文档（用 llms.txt 索引）|
| **传统 ML** | `classical-ml` | 完整生命周期：tracking → registry → evaluate → deploy → monitor → optimize |
| **部署** | `sagemaker-mlflow` | 连 AWS SageMaker Managed MLflow 当后端 |
| **通用** | `mlflow-agent` | 通用分发器（不确定用哪个时路由）|
| | `fix-agent-issue` | 修 agent 行为（探索→计划→实现→验证闭环）|

### 怎么用

**最简单（所有 AI 助手通用）**：直接让 AI 助手读 SKILL.md 再做事：
> 先读 `mlflow_skills/classical-ml/SKILL.md`，然后按里面 Step 1 帮我给这个 sklearn 训练加追踪。

**配置自动触发**：按你所用 AI 助手的 rules / skills 机制，把对应 skill 目录加进去。详见 [`mlflow_skills/README.md`](mlflow_skills/README.md)。

**可选 Auto-Suggestion Hook**：`mlflow_skills/hooks/` 里有 `UserPromptSubmit` hook，自动检测你的 prompt 匹配哪个 skill 并提示。安装方法见 [`mlflow_skills/hooks/README.md`](mlflow_skills/hooks/README.md)。

> 💡 完整 vibecoding 教程见 [`notes/00_quickstart.md`](notes/00_quickstart.md) 的 **Chapter 7** 和 **Skill 段**。

---

## 推荐阅读顺序

1. 📌 **[`notes/00_quickstart.md`](notes/00_quickstart.md)** — 作业指导书（新人第一站）
2. 📌 **[`notes/appendix/mlflow3_breaking_changes.md`](notes/appendix/mlflow3_breaking_changes.md)** — MLflow 3 vs 2 关键变化（必读，影响所有示例代码）
3. 按 phase 顺序跑脚本 + 看对应笔记：
   - 跑 `scripts/01_basics/01_hello_mlflow.py` → 看 [notes/01_basics.md](notes/01_basics.md)
   - 跑 `scripts/02_registry/02a_log_model.py` → 看 [notes/02_registry.md](notes/02_registry.md)
   - ...以此类推
4. 跑完前 4 阶段后填 `.env`（参考下方"API Key 配置"），继续 5-9 阶段
5. **[notes/appendix/api_cheatsheet.md](notes/appendix/api_cheatsheet.md)** — 用到 API 时速查
6. 🎓 跑 `scripts/capstone/capstone_support_pilot.py` → 看 [notes/capstone.md](notes/capstone.md)

---

## 环境

### 一键创建（推荐）

```bash
# conda 方式（自动建好 env + 装全所有依赖）
conda env create -f environment.yml
conda activate mlflow

# 或 pip 方式
pip install -r requirements.txt
```

验证：
```bash
mlflow --version  # mlflow, version 3.15.1
```

### 依赖清单

| 文件 | 用途 |
|------|------|
| `requirements.txt` | Phase 1-9 + capstone 主依赖 |
| `requirements-phase10.txt` | 深度学习图像分类额外依赖（torch/timm）|
| `environment.yml` | conda 一键建环境（含上述全部）|

| 包 | 版本 | 用途 |
|----|------|------|
| mlflow | 3.15.1 | 主框架 |
| openai | 2.53.0 | OpenAI LLM 集成 |
| anthropic | 0.120.2 | Claude 集成 |
| langchain | 1.3.14 | LangChain Agent 集成 |
| gepa | 0.1.4 | Prompt Optimization（Phase 8 用）|
| pandas / numpy / scikit-learn | latest | 传统 ML 示例 |
| jupyter | latest | Notebook 体验 |
| **Phase 10 额外需要** | | |
| torch (CPU) | 2.13+ | 深度学习（9 个 CNN） |
| torchvision | 0.28+ | 数据集和 transforms |
| timm | 1.0+ | ResNet/EfficientNet/DenseNet 预训练模型 |

---

## API Key 配置

Phase 5 之后的 LLM 示例需要 API key：

```bash
cp .env.example .env
# 编辑 .env 填入你的 key
```

`.env.example` 提供 DeepSeek 模板（推荐国内直连）。也支持智谱 GLM、阿里云百炼、Moonshot、零一万物（都兼容 OpenAI 协议），或 OpenAI/Anthropic 官方。

`scripts/05_tracing/env_bootstrap.py` 会自动把国内服务商的 key 桥接到 `OPENAI_API_KEY` / `OPENAI_API_BASE`，让 `mlflow.openai.autolog()` 直接可用。

---

## 常用命令速查

```bash
# 启动 MLflow UI（最常用！）
mlflow ui --port 5000
# 浏览器打开 http://localhost:5000

# 启动生产级 Tracking Server
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns

# 部署模型
mlflow models serve -m "models:/MyModel@champion" -p 5001

# 数据库升级（首次用 Registry 必须跑）
mlflow db upgrade sqlite:///mlflow.db
```

更多命令见 [`notes/00_quickstart.md`](notes/00_quickstart.md) 的 **Chapter 14 速查表**。

---

## 文档链接

**官方中文文档**：
- 传统 ML：https://mlflow.org.cn/docs/latest/ml/
- LLM/Agent：https://mlflow.org.cn/docs/latest/genai/
- MLflow 3 迁移：https://mlflow.org.cn/docs/latest/ml/mlflow-3/

**官方英文文档**：https://mlflow.org/docs/latest/

**MLflow Skills**：https://github.com/mlflow/skills

---

## 学习约定

- 每个 phase 一个目录，配一个同名笔记（`notes/<NN>_<name>.md`）
- 不直接对应脚本的参考文档放 `notes/appendix/`
- 跑完脚本必看 UI 效果
- 遇到 API 不一致：先查 [notes/appendix/mlflow3_breaking_changes.md](notes/appendix/mlflow3_breaking_changes.md)
- 找 API 用法：查 [notes/appendix/api_cheatsheet.md](notes/appendix/api_cheatsheet.md)
- vibecoding 时：让 AI 助手读 `mlflow_skills/<name>/SKILL.md` 再操作
