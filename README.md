# MLflow 3 学习项目

> 从零开始学 **MLflow 3**（2025-2026 主流版），同时覆盖传统 ML 和 LLM/Agent 工作流
> 基于官方中文文档站系统性调研：`mlflow.org.cn/docs/latest/ml/` + `mlflow.org.cn/docs/latest/genai/`

## 目录结构

```
MLFlowLearning/
├── README.md                 本文件
├── .env.example              API Key 模板（国内 LLM 服务商）
├── notes/                    学习笔记
│   ├── 01_basics.md          ↔ 01_basics/        （Phase 1）
│   ├── 02_registry.md        ↔ 02_registry/      （Phase 2）
│   ├── 03_tracking.md        ↔ 03_tracking/      （Phase 3）
│   ├── 04_evaluate.md        ↔ 04_evaluate/      （Phase 4）
│   ├── 05_tracing.md         ↔ 05_tracing/       （Phase 5）
│   ├── 06_prompts.md         ↔ 06_prompts/       （Phase 6）
│   ├── 07_evaluation.md      ↔ 07_evaluation/    （Phase 7）
│   ├── 08_agents.md          ↔ 08_agents/        （Phase 8）
│   ├── 09_deployment.md      ↔ 09_deployment/    （Phase 9）
│   ├── capstone.md           ↔ capstone/         （毕业项目）
│   └── appendix/             不直接对应脚本的参考文档
│       ├── roadmap.md              9 阶段路线图
│       ├── api_cheatsheet.md       MLflow 3 API 速查
│       └── mlflow3_breaking_changes.md  MLflow 3 vs 2 关键变化
├── 01_basics/                Phase 1 脚本：入门与追踪基础
├── 02_registry/              Phase 2 脚本：模型格式与注册表
├── 03_tracking/              Phase 3 脚本：追踪服务器与数据集血缘
├── 04_evaluate/              Phase 4 脚本：评估、服务与经典 ML 验证
├── 05_tracing/               Phase 5 脚本：GenAI 追踪（含 env_bootstrap.py）
├── 06_prompts/               Phase 6 脚本：Prompt Registry
├── 07_evaluation/            Phase 7 脚本：GenAI 评估
├── 08_agents/                Phase 8 脚本：版本追踪 + ResponsesAgent
├── 09_deployment/            Phase 9 脚本：部署与生产可观测性
└── capstone/                 毕业项目 SupportPilot
```

**约定**：每个 phase 一个目录，一个脚本目录对应一个同名笔记（`notes/<NN>_<name>.md`）。
不直接对应脚本的参考文档放 `notes/appendix/`。

## 9 阶段速览

| # | 主题 | 关键能力 | 脚本目录 | 笔记 | API Key |
|---|------|----------|---------|------|---------|
| 1 | 入门与追踪基础 | Experiment/Run/Autolog | `01_basics/` | [01_basics](notes/01_basics.md) | ❌ |
| 2 | 模型格式与注册表 | MLmodel + 签名 + Registry + Aliases | `02_registry/` | [02_registry](notes/02_registry.md) | ❌ |
| 3 | 追踪服务器与数据集血缘 | sqlite backend + `mlflow.data` + `search_logged_models` | `03_tracking/` | [03_tracking](notes/03_tracking.md) | ❌ |
| 4 | 评估、服务与 ML 验证 | `mlflow.models.evaluate` + `validate_evaluation_results` + `models serve` | `04_evaluate/` | [04_evaluate](notes/04_evaluate.md) | ❌ |
| 5 | **GenAI 追踪**（重点） | OpenAI/Anthropic autolog + `@mlflow.trace` + 元数据 | `05_tracing/` | [05_tracing](notes/05_tracing.md) | ✅ |
| 6 | Prompt Registry + Flavors | 注册/版本化 + alias + LangChain autolog | `06_prompts/` | [06_prompts](notes/06_prompts.md) | ✅ |
| 7 | GenAI 评估 + 自定义 Scorer | `mlflow.genai.evaluate` + `@scorer` + `make_judge` | `07_evaluation/` | [07_evaluation](notes/07_evaluation.md) | ✅ |
| 8 | 版本追踪 + 提示词优化 + ResponsesAgent | `set_active_model` + GepaOptimizer + ResponsesAgent | `08_agents/` | [08_agents](notes/08_agents.md) | ✅ |
| 9 | 部署 + 生产可观测性 | trace 采样 + PII 脱敏 + 生产部署参考 | `09_deployment/` | [09_deployment](notes/09_deployment.md) | ✅ |
| 🎓 | **SupportPilot** 毕业项目 | sklearn gate + LangChain RAG + Prompt Registry + GenAI eval | `capstone/` | [capstone](notes/capstone.md) | ✅ |

**设计理念**：先用 scikit-learn（无需 API key）跑通 MLflow 核心概念 → 再用同一套概念套到 LLM/Agent → 毕业项目串联全能力。

## 推荐阅读顺序

1. 📌 **[notes/appendix/roadmap.md](notes/appendix/roadmap.md)** — 9 阶段完整路线图（带代码模式、动手任务、官方文档链接）
2. 📌 **[notes/appendix/mlflow3_breaking_changes.md](notes/appendix/mlflow3_breaking_changes.md)** — MLflow 3 vs 2 关键变化（必读，影响所有示例代码）
3. 按 phase 顺序跑脚本 + 看对应笔记：
   - 跑 `01_basics/01_hello_mlflow.py` → 看 [notes/01_basics.md](notes/01_basics.md)
   - 跑 `02_registry/02a_log_model.py` → 看 [notes/02_registry.md](notes/02_registry.md)
   - ...以此类推
4. 跑完前 4 阶段后填 `.env`（参考下方"API Key 配置"），继续 5-9 阶段
5. **[notes/appendix/api_cheatsheet.md](notes/appendix/api_cheatsheet.md)** — 用到 API 时速查
6. 🎓 跑 `capstone/capstone_support_pilot.py` → 看 [notes/capstone.md](notes/capstone.md)

## 环境

```bash
conda activate mlflow
mlflow --version  # mlflow, version 3.15.1
```

| 包 | 版本 | 用途 |
|----|------|------|
| mlflow | 3.15.1 | 主框架 |
| openai | 2.53.0 | OpenAI LLM 集成 |
| anthropic | 0.120.2 | Claude 集成 |
| langchain | 1.3.14 | LangChain Agent 集成 |
| gepa | 0.1.4 | Prompt Optimization（Phase 8 用）|
| pandas / numpy / scikit-learn | latest | 传统 ML 示例 |
| jupyter | latest | Notebook 体验 |

## API Key 配置

Phase 5 之后的 LLM 示例需要 API key：

```bash
cp .env.example .env
# 编辑 .env 填入你的 key
```

支持：DeepSeek（推荐）、智谱 GLM、阿里云百炼、Moonshot、零一万物（国内 OpenAI 兼容），或 OpenAI/Anthropic 官方。详见 `.env.example`。

`05_tracing/env_bootstrap.py` 会自动把国内服务商的 key 桥接到 `OPENAI_API_KEY` / `OPENAI_API_BASE`，让 `mlflow.openai.autolog()` 直接可用。

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

# 搜索 MLflow 文档
python .claude/skills/anysearch/scripts/anysearch_cli.py search "MLflow prompt registry" 5
```

## 文档链接

**官方中文文档**：
- 传统 ML：https://mlflow.org.cn/docs/latest/ml/
- LLM/Agent：https://mlflow.org.cn/docs/latest/genai/
- MLflow 3 迁移：https://mlflow.org.cn/docs/latest/ml/mlflow-3/

**官方英文文档**：https://mlflow.org/docs/latest/

## 学习约定

- 每个 phase 一个目录，配一个同名笔记（`notes/<NN>_<name>.md`）
- 不直接对应脚本的参考文档放 `notes/appendix/`
- 跑完脚本必看 UI 效果
- 遇到 API 不一致：先查 [notes/appendix/mlflow3_breaking_changes.md](notes/appendix/mlflow3_breaking_changes.md)
- 找 API 用法：查 [notes/appendix/api_cheatsheet.md](notes/appendix/api_cheatsheet.md)