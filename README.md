# MLflow 3 学习项目

> 从零开始学 **MLflow 3**（2025-2026 主流版），同时覆盖传统 ML 和 LLM/Agent 工作流
> 基于官方中文文档站系统性调研：`mlflow.org.cn/docs/latest/ml/` + `mlflow.org.cn/docs/latest/genai/`

## 学习入口

📌 **从 [notes/roadmap.md](notes/roadmap.md) 开始**——完整 9 阶段路线图（带代码模式、动手任务、官方文档链接）

辅助文档：
- **[notes/api_cheatsheet.md](notes/api_cheatsheet.md)**——MLflow 3 全量 API 速查（按用途分类）
- **[notes/mlflow3_breaking_changes.md](notes/mlflow3_breaking_changes.md)**——MLflow 3 vs 2 的关键变化（必读）
- **[notes/phase1_basics.md](notes/phase1_basics.md)**——阶段 1 的核心概念笔记

## 9 阶段速览

| # | 主题 | 关键能力 | 目录 | 进度 |
|---|------|----------|------|------|
| 1 | 入门与追踪基础 | Experiment/Run/Autolog/手动记录 | `01_basics/` | ✅ |
| 2 | 模型格式与注册表 | MLmodel + 签名 + Registry + Aliases | `02_tracking/`, `03_registry/` | ⏳ |
| 3 | 追踪服务器与数据集血缘 | sqlite backend + 数据集血缘 + MLflow 3 LoggedModel 搜索 | `02_tracking/` | ⏳ |
| 4 | 评估、服务与 ML 验证 | `mlflow.models.evaluate` + 自定义指标 + `models serve` | `02_tracking/` | ⏳ |
| 5 | **GenAI 追踪**（重点） | OpenAI/Anthropic autolog + `@mlflow.trace` | `04_tracing/` | ⏳ |
| 6 | Prompt Registry + Flavors | 注册/版本化提示词 + 别名 + LangChain | `05_prompts/` | ⏳ |
| 7 | GenAI 评估 + 自定义 Scorer | `mlflow.genai.evaluate` + `@scorer` + LLM-as-judge | `06_evaluation/` | ⏳ |
| 8 | 版本追踪 + 提示词优化 + ResponsesAgent | `set_active_model` + git 集成 + GepaOptimizer + ResponsesAgent | `07_agents/` | ⏳ |
| 9 | 部署 + 生产可观测性 | Docker + Postgres+MinIO + 采样/PII + AI Insights + Agent Server | `08_project/` | ⏳ |
| 🎓 | **SupportPilot 毕业项目** | 端到端集成全部 MLflow 3 能力 | `08_project/` | ⏳ |

**设计理念**：先用 scikit-learn（无需 API key）跑通 MLflow 核心概念 → 再用同一套概念套到 LLM/Agent → 毕业项目串联全能力。

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
| pandas / numpy / scikit-learn | latest | 传统 ML 示例 |
| jupyter | latest | Notebook 体验 |

## API Key 配置

阶段 5 之后的 LLM 示例需要 API key：

```bash
cp .env.example .env
# 编辑 .env 填入你的 key
```

支持：OpenAI、Anthropic Claude，或任意 OpenAI 兼容端点（DeepSeek、智谱等）。详见 `.env.example`。

## 常用命令速查

```bash
# 启动 MLflow UI（最常用！）
mlflow ui --port 5000
# 浏览器打开 http://localhost:5000

# 启动生产级 Tracking Server
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns

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

- 每阶段先看 `notes/roadmap.md` 对应章节的「必学代码模式」
- 跑完脚本必看 UI 效果
- 遇到 API 不一致：先查 `notes/mlflow3_breaking_changes.md`
- 找 API 用法：查 `notes/api_cheatsheet.md`