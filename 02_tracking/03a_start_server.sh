#!/bin/bash
# 阶段 3 - 启动生产级 Tracking Server
# ===================================
#
# 三种启动方式（按场景选）：

# ============ 方式 1: 纯本地（开发用，最简） ============
# SQLite 后端 + 本地文件构件库
# 注意：只能用 mlflow.* API，UI 通过 mlflow ui 另开
# conda activate mlflow
# python your_script.py  # 不需要启动 server

# ============ 方式 2: Tracking Server + SQLite（推荐学习用）============
# 启 server（一个终端），跑代码（另一个终端）
# server 启完后所有 mlflow.* API 自动走 HTTP
mlflow server \
  --backend-store-uri sqlite:///$(pwd)/mlflow.db \
  --default-artifact-root $(pwd)/mlruns \
  --host 0.0.0.0 \
  --port 5000

# ⚠️ MLflow 3.5+ 必须配 --allowed-hosts 防 DNS rebinding
# mlflow server ... --allowed-hosts "localhost,127.0.0.1"

# 启完后浏览器开 http://localhost:5000

# ============ 方式 3: PostgreSQL + S3/MinIO（生产用）============
# mlflow server \
#   --backend-store-uri postgresql://user:pass@host:5432/mlflowdb \
#   --default-artifact-root s3://my-bucket/mlflow-artifacts \
#   --host 0.0.0.0 --port 5000

# ============ 启用 Model Registry 必须先跑：============
# 第一次用 sqlite backend + Registry 时，跑一次升级
mlflow db upgrade sqlite:///$(pwd)/mlflow.db

# ============ 清理过期数据 ============
# 删除已删除/未记录的元数据
# mlflow gc --backend-store-uri sqlite:///$(pwd)/mlflow.db

# ============ 客户端连远端 server ============
# 设环境变量或代码里 set_tracking_uri
# export MLFLOW_TRACKING_URI=http://localhost:5000
# 或
# import mlflow; mlflow.set_tracking_uri("http://localhost:5000")