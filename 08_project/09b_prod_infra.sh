#!/bin/bash
# 阶段 9 示例 2: 生产级 MLflow 部署配置（参考）
# =================================================
#
# 本地无法演示完整生产部署，但提供完整的参考配置。
# 需要 Docker / PostgreSQL / S3（或 MinIO）。

# ====================================================
# 1. Docker Compose: MLflow + Postgres + MinIO 一键起
# ====================================================
# 单独建一个 docker-compose.yml 文件，内容如下：

: <<'YAML'
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
    ports:
      - "5000:5000"

volumes:
  pgdata:
  miniodata:
YAML

# 启动：
#   docker compose up -d
# 访问 http://localhost:5000

# ====================================================
# 2. 客户端连生产 server
# ====================================================
# 设环境变量
# export MLFLOW_TRACKING_URI=http://localhost:5000

# 或者代码里设
# mlflow.set_tracking_uri("http://mlflow.yourcompany.com:5000")

# ====================================================
# 3. 容器化模型（任何 MLflow 模型）
# ====================================================
# mlflow models build-docker \
#   -m models:/my-model@champion \
#   -n my-model:v1

# 跑容器
# docker run -p 5001:8080 my-model:v1

# ====================================================
# 4. 生产 checklist
# ====================================================
: <<'CHECKLIST'
□ Backend store: PostgreSQL (不要用 SQLite)
□ Artifact store: S3 / MinIO (不要用本地文件系统)
□ MLflow 3.5+: 配置 --allowed-hosts 防 DNS rebinding
□ 配置 HTTPS (nginx 反向代理)
□ 启用 trace 采样（10-20%）
□ 启用 PII 脱敏
□ 设置 retention policy（定期清理过期数据）
  mlflow gc --backend-store-uri $POSTGRES_URI
□ 配置 backup（Postgres + S3 cross-region）
□ 监控 MLflow server 自身健康（liveness probe）
□ 注册表权限控制（mlflow 3.5+ 有 auth）
CHECKLIST

echo "📋 部署清单：参考 docker-compose.yml + 上面的 checklist"