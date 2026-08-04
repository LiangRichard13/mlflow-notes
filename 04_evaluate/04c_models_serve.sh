#!/bin/bash
# 阶段 4 示例 3: mlflow models serve - 本地部署模型为 REST API
# ======================================================

# ============ 1. 启动 mlflow server（一个终端）============
# 注意：models serve 命令需要 server 模式运行，不能纯本地文件
# 用 sqlite 后端即可
mlflow server \
  --backend-store-uri sqlite:///$(pwd)/mlflow.db \
  --default-artifact-root $(pwd)/mlruns \
  --host 0.0.0.0 \
  --port 5000

# ============ 2. 注册并标记 champion（先跑过 02a/02b 之后）============
# python 03_registry/02a_log_model.py
# python 03_registry/02b_register_alias.py

# ============ 3. 在另一个终端启动 models serve ============
# 部署 champion 模型到本地 5001 端口
mlflow models serve \
  -m "models:/WineQualityClassifier@champion" \
  -p 5001

# ============ 4. curl 调 /invocations ============

# 4.1 JSON 格式（推荐）
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

# 4.2 CSV 格式
# 先把一行数据写到文件
echo "alcohol,malic_acid,ash,alcalinity_of_ash,magnesium,total_phenols,flavanoids,nonflavanoid_phenols,proanthocyanins,color_intensity,hue,od280/od315_of_diluted_wines,proline
13.0,1.5,2.5,19.0,100,2.8,3.0,0.3,1.8,5.0,1.0,3.0,1000" > /tmp/wine_sample.csv

curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: text/csv" \
  --data-binary @/tmp/wine_sample.csv

# 4.3 Tensor Input（如果有图像等）
# mlflow models serve 自动支持，根据模型签名选择格式

# ============ 5. 看容器化部署（生产）============
# 把模型打成 Docker 镜像
mlflow models build-docker \
  -m "models:/WineQualityClassifier@champion" \
  -n "wine-classifier"

# 跑容器
docker run -p 5001:8080 wine-classifier

# ============ 6. 不用 server，纯命令行推理 ============
# mlflow models predict -m <uri> -i <input.csv> -o <output.csv>