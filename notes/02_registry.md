# 阶段 2 学习笔记：模型格式与注册表

> 对应脚本：`02_registry/02a_log_model.py`、`02b_register_alias.py`、`02c_load_predict.py`

## 一、MLflow Model 格式

一个 MLflow Model 本质上是一个**目录**：

```
my-model/
├── MLmodel              # 主元数据（YAML）
├── model.pkl            # 序列化后的模型
├── python_env.yaml      # Python 依赖（conda 环境）
├── conda.yaml           # Conda 环境
├── requirements.txt     # pip 依赖
└── signature.pkl        # 输入输出 schema（可选）
```

`MLmodel` 文件是关键：

```yaml
artifact_path: classifier_model
flavors:
  python_function:
    env: conda.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    predict_fn: predict
  sklearn:
    code: null
    pickled_model: model.pkl
    serialization_format: cloudpickle
mlflow_version: 3.15.1
model_size_bytes: 1234
run_id: abc123
```

## 二、签名（Signature）

签名让部署服务能**自动校验请求格式**：

```python
from mlflow.models import infer_signature
signature = infer_signature(X_train, model.predict(X_train))

# 推断的签名包含：
# - inputs: 列名 + 类型
# - outputs: 列名 + 类型
mlflow.sklearn.log_model(model, name="model", signature=signature, input_example=X_train.head(3))
```

## 三、注册模型

```python
from mlflow import MlflowClient

# 注册（同名注册会创建新版本）
result = mlflow.register_model(
    f"runs:/{run.info.run_id}/model",
    "MyModel",
)
print(f"v{result.version}, source={result.source}")
```

## 四、别名（Alias，替代已废弃的 Stage）

```python
client = MlflowClient()

# 设别名（可同时多个）
client.set_registered_model_alias("MyModel", "champion", version=1)
client.set_registered_model_alias("MyModel", "challenger", version=2)

# 别名可以"热切换"，所有加载方自动用新版本
# client.set_registered_model_alias("MyModel", "champion", version=2)

# 查询
client.get_registered_model("MyModel")  # 别名映射
```

**为什么用别名不用 stage**：
- Stage（Staging/Production）是 MLflow 2 的概念，**已被废弃**
- Alias 更灵活：一个版本可以有多个别名
- 切换别名是原子操作，所有引用立即生效

## 五、用别名加载模型

```python
# 三种 URI 方式
model = mlflow.sklearn.load_model("runs:/<run_id>/model")      # 从 Run
model = mlflow.sklearn.load_model("models:/MyModel@champion")  # 用别名（推荐）
model = mlflow.sklearn.load_model("models:/MyModel/3")          # 用版本号
```

加载后可直接用——sklearn Pipeline 的所有步骤（StandardScaler + RF）都还原。

## 六、完整工作流（02a/02b/02c 的串联）

```bash
python 02_registry/02a_log_model.py      # 训练 + log + 自动注册 Run
python 02_registry/02b_register_alias.py # 注册到 Registry + 设 champion 别名
python 02_registry/02c_load_predict.py   # 用别名加载 + 推理
```

## 七、关键避坑

1. **`set_registered_model_alias` 改名**：MLflow 2 用 `transition_model_version_stage`（**已废弃**）
2. **同名注册自动版本化**：v1, v2, v3...，但**version 不会因为删除而复用**
3. **Registry 需要 backend store**：纯文件系统模式不支持，必须 sqlite/postgres
4. **model_uri 的几种形式**：
   - `runs:/<run_id>/<artifact_path>`
   - `models:/<name>@<alias>`  ← 推荐
   - `models:/<name>/<version>`