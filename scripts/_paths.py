"""
统一的 MLflow tracking URI 引导模块
====================================

问题
----
scripts/ 下的所有 phase 脚本都需要一个 MLflow backend store。
本项目默认用 ``sqlite:///<project-root>/mlflow.db``。

如果脚本直接写：
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

MLflow 会把它解析成 **当前进程 cwd 下的 mlflow.db**——而很多人从 IDE
里直接运行脚本（cwd = scripts/02_registry/），结果 db 就落到了子目录里，
根目录的 ``mlflow ui`` 看不到。

修复
----
本模块提供 ``setup_tracking()``，它通过 ``__file__`` 算出 **项目根目录**的
绝对路径，然后设成 ``sqlite:///<project-root>/mlflow.db``。
这样无论你在哪儿跑脚本，tracking URI 都指向同一个 db。

优先级
------
1. 环境变量 ``MLFLOW_TRACKING_URI`` 已设置（用户想连远程 server / Postgres）→ 不覆盖
2. 当前 tracking URI 是 ``http://`` 或 ``https://``（连 server）→ 不覆盖
3. 否则 → 设为项目根下的 ``mlflow.db`` 绝对路径

用法
----
在每个 phase 脚本顶部，``import mlflow`` 之后第一行：

    import _paths
    _paths.setup_tracking()

或者直接 ``import _paths`` 一行——模块导入时会自动跑 setup_tracking()（对
没有环境变量的"普通用户"来说最方便；有自定义环境变量的用户调用显式
函数时也更清楚）。

返回最终的 tracking URI，方便脚本打印给用户看：
    uri = _paths.setup_tracking()
"""

import os
from pathlib import Path
import mlflow

# scripts/_paths.py → scripts/ → <project-root>
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "mlflow.db"
DEFAULT_TRACKING_URI = f"sqlite:///{DEFAULT_DB_PATH}"


def setup_tracking() -> str:
    """
    设置 MLflow tracking URI 到项目根的 mlflow.db（除非用户已通过环境变量指定）。

    返回最终的 tracking URI。
    """
    # 1. 用户已经显式设了环境变量 → 完全尊重
    if "MLFLOW_TRACKING_URI" in os.environ:
        return os.environ["MLFLOW_TRACKING_URI"]

    # 2. 当前已经在连 HTTP server（03a_start_server.sh 等场景）→ 不覆盖
    current = mlflow.get_tracking_uri()
    if current.startswith(("http://", "https://")):
        return current

    # 3. 否则强制指向项目根的 mlflow.db
    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
    return DEFAULT_TRACKING_URI


def project_root() -> Path:
    """返回项目根目录的绝对路径，方便其他需要锚定到项目根的脚本使用。"""
    return PROJECT_ROOT


# 模块导入时自动执行——和 env_bootstrap.py 风格一致
# （已经显式设了环境变量或在连 server 的脚本不会受影响）
setup_tracking()