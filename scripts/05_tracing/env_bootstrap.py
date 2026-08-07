"""
环境变量引导：让 MLflow 能直接用国内 LLM 服务商
================================================

问题：MLflow 的 mlflow.openai.autolog() 默认只读 OPENAI_API_KEY / OPENAI_API_BASE
解决：在每个 Phase 5+ 脚本开头 import 这个模块，自动：
  1. 读 .env
  2. 把 DEEPSEEK_API_KEY / ZHIPU_API_KEY 等映射到 OPENAI_API_KEY
  3. 设置正确的 OPENAI_API_BASE
  4. 导出 MLflow 能识别的环境变量

用法：
  from env_bootstrap import setup_env
  setup_env()

或者直接 import 让副作用生效：
  import env_bootstrap   # 自动生效
"""

import os
from pathlib import Path
from dotenv import load_dotenv


# 优先级：环境变量 > .env 文件
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# 国内 LLM 服务商 → (env_key_for_api_key, env_key_for_base, default_base_url)
PROVIDERS = {
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
    "zhipu":    ("ZHIPU_API_KEY",    "ZHIPU_API_BASE",    "https://open.bigmodel.cn/api/paas/v4/"),
    "dashscope":("DASHSCOPE_API_KEY","DASHSCOPE_API_BASE","https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "moonshot": ("MOONSHOT_API_KEY", "MOONSHOT_API_BASE", "https://api.moonshot.cn/v1"),
    "yi":       ("YI_API_KEY",       "YI_API_BASE",       "https://api.lingyiwanwu.com/v1"),
}


def _ensure_v1_suffix(url: str) -> str:
    """
    规范化 base URL：如果 URL 不以 /v1 结尾，自动补上。

    为什么需要：DeepSeek / 智谱 / 通义千问 / 月之暗面 等国内服务商的
    OpenAI 兼容协议路径都带 /v1 前缀（例如 https://api.deepseek.com/v1/chat/completions）。
    用户在 .env 里写 https://api.deepseek.com 时，OpenAI SDK 会请求
    https://api.deepseek.com/chat/completions → Connection refused（路径不存在）。

    自动补 /v1 后用户拿到正确 URL，避免踩坑。
    """
    url = url.rstrip("/")  # 去尾部斜杠避免 //v1
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


def setup_env() -> None:
    """从 .env 加载 + 桥接国内服务商 → OPENAI_*"""

    # 1) 加载 .env
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=False)  # 不覆盖已有环境变量
        print(f"✓ 已加载 {ENV_FILE}")
    else:
        print(f"⚠️  {ENV_FILE} 不存在，请先 cp .env.example .env")

    # 2) 如果 OPENAI_API_KEY 已设，直接返回
    if os.getenv("OPENAI_API_KEY"):
        # 但确保 OPENAI_API_BASE 也对
        existing_base = os.getenv("OPENAI_API_BASE")
        if existing_base:
            # 即使 OPENAI_API_BASE 是用户自己设的，也走规范化（防漏 /v1）
            normalized = _ensure_v1_suffix(existing_base)
            if normalized != existing_base:
                print(f"⚠️  OPENAI_API_BASE 末尾缺 /v1，已自动修正：")
                print(f"    原值: {existing_base}")
                print(f"    修正: {normalized}")
                os.environ["OPENAI_API_BASE"] = normalized
        else:
            print("⚠️ OPENAI_API_KEY 已设，但缺 OPENAI_API_BASE，MLflow 会用官方端点")
        return

    # 3) 否则尝试从国内服务商桥接
    for provider_name, (key_env, base_env, default_base) in PROVIDERS.items():
        api_key = os.getenv(key_env)
        if api_key:
            raw_url = os.getenv(base_env, default_base)
            base_url = _ensure_v1_suffix(raw_url)
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_API_BASE"] = base_url
            print(f"✓ 桥接 {provider_name} → OPENAI_API_KEY / OPENAI_API_BASE")
            print(f"  base: {base_url}")
            if raw_url != base_url:
                print(f"  ℹ️  .env 里的 {base_env} 没带 /v1，已自动补上")
            return

    print("⚠️ 没找到任何 LLM API key，请检查 .env")


def get_default_model() -> str:
    """从环境变量读默认模型名"""
    return os.getenv("DEEPSEEK_MODEL", os.getenv("OPENAI_DEFAULT_MODEL", "deepseek-v4-flash"))


# 模块导入时自动执行一次
setup_env()