"""配置管理：从环境变量加载 NLU 模块参数。

支持从项目根目录或 config/ 目录加载 .env 文件。
优先使用 config/.env（用户实际配置），回退到根目录 .env。
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from dotenv import load_dotenv

# 按优先级加载 .env: config/.env > .env
_project_root = Path(__file__).resolve().parent.parent
_config_env = _project_root / "config" / ".env"
_root_env = _project_root / ".env"

if _config_env.exists():
    load_dotenv(_config_env)
elif _root_env.exists():
    load_dotenv(_root_env)


@dataclass
class NLUConfig:
    """NLU 模块配置，全部从环境变量读取。

    环境变量名兼容您的 config/.env 格式：
    - API_KEY: API 密钥
    - BASE_URL: API 地址
    - MODEL: 模型名称
    """

    api_key: str = field(
        default_factory=lambda: os.getenv("API_KEY", "")
    )
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    )
    model: str = field(
        default_factory=lambda: os.getenv("MODEL", "qwen-turbo")
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("NLU_MAX_TOKENS", "256"))
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("NLU_TEMPERATURE", "0.0"))
    )
    timeout: int = field(
        default_factory=lambda: int(os.getenv("NLU_TIMEOUT", "30"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("NLU_MAX_RETRIES", "3"))
    )


# 全局默认配置实例
nlu_config = NLUConfig()
