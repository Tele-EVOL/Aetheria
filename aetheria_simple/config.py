from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv


def _env_str(name: str, default: Optional[str] = None) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default if default is not None else ""
    value = raw.strip()
    return value if value else (default if default is not None else "")


def _env_required(primary: str, *fallbacks: str, description: str) -> str:
    for key in (primary, *fallbacks):
        value = _env_str(key)
        if value:
            return value
    candidates = ", ".join((primary, *fallbacks))
    raise ValueError(f"缺少必要的环境变量({description}): {candidates}")


def _detect_provider(azure_endpoint: str, openai_key: str) -> str:
    explicit = _env_str("AETHERIA_SIMPLE_PROVIDER") or _env_str("AETHERIA_PROVIDER")
    explicit = explicit.lower()
    if explicit in {"azure", "openai"}:
        return explicit

    if azure_endpoint:
        return "azure"
    if openai_key:
        return "openai"

    raise ValueError(
        "未找到 Azure 或 OpenAI 配置，请设置 AETHERIA_SIMPLE_AZURE_ENDPOINT "
        "+ AETHERIA_SIMPLE_API_KEY，或设置 OPENAI_API_KEY。"
    )


def _load_deployment_map(provider: str) -> Dict[str, str]:
    defaults = {
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini"
        if provider == "azure"
        else "gpt-4o-mini",
        "gpt-4.1": "gpt-4.1",
    }

    raw_map = _env_str("AETHERIA_SIMPLE_DEPLOYMENT_MAP") or _env_str(
        "AZURE_DEPLOYMENT_MAP"
    )
    if raw_map:
        try:
            data = json.loads(raw_map)
            if isinstance(data, dict):
                defaults.update({str(k): str(v) for k, v in data.items()})
        except json.JSONDecodeError:
            print("[aetheria_simple] 部署映射解析失败，使用默认映射。")

    for model in list(defaults.keys()):
        env_key = f"AETHERIA_SIMPLE_DEPLOYMENT_{model.upper().replace('-', '_')}"
        override = _env_str(env_key)
        if override:
            defaults[model] = override
    return defaults


def _default_path(*parts: str) -> str:
    base_dir = Path(__file__).resolve().parent
    return str(base_dir.joinpath(*parts))


load_dotenv()

AZURE_ENDPOINT = _env_str("AETHERIA_SIMPLE_AZURE_ENDPOINT", _env_str("AZURE_ENDPOINT"))
AZURE_API_KEY = _env_str("AETHERIA_SIMPLE_API_KEY", _env_str("API_KEY", _env_str("AZURE_API_KEY")))
OPENAI_API_KEY = _env_str("AETHERIA_SIMPLE_OPENAI_API_KEY", _env_str("OPENAI_API_KEY"))
OPENAI_BASE = _env_str("AETHERIA_SIMPLE_OPENAI_BASE", _env_str("OPENAI_BASE"))

PROVIDER = _detect_provider(AZURE_ENDPOINT, OPENAI_API_KEY)
USING_AZURE = PROVIDER == "azure"

if USING_AZURE:
    API_KEY = _env_required(
        "AETHERIA_SIMPLE_API_KEY", "API_KEY", "AZURE_API_KEY", description="Azure API Key"
    )
    API_VERSION = _env_str(
        "AETHERIA_SIMPLE_API_VERSION", _env_str("AZURE_API_VERSION", "2024-12-01-preview")
    )
else:
    API_KEY = _env_required(
        "AETHERIA_SIMPLE_OPENAI_API_KEY", "OPENAI_API_KEY", description="OpenAI API Key"
    )
    API_VERSION = _env_str("AETHERIA_SIMPLE_API_VERSION", _env_str("OPENAI_API_VERSION", ""))
    # Avoid leaking a stale Azure endpoint when Azure is not selected
    AZURE_ENDPOINT = ""

DEPLOYMENT_MAP = _load_deployment_map(PROVIDER)
# Backwards-compatible alias for older code paths
AZURE_DEPLOYMENT_MAP = DEPLOYMENT_MAP

EMBEDDING_MODEL_NAME = _env_str(
    "AETHERIA_SIMPLE_EMBEDDING_MODEL", "text-embedding-3-large"
)
CASE_LIBRARY_PATH = _env_str(
    "AETHERIA_SIMPLE_CASE_LIBRARY_PATH",
    _default_path("case_libraries", "default_case_library.json"),
)
CHROMA_PERSIST_DIR = _env_str(
    "AETHERIA_SIMPLE_CHROMA_PERSIST_DIR", _default_path("chroma_db")
)
CHROMA_COLLECTION_NAME = _env_str(
    "AETHERIA_SIMPLE_CHROMA_COLLECTION", "usb_only_img_case_library"
)


@dataclass
class SimpleRAGConfig:
    """Simple retrieval settings."""

    top_k: int = 3
    collection_name: str = CHROMA_COLLECTION_NAME


@dataclass
class SimpleModelConfig:
    """Models used by each role."""

    supporter: Optional[str] = "gpt-4o-mini"
    strict: Optional[str] = "gpt-4o-mini"
    loose: Optional[str] = "gpt-4o-mini"
    arbiter: Optional[str] = "gpt-4o"

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {
            "Supporter": self.supporter,
            "Strict Debater": self.strict,
            "Loose Debater": self.loose,
            "Holistic Arbiter": self.arbiter,
        }


@dataclass
class SimpleRunConfig:
    """Top-level configuration object for the simplified evaluator."""

    debate_rounds: int = 2
    rag: SimpleRAGConfig = field(default_factory=SimpleRAGConfig)
    models: SimpleModelConfig = field(default_factory=SimpleModelConfig)
    strict_bias: float = 0.6  # Used when computing weighted panel scores
    prompt_profile: str = "default"
    use_supporter: bool = True
    use_strict_debater: bool = True
    use_loose_debater: bool = True


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def load_config() -> SimpleRunConfig:
    """Load the runtime configuration with optional environment overrides."""

    rag_top_k = _env_int("AETHERIA_SIMPLE_RAG_TOP_K", 3)
    debate_rounds = _env_int("AETHERIA_SIMPLE_ROUNDS", 2)
    rag_collection = _env_str(
        "AETHERIA_SIMPLE_RAG_COLLECTION", CHROMA_COLLECTION_NAME
    )

    models = SimpleModelConfig(
        supporter=_env_str("AETHERIA_SIMPLE_SUPPORTER_MODEL", "gpt-4o-mini"),
        strict=_env_str("AETHERIA_SIMPLE_STRICT_MODEL", "gpt-4o-mini"),
        loose=_env_str("AETHERIA_SIMPLE_LOOSE_MODEL", "gpt-4o-mini"),
        arbiter=_env_str("AETHERIA_SIMPLE_ARBITER_MODEL", "gpt-4o"),
    )

    use_supporter = _env_bool("AETHERIA_SIMPLE_ENABLE_SUPPORTER", True)
    use_strict = _env_bool("AETHERIA_SIMPLE_ENABLE_STRICT", True) 
    use_loose = _env_bool("AETHERIA_SIMPLE_ENABLE_LOOSE", True)

    config = SimpleRunConfig(
        debate_rounds=max(1, debate_rounds),
        rag=SimpleRAGConfig(
            top_k=max(1, rag_top_k),
            collection_name=rag_collection,
        ),
        models=models,
        prompt_profile=_env_str("AETHERIA_SIMPLE_PROMPT_PROFILE", "default"),
        use_supporter=use_supporter,
        use_strict_debater=use_strict,
        use_loose_debater=use_loose,
    )
    return config


DEFAULT_CONFIG = load_config()

BUILD_DATABASE_BATCH_SIZE = max(
    1, _env_int("AETHERIA_SIMPLE_BUILD_DB_BATCH", 100)
)
