from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    prompt_dir: Path = PROJECT_ROOT / "prompts"
    uploads_dir: Path = PROJECT_ROOT / "data" / "uploads"
    exports_dir: Path = PROJECT_ROOT / "data" / "exports"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"
    sample_db_path: Path = PROJECT_ROOT / "data" / "sample.db"
    sample_questions_path: Path = PROJECT_ROOT / "data" / "sample_questions.json"
    spider_root: Path = PROJECT_ROOT / "data" / "spider_data" / "spider_data"
    bird_root: Path = PROJECT_ROOT / "data" / "dev" / "dev_20240627"
    few_shot_examples_path: Path = PROJECT_ROOT / "prompts" / "few_shot_examples.json"
    system_prompt_path: Path = PROJECT_ROOT / "prompts" / "system_prompt.txt"
    api_host: str = os.getenv("TEXT2SQL_API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("TEXT2SQL_API_PORT", "8000"))
    default_provider: str = os.getenv("TEXT2SQL_PROVIDER", "auto")
    default_model_name: str = os.getenv("TEXT2SQL_MODEL_NAME", "google/flan-t5-base")
    auto_spider_few_shots: bool = os.getenv("TEXT2SQL_AUTO_SPIDER_FEWSHOTS", "1") == "1"
    spider_few_shot_count: int = int(os.getenv("TEXT2SQL_SPIDER_FEWSHOT_COUNT", "6"))
    max_result_rows: int = int(os.getenv("TEXT2SQL_MAX_RESULT_ROWS", "200"))
    sql_timeout_seconds: float = float(os.getenv("TEXT2SQL_SQL_TIMEOUT_SECONDS", "10"))
    value_index_enabled: bool = os.getenv("TEXT2SQL_VALUE_INDEX_ENABLED", "1") == "1"
    value_index_max_rows: int = int(os.getenv("TEXT2SQL_VALUE_INDEX_MAX_ROWS", "2000"))
    log_level: str = os.getenv("TEXT2SQL_LOG_LEVEL", "INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.exports_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
