"""Shared test configuration.

Spider and BIRD are large downloads and are gitignored, so tests that need them
skip by default. That is convenient but dishonest on a machine where the data is
supposed to be present, so setting TEXT2SQL_REQUIRE_DATASETS=1 turns those skips
into failures and gives an accurate coverage picture.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.config import get_settings


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "requires_spider: needs the Spider dataset to be downloaded"
    )
    config.addinivalue_line(
        "markers", "requires_bird: needs the BIRD dev dataset to be downloaded"
    )


def _datasets_required() -> bool:
    return os.getenv("TEXT2SQL_REQUIRE_DATASETS") == "1"


def missing_dataset(path: Path, name: str) -> None:
    """Skip normally, but fail when datasets are declared to be present."""
    message = f"{name} is not present at {path}"
    if _datasets_required():
        pytest.fail(f"{message} (TEXT2SQL_REQUIRE_DATASETS=1)")
    pytest.skip(message)


@pytest.fixture(scope="session")
def spider_root() -> Path:
    return get_settings().spider_root


@pytest.fixture
def spider_db(spider_root: Path) -> str:
    """Path to the concert_singer database, or skip/fail if Spider is absent."""
    path = spider_root / "database" / "concert_singer" / "concert_singer.sqlite"
    if not path.exists():
        missing_dataset(path, "Spider concert_singer database")
    return str(path)
